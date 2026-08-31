"""Migration 94C: football-only game environment / scoring-margin model.

M94B showed that explicit game-state mechanics improve team rushing volume, but
its pregame lead/neutral/trail forecasts remain weak. M94C isolates that upstream
problem. It predicts a football-only scoring environment first, then maps the
predicted margin environment into lead/neutral/trail play shares before applying
the same state-conditioned rushing mechanics used by M94B.

No sportsbook inputs are used. M91 projections are frozen pregame football
signals. M94B's cached 2023-2025 game-state history is reused so PBP is not
rebuilt on every research run.

Protocol:
- 2024 W1-12 development training
- 2024 W13-18 architecture holdout
- freeze margin/state architecture and blend
- refit on all 2024
- untouched 2025 validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_rb_explicit_game_state import (
    STATE_NAMES,
    _fit_predict_plays,
    _select_scalar_family,
    build_state_features,
)
from scripts.backtest.evaluate_rb_team_rush_volume import (
    PLAYER_KEYS,
    TEAM_KEYS,
    _legacy_guard,
    _metrics,
    _player_candidate,
    _rb_summary,
)

BLEND_GRID = (0.25, 0.50, 0.75, 1.00)
MARGIN_TARGETS = ("mean_score_diff", "final_observed_score_diff")


def _find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _models(seed: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=14.0)),
        ]),
        "gbr": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                random_state=seed, n_estimators=180, learning_rate=0.03,
                max_depth=2, min_samples_leaf=10, loss="huber",
            )),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=seed, n_estimators=400, max_depth=5,
                min_samples_leaf=8, max_features=0.7, n_jobs=-1,
            )),
        ]),
    }


def _projection_strength(pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frozen pregame player projections into team-strength signals."""
    p = pred.copy()
    p["market"] = p["market"].astype(str).str.lower()
    p["ml_proj"] = pd.to_numeric(p["ml_proj"], errors="coerce")
    aliases = {
        "proj_pass_yards": ["pass_yards", "passing_yards"],
        "proj_pass_att": ["pass_att", "passing_attempts"],
        "proj_rush_yards": ["rush_yards", "rushing_yards"],
        "proj_rush_att": ["rush_att", "rushing_attempts"],
        "proj_rec_yards": ["rec_yards", "receiving_yards"],
        "proj_receptions": ["receptions"],
    }
    base = p[TEAM_KEYS].drop_duplicates().copy()
    for out_col, names in aliases.items():
        g = p.loc[p["market"].isin(names)].copy()
        if g.empty:
            continue
        if out_col in {"proj_pass_yards", "proj_pass_att"}:
            # One primary QB should represent team passing expectation; max is
            # robust to backup rows while remaining pregame-only.
            agg = g.groupby(TEAM_KEYS)["ml_proj"].max().rename(out_col).reset_index()
        else:
            agg = g.groupby(TEAM_KEYS)["ml_proj"].sum(min_count=1).rename(out_col).reset_index()
        base = base.merge(agg, on=TEAM_KEYS, how="left", validate="one_to_one")
    return base


def _add_strength_edges(x: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    ps = _projection_strength(pred)
    out = out.merge(ps, on=TEAM_KEYS, how="left", validate="one_to_one")
    opp = ps.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in ps.columns if c not in TEAM_KEYS}})
    out = out.merge(opp, on=["season", "week", "opponent"], how="left", validate="one_to_one")

    for c in [c for c in ps.columns if c not in TEAM_KEYS]:
        oc = f"opp_{c}"
        if oc in out.columns:
            out[f"edge_{c}"] = pd.to_numeric(out[c], errors="coerce") - pd.to_numeric(out[oc], errors="coerce")

    # Explicit pregame strength/matchup edges already present in M94B features.
    pair_suffixes = [
        "mean_score_diff_avg1", "mean_score_diff_avg3", "mean_score_diff_avg5",
        "final_observed_score_diff_avg1", "final_observed_score_diff_avg3", "final_observed_score_diff_avg5",
        "won_observed_avg1", "won_observed_avg3", "won_observed_avg5",
        "success_rate_off_avg1", "success_rate_off_avg3", "success_rate_off_avg5",
        "success_rate_def_avg1", "success_rate_def_avg3", "success_rate_def_avg5",
        "pressure_rate_allowed_avg3", "pressure_rate_generated_avg3",
        "def_rush_epa_avg3", "def_pass_epa_avg3", "explosive_play_rate_allowed_avg3",
        "plays_est_avg3", "neutral_pace_avg3", "proe_avg3", "dropback_rate_avg3",
    ]
    for suffix in pair_suffixes:
        candidates = [
            (f"gs_team_{suffix}", f"gs_opp_{suffix}"),
            (f"team_{suffix}", f"opp_{suffix}"),
        ]
        for a, b in candidates:
            if a in out.columns and b in out.columns:
                out[f"env_edge_{a.replace('gs_team_', '').replace('team_', '')}"] = (
                    pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")
                )
                break
    return out


def _add_margin_labels(x: pd.DataFrame, state_hist: pd.DataFrame) -> pd.DataFrame:
    labels = state_hist[TEAM_KEYS + list(MARGIN_TARGETS)].copy()
    out = x.merge(labels, on=TEAM_KEYS, how="left", validate="one_to_one")
    if out[list(MARGIN_TARGETS)].isna().any().any():
        raise RuntimeError("M94C margin labels incomplete")
    return out


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "season", "week", "team", "opponent", "actual_team_rush_att",
        "actual_off_plays", "actual_rush_att_pbp", "lead_play_share",
        "neutral_play_share", "trail_play_share", *MARGIN_TARGETS,
        "pred_mean_margin", "pred_final_margin",
    }
    cols = []
    for c in frame.columns:
        if c in blocked:
            continue
        v = pd.to_numeric(frame[c], errors="coerce")
        if v.notna().any():
            cols.append(c)
    return sorted(cols)


def _select_margin_family(train: pd.DataFrame, hold: pd.DataFrame, features: list[str], target: str, seed: int) -> tuple[str, pd.DataFrame]:
    rows = []
    for name, model in _models(seed).items():
        model.fit(train[features], pd.to_numeric(train[target], errors="coerce"))
        pred = np.clip(model.predict(hold[features]), -35.0, 35.0)
        m = _metrics(hold[target], pd.Series(pred, index=hold.index))
        rows.append({"target": target, "model": name, **m})
    grid = pd.DataFrame(rows).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)
    return str(grid.iloc[0]["model"]), grid


def _predict_margin(train: pd.DataFrame, test: pd.DataFrame, features: list[str], family: str, target: str, seed: int) -> np.ndarray:
    model = _models(seed)[family]
    model.fit(train[features], pd.to_numeric(train[target], errors="coerce"))
    return np.clip(model.predict(test[features]), -35.0, 35.0)


def _expanding_margin_oof(x24: pd.DataFrame, features: list[str], mean_family: str, final_family: str) -> pd.DataFrame:
    pieces = []
    weeks = sorted(pd.to_numeric(x24["week"], errors="coerce").dropna().astype(int).unique())
    for week in weeks:
        train = x24.loc[pd.to_numeric(x24["week"], errors="coerce").lt(week)].copy()
        test = x24.loc[pd.to_numeric(x24["week"], errors="coerce").eq(week)].copy()
        if week < 5 or len(train) < 90 or test.empty:
            continue
        z = test[TEAM_KEYS + ["home", "lead_play_share", "neutral_play_share", "trail_play_share"]].copy()
        z["pred_mean_margin"] = _predict_margin(train, test, features, mean_family, "mean_score_diff", 9461)
        z["pred_final_margin"] = _predict_margin(train, test, features, final_family, "final_observed_score_diff", 9462)
        pieces.append(z)
    if not pieces:
        raise RuntimeError("M94C could not build 2024 expanding margin predictions")
    return pd.concat(pieces, ignore_index=True)


def _margin_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mean = pd.to_numeric(out["pred_mean_margin"], errors="coerce")
    final = pd.to_numeric(out["pred_final_margin"], errors="coerce")
    out["margin_blend"] = 0.65 * mean + 0.35 * final
    out["margin_abs"] = out["margin_blend"].abs()
    out["margin_positive"] = out["margin_blend"].clip(lower=0)
    out["margin_negative"] = (-out["margin_blend"]).clip(lower=0)
    out["margin_disagreement"] = mean - final
    return out


def _state_map_models(seed: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=8.0)),
        ]),
        "gbr": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                random_state=seed, n_estimators=120, learning_rate=0.035,
                max_depth=2, min_samples_leaf=12, loss="huber",
            )),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=seed, n_estimators=300, max_depth=4,
                min_samples_leaf=10, max_features=1.0, n_jobs=-1,
            )),
        ]),
    }


def _state_features() -> list[str]:
    return ["pred_mean_margin", "pred_final_margin", "margin_blend", "margin_abs", "margin_positive", "margin_negative", "margin_disagreement", "home"]


def _select_state_mapper(train_map: pd.DataFrame, hold: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    feats = _state_features()
    rows = []
    for family in _state_map_models(9470):
        maes = []
        rmses = []
        for i, state in enumerate(STATE_NAMES):
            model = _state_map_models(9470 + i)[family]
            target = f"{state}_play_share"
            model.fit(train_map[feats], pd.to_numeric(train_map[target], errors="coerce"))
            p = np.clip(model.predict(hold[feats]), 0.001, 0.999)
            m = _metrics(hold[target], pd.Series(p, index=hold.index))
            maes.append(float(m["mae"])); rmses.append(float(m["rmse"]))
        rows.append({"model": family, "mean_state_share_mae": float(np.mean(maes)), "mean_state_share_rmse": float(np.mean(rmses))})
    grid = pd.DataFrame(rows).sort_values(["mean_state_share_mae", "mean_state_share_rmse", "model"]).reset_index(drop=True)
    return str(grid.iloc[0]["model"]), grid


def _fit_predict_state_mapper(train_map: pd.DataFrame, test: pd.DataFrame, family: str) -> pd.DataFrame:
    feats = _state_features()
    raw = []
    for i, state in enumerate(STATE_NAMES):
        model = _state_map_models(9480 + i)[family]
        target = f"{state}_play_share"
        model.fit(train_map[feats], pd.to_numeric(train_map[target], errors="coerce"))
        raw.append(np.clip(model.predict(test[feats]), 0.001, 0.999))
    arr = np.column_stack(raw)
    arr = arr / arr.sum(axis=1, keepdims=True)
    out = test.copy()
    for i, state in enumerate(STATE_NAMES):
        out[f"pred_{state}_play_share"] = arr[:, i]
    return out


def _structured_team_rush(x: pd.DataFrame) -> pd.Series:
    rate = pd.Series(0.0, index=x.index, dtype=float)
    defaults = {"lead": 0.52, "neutral": 0.43, "trail": 0.33}
    for state in STATE_NAMES:
        r = pd.to_numeric(x.get(f"gs_team_{state}_rush_rate_shrunk"), errors="coerce").fillna(defaults[state]).clip(0.15, 0.75)
        s = pd.to_numeric(x[f"pred_{state}_play_share"], errors="coerce").fillna(1 / 3)
        rate += r * s
    return (pd.to_numeric(x["pred_off_plays"], errors="coerce") * rate).clip(8.0, 50.0)


def _choose_blend(hold: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    actual = pd.to_numeric(hold["actual_team_rush_att"], errors="coerce")
    for alpha in BLEND_GRID:
        cand = ((1 - alpha) * pd.to_numeric(hold["baseline_team_rush_att"], errors="coerce") + alpha * pd.to_numeric(hold["structured_team_rush_att"], errors="coerce")).clip(8, 50)
        m = _metrics(actual, cand)
        hi = actual.ge(30); lo = actual.le(20)
        hm = _metrics(actual.loc[hi], cand.loc[hi]); lm = _metrics(actual.loc[lo], cand.loc[lo])
        rows.append({"alpha": alpha, "all_mae": m["mae"], "all_rmse": m["rmse"], "all_bias": m["bias"], "rush_30_plus_mae": hm["mae"], "rush_20_or_less_mae": lm["mae"]})
    grid = pd.DataFrame(rows).sort_values(["all_mae", "rush_30_plus_mae", "rush_20_or_less_mae", "alpha"]).reset_index(drop=True)
    return float(grid.iloc[0]["alpha"]), grid


def _team_compare(x: pd.DataFrame, m94b: pd.DataFrame, scope: str) -> pd.DataFrame:
    z = x.merge(m94b[TEAM_KEYS + ["candidate_team_rush_att"]].rename(columns={"candidate_team_rush_att": "m94b_team_rush_att"}), on=TEAM_KEYS, how="left", validate="one_to_one")
    actual = pd.to_numeric(z["actual_team_rush_att"], errors="coerce")
    masks = {"all_team_games": pd.Series(True, index=z.index), "actual_20_or_less": actual.le(20), "actual_21_29": actual.between(21, 29), "actual_30_plus": actual.ge(30), "actual_35_plus": actual.ge(35)}
    rows = []
    for name, mask in masks.items():
        g = z.loc[mask]
        b = _metrics(g["actual_team_rush_att"], g["baseline_team_rush_att"])
        old = _metrics(g["actual_team_rush_att"], g["m94b_team_rush_att"])
        new = _metrics(g["actual_team_rush_att"], g["candidate_team_rush_att"])
        rows.append({"season_scope": scope, "slice": name, "n": b["n"], "m91_mae": b["mae"], "m94b_mae": old["mae"], "m94c_mae": new["mae"], "gain_vs_m91": b["mae"] - new["mae"], "gain_vs_m94b": old["mae"] - new["mae"], "m91_corr": b["correlation"], "m94b_corr": old["correlation"], "m94c_corr": new["correlation"]})
    return pd.DataFrame(rows)


def _rb_compare(rb: pd.DataFrame, m94b_rb: pd.DataFrame) -> pd.DataFrame:
    old = m94b_rb[PLAYER_KEYS + ["candidate_rush_att", "candidate_rush_yards", "candidate_rush_rec_yards"]].rename(columns={"candidate_rush_att": "m94b_rush_att", "candidate_rush_yards": "m94b_rush_yards", "candidate_rush_rec_yards": "m94b_rush_rec_yards"})
    z = rb.merge(old, on=PLAYER_KEYS, how="left", validate="one_to_one")
    a = pd.to_numeric(z["actual_rush_att"], errors="coerce")
    masks = {"all_rb": pd.Series(True, index=z.index), "actual_0_5": a.le(5), "actual_6_10": a.between(6,10), "actual_11_14": a.between(11,14), "actual_15_plus": a.ge(15), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25), "bellcow_60": z["bellcow_60"].fillna(False)}
    specs = [("rush_att", "actual_rush_att", "base_rush_att", "m94b_rush_att", "candidate_rush_att"), ("rush_yards", "actual_rush_yards", "base_rush_yards", "m94b_rush_yards", "candidate_rush_yards"), ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "m94b_rush_rec_yards", "candidate_rush_rec_yards")]
    rows = []
    for slice_name, mask in masks.items():
        g = z.loc[mask]
        for market, actual, base, oldc, newc in specs:
            bm = _metrics(g[actual], g[base]); om = _metrics(g[actual], g[oldc]); nm = _metrics(g[actual], g[newc])
            rows.append({"slice": slice_name, "market": market, "n": bm["n"], "m91_mae": bm["mae"], "m94b_mae": om["mae"], "m94c_mae": nm["mae"], "gain_vs_m91": bm["mae"] - nm["mae"], "gain_vs_m94b": om["mae"] - nm["mae"], "m91_bias": bm["bias"], "m94b_bias": om["bias"], "m94c_bias": nm["bias"], "m91_corr": bm["correlation"], "m94b_corr": om["correlation"], "m94c_corr": nm["correlation"]})
    return pd.DataFrame(rows)


def _carry_tail_diagnostics(rb: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(rb["actual_rush_att"], errors="coerce")
    rows = []
    for name, mask in {"all_rb": pd.Series(True, index=rb.index), "actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25)}.items():
        g = rb.loc[mask]
        row = {"slice": name, "n": int(len(g))}
        for label, col in [("actual", "actual_rush_att"), ("m91", "base_rush_att"), ("m94c", "candidate_rush_att")]:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{label}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{label}_median"] = float(v.median()) if len(v) else np.nan
            row[f"{label}_min"] = float(v.min()) if len(v) else np.nan
            row[f"{label}_max"] = float(v.max()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _state_summary(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for state in STATE_NAMES:
        m = _metrics(x[f"{state}_play_share"], x[f"pred_{state}_play_share"])
        rows.append({"season_scope": scope, "target": f"{state}_play_share", **m})
    for target, col in [("mean_score_diff", "pred_mean_margin"), ("final_observed_score_diff", "pred_final_margin"), ("actual_off_plays", "pred_off_plays")]:
        m = _metrics(x[target], x[col]); rows.append({"season_scope": scope, "target": target, **m})
    return pd.DataFrame(rows)


def _value(table: pd.DataFrame, slice_name: str, col: str) -> float:
    q = table.loc[table["slice"].eq(slice_name), col]
    return float(q.iloc[0]) if len(q) else np.nan


def _rb_value(table: pd.DataFrame, slice_name: str, market: str, col: str) -> float:
    q = table.loc[table["slice"].eq(slice_name) & table["market"].eq(market), col]
    return float(q.iloc[0]) if len(q) else np.nan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--m94b-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m94c"))
    args = p.parse_args()

    state_hist = _lower(pd.read_csv(_find_one(args.m94b_root, "m94b_game_state_history.csv"), low_memory=False))
    m94b_hold = _lower(pd.read_csv(_find_one(args.m94b_root, "m94b_2024_holdout_trace.csv"), low_memory=False))
    m94b_25 = _lower(pd.read_csv(_find_one(args.m94b_root, "m94b_2025_team_trace.csv"), low_memory=False))
    m94b_rb25 = _lower(pd.read_csv(_find_one(args.m94b_root, "m94b_2025_rb_trace.csv"), low_memory=False))

    x24, pred24 = build_state_features(args.m91_root, state_hist, 2024)
    x25, pred25 = build_state_features(args.m91_root, state_hist, 2025)
    x24 = _add_margin_labels(_add_strength_edges(x24, pred24), state_hist)
    x25 = _add_margin_labels(_add_strength_edges(x25, pred25), state_hist)
    features = sorted(set(_feature_cols(x24)) & set(_feature_cols(x25)))

    week24 = pd.to_numeric(x24["week"], errors="coerce")
    train24 = x24.loc[week24.le(12)].copy()
    hold24 = x24.loc[week24.ge(13)].copy()

    mean_family, mean_grid = _select_margin_family(train24, hold24, features, "mean_score_diff", 9451)
    final_family, final_grid = _select_margin_family(train24, hold24, features, "final_observed_score_diff", 9452)
    play_family, play_grid = _select_scalar_family(train24, hold24, features, "actual_off_plays", 9453)

    # Clean mapper training uses expanding 2024 predictions only; no target row's
    # margin label is used to generate its mapper input.
    oof24 = _margin_state_features(_expanding_margin_oof(x24, features, mean_family, final_family))
    train_map = oof24.loc[pd.to_numeric(oof24["week"], errors="coerce").le(12)].copy()

    hold24["pred_mean_margin"] = _predict_margin(train24, hold24, features, mean_family, "mean_score_diff", 9491)
    hold24["pred_final_margin"] = _predict_margin(train24, hold24, features, final_family, "final_observed_score_diff", 9492)
    hold24 = _margin_state_features(hold24)
    state_family, state_grid = _select_state_mapper(train_map, hold24)
    hold24 = _fit_predict_state_mapper(train_map, hold24, state_family)
    hold24 = _fit_predict_plays(train24, hold24, features, play_family)
    hold24["structured_team_rush_att"] = _structured_team_rush(hold24)
    alpha, blend_grid = _choose_blend(hold24)
    hold24["candidate_team_rush_att"] = ((1-alpha) * pd.to_numeric(hold24["baseline_team_rush_att"], errors="coerce") + alpha * hold24["structured_team_rush_att"]).clip(8,50)

    # 2025: freeze all families/alpha, refit football models on all 2024.
    x25["pred_mean_margin"] = _predict_margin(x24, x25, features, mean_family, "mean_score_diff", 9501)
    x25["pred_final_margin"] = _predict_margin(x24, x25, features, final_family, "final_observed_score_diff", 9502)
    x25 = _margin_state_features(x25)
    mapper_all24 = oof24.copy()
    x25 = _fit_predict_state_mapper(mapper_all24, x25, state_family)
    x25 = _fit_predict_plays(x24, x25, features, play_family)
    x25["structured_team_rush_att"] = _structured_team_rush(x25)
    x25["candidate_team_rush_att"] = ((1-alpha) * pd.to_numeric(x25["baseline_team_rush_att"], errors="coerce") + alpha * x25["structured_team_rush_att"]).clip(8,50)

    team24 = _team_compare(hold24, m94b_hold, "2024_w13_18_holdout")
    team25 = _team_compare(x25, m94b_25, "2025_validation")
    team_summary = pd.concat([team24, team25], ignore_index=True)
    state_summary = pd.concat([_state_summary(hold24, "2024_w13_18_holdout"), _state_summary(x25, "2025_validation")], ignore_index=True)

    rb25 = _player_candidate(pred25, x25, "candidate_team_rush_att")
    rb_summary = _rb_summary(rb25)
    rb_compare = _rb_compare(rb25, m94b_rb25)
    tail = _carry_tail_diagnostics(rb25)
    guard = _legacy_guard(pred25, rb25)

    pass_gate = (
        _value(team25, "all_team_games", "gain_vs_m94b") > 0
        and _value(team25, "actual_20_or_less", "gain_vs_m94b") >= 0
        and _value(team25, "actual_30_plus", "gain_vs_m94b") >= 0
        and _rb_value(rb_compare, "all_rb", "rush_att", "gain_vs_m94b") > 0
        and _rb_value(rb_compare, "all_rb", "rush_yards", "gain_vs_m94b") >= 0
        and _rb_value(rb_compare, "actual_20_plus", "rush_att", "gain_vs_m94b") >= 0
        and float(guard["mae_gain"].iloc[0]) >= 0
    )

    disposition = pd.DataFrame([{
        "mean_margin_family": mean_family,
        "final_margin_family": final_family,
        "state_mapper_family": state_family,
        "play_model_family": play_family,
        "frozen_blend_alpha": alpha,
        "feature_count": len(features),
        "development_train_weeks": "2024_01-12",
        "development_holdout_weeks": "2024_13-18",
        "validation_season": 2025,
        "validation_pass": int(pass_gate),
        "disposition": "ADVANCE_GAME_ENVIRONMENT_SIGNAL" if pass_gate else "DO_NOT_ADVANCE_M94C_TO_PRODUCTION",
        "note": "Research only; no sportsbook inputs and no production change.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean_grid.to_csv(args.out_dir / "m94c_mean_margin_grid.csv", index=False)
    final_grid.to_csv(args.out_dir / "m94c_final_margin_grid.csv", index=False)
    play_grid.to_csv(args.out_dir / "m94c_play_grid.csv", index=False)
    state_grid.to_csv(args.out_dir / "m94c_state_mapper_grid.csv", index=False)
    blend_grid.to_csv(args.out_dir / "m94c_blend_grid.csv", index=False)
    team_summary.to_csv(args.out_dir / "m94c_team_volume_summary.csv", index=False)
    state_summary.to_csv(args.out_dir / "m94c_game_environment_summary.csv", index=False)
    rb_summary.to_csv(args.out_dir / "m94c_rb_validation_summary.csv", index=False)
    rb_compare.to_csv(args.out_dir / "m94c_rb_vs_m94b.csv", index=False)
    tail.to_csv(args.out_dir / "m94c_carry_tail_diagnostics.csv", index=False)
    guard.to_csv(args.out_dir / "m94c_legacy_guard.csv", index=False)
    disposition.to_csv(args.out_dir / "m94c_disposition.csv", index=False)
    hold24.to_csv(args.out_dir / "m94c_2024_holdout_trace.csv", index=False)
    x25.to_csv(args.out_dir / "m94c_2025_team_trace.csv", index=False)
    rb25.to_csv(args.out_dir / "m94c_2025_rb_trace.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.out_dir / "m94c_features.csv", index=False)

    print("[rb_m94c] selected architecture")
    print(disposition.to_string(index=False))
    print("\n[rb_m94c] team volume: M91 vs M94B vs M94C")
    print(team_summary.to_string(index=False))
    print("\n[rb_m94c] game environment")
    print(state_summary.to_string(index=False))
    print("\n[rb_m94c] RB validation vs M94B")
    print(rb_compare.loc[rb_compare["slice"].isin(["all_rb", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"])].to_string(index=False))
    print("\n[rb_m94c] carry-tail diagnostics")
    print(tail.to_string(index=False))
    print("\n[rb_m94c] legacy guard")
    print(guard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
