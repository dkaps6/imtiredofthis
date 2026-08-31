"""Migration 94B: explicit football-only game-state model for team rushing volume.

M94 showed that a direct 109-feature regression modestly improved average team
rush volume but could not reliably distinguish low-rush and high-rush scripts.
M94B decomposes the problem into football mechanics instead of asking one model
to jump directly from pregame features to team carries:

    expected offensive plays
      x predicted ahead/neutral/behind play shares
      x pregame team rush tendency within each score state

No sportsbook fields are used. Target-game PBP is used only to create training /
evaluation labels. Every pregame feature and state-conditioned rush tendency is
built from games strictly before the target week.

Development protocol:
- 2024 W1-12: development training
- 2024 W13-18: architecture/model-family/blend holdout
- freeze choices
- refit on all 2024
- 2025: untouched temporal validation

The frozen M91 ML projection remains the baseline and the player allocation is
held fixed, so M94B isolates team-volume/game-state value.
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

from scripts._opponent_map import canon_team
from scripts.backtest.evaluate_rb_team_rush_volume import (
    TEAM_KEYS,
    _legacy_guard,
    _metrics,
    _player_candidate,
    _rb_summary,
    build_features as build_m94_features,
)

STATE_NAMES = ("lead", "neutral", "trail")
STATE_THRESHOLD = 3.0
ROLL_WINDOWS = (1, 3, 5)
BLEND_GRID = (0.25, 0.50, 0.75, 1.00)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _num(frame: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def load_game_state_observations(seasons: list[int]) -> pd.DataFrame:
    """Build completed team-game score-state observations from nflverse PBP."""
    import nflreadpy as nfl

    pbp = _lower(nfl.load_pbp(seasons=seasons).to_pandas())
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            pbp = reg
    required = {"season", "week", "posteam", "rush_attempt", "qb_dropback"}
    missing = required - set(pbp.columns)
    if missing:
        raise RuntimeError(f"M94B PBP missing columns: {sorted(missing)}")

    pbp["team"] = pbp["posteam"].map(canon_team)
    pbp["rush_attempt"] = _num(pbp, "rush_attempt", 0).fillna(0)
    pbp["qb_dropback"] = _num(pbp, "qb_dropback", 0).fillna(0)
    pbp["off_play"] = (pbp["rush_attempt"].eq(1) | pbp["qb_dropback"].eq(1)).astype(int)
    pbp = pbp.loc[pbp["off_play"].eq(1) & pbp["team"].ne("")].copy()

    if "score_differential" in pbp.columns:
        diff = _num(pbp, "score_differential")
    elif {"posteam_score", "defteam_score"}.issubset(pbp.columns):
        diff = _num(pbp, "posteam_score") - _num(pbp, "defteam_score")
    else:
        raise RuntimeError("M94B PBP has no score differential fields")
    pbp["score_diff"] = diff.fillna(0.0)
    pbp["state"] = np.select(
        [pbp["score_diff"].gt(STATE_THRESHOLD), pbp["score_diff"].lt(-STATE_THRESHOLD)],
        ["lead", "trail"], default="neutral",
    )
    pbp["down_num"] = _num(pbp, "down")
    pbp["game_seconds_remaining_num"] = _num(pbp, "game_seconds_remaining")

    group_cols = ["season", "week", "team"]
    rows: list[dict] = []
    for key, g in pbp.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, key))
        rec["season"] = int(rec["season"])
        rec["week"] = int(rec["week"])
        rec["actual_off_plays"] = float(len(g))
        rec["actual_rush_att_pbp"] = float(g["rush_attempt"].sum())
        rec["mean_score_diff"] = float(g["score_diff"].mean())
        rec["final_observed_score_diff"] = float(g["score_diff"].iloc[-1])
        rec["won_observed"] = float(rec["final_observed_score_diff"] > 0)
        for state in STATE_NAMES:
            sg = g.loc[g["state"].eq(state)]
            plays = float(len(sg))
            rushes = float(sg["rush_attempt"].sum())
            rec[f"{state}_plays"] = plays
            rec[f"{state}_rushes"] = rushes
            rec[f"{state}_play_share"] = plays / len(g) if len(g) else np.nan
            rec[f"{state}_rush_rate"] = rushes / plays if plays > 0 else np.nan
        neutral_early = g.loc[
            g["state"].eq("neutral")
            & g["down_num"].isin([1, 2])
            & (g["game_seconds_remaining_num"].isna() | g["game_seconds_remaining_num"].gt(900))
        ]
        rec["neutral_early_down_rush_rate"] = (
            float(neutral_early["rush_attempt"].mean()) if len(neutral_early) else np.nan
        )
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    if out.empty or out.duplicated(group_cols).any():
        raise RuntimeError("M94B game-state observations invalid")
    return out


def _prior_rows(hist: pd.DataFrame, team: str, season: int, week: int) -> pd.DataFrame:
    s = pd.to_numeric(hist["season"], errors="coerce")
    w = pd.to_numeric(hist["week"], errors="coerce")
    mask = (
        hist["team"].astype(str).eq(str(team))
        & (s.lt(int(season)) | (s.eq(int(season)) & w.lt(int(week))))
    )
    return hist.loc[mask].sort_values(["season", "week"])


def _league_state_rate(hist: pd.DataFrame, season: int, week: int, state: str) -> float:
    s = pd.to_numeric(hist["season"], errors="coerce")
    w = pd.to_numeric(hist["week"], errors="coerce")
    prior = hist.loc[s.lt(int(season)) | (s.eq(int(season)) & w.lt(int(week)))].copy()
    plays = pd.to_numeric(prior.get(f"{state}_plays"), errors="coerce").fillna(0).sum()
    rushes = pd.to_numeric(prior.get(f"{state}_rushes"), errors="coerce").fillna(0).sum()
    return float(rushes / plays) if plays > 0 else 0.43


def _state_history_features(hist: pd.DataFrame, team: str, season: int, week: int, prefix: str) -> dict[str, float]:
    g = _prior_rows(hist, team, season, week)
    rec: dict[str, float] = {f"{prefix}state_history_games": float(len(g))}
    metrics = [
        "actual_off_plays", "actual_rush_att_pbp", "mean_score_diff",
        "final_observed_score_diff", "won_observed", "neutral_early_down_rush_rate",
        *[f"{s}_play_share" for s in STATE_NAMES],
        *[f"{s}_rush_rate" for s in STATE_NAMES],
    ]
    for metric in metrics:
        if metric not in hist.columns:
            continue
        vals = pd.to_numeric(g[metric], errors="coerce")
        for n in ROLL_WINDOWS:
            z = vals.tail(n).dropna()
            rec[f"{prefix}{metric}_avg{n}"] = float(z.mean()) if len(z) else np.nan

    recent = g.tail(5)
    for state in STATE_NAMES:
        league_rate = _league_state_rate(hist, season, week, state)
        plays = pd.to_numeric(recent.get(f"{state}_plays"), errors="coerce").fillna(0).sum()
        rushes = pd.to_numeric(recent.get(f"{state}_rushes"), errors="coerce").fillna(0).sum()
        pseudo = 24.0
        rec[f"{prefix}{state}_rush_rate_shrunk"] = float(
            (rushes + pseudo * league_rate) / (plays + pseudo)
        )
        rec[f"{prefix}{state}_recent_plays"] = float(plays)
    return rec


def build_state_features(root: Path, state_hist: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, pred = build_m94_features(root, season)
    labels = state_hist.loc[pd.to_numeric(state_hist["season"], errors="coerce").eq(int(season))].copy()
    label_cols = TEAM_KEYS + [
        "actual_off_plays", "actual_rush_att_pbp", "lead_play_share",
        "neutral_play_share", "trail_play_share",
    ]
    base = base.merge(labels[label_cols], on=TEAM_KEYS, how="left", validate="one_to_one")
    if base[["actual_off_plays", "lead_play_share", "neutral_play_share", "trail_play_share"]].isna().any().any():
        raise RuntimeError(f"M94B target-game state labels incomplete for {season}")

    rows: list[dict] = []
    for _, r in base.iterrows():
        rec = r.to_dict()
        rec.update(_state_history_features(state_hist, str(r["team"]), int(r["season"]), int(r["week"]), "gs_team_"))
        rec.update(_state_history_features(state_hist, str(r["opponent"]), int(r["season"]), int(r["week"]), "gs_opp_"))
        for n in ROLL_WINDOWS:
            a = f"gs_team_mean_score_diff_avg{n}"
            b = f"gs_opp_mean_score_diff_avg{n}"
            if a in rec and b in rec:
                rec[f"gs_score_diff_edge_avg{n}"] = rec[a] - rec[b] if pd.notna(rec[a]) and pd.notna(rec[b]) else np.nan
            a = f"gs_team_won_observed_avg{n}"
            b = f"gs_opp_won_observed_avg{n}"
            if a in rec and b in rec:
                rec[f"gs_win_rate_edge_avg{n}"] = rec[a] - rec[b] if pd.notna(rec[a]) and pd.notna(rec[b]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows), pred


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "season", "week", "team", "opponent", "actual_team_rush_att",
        "actual_off_plays", "actual_rush_att_pbp", "lead_play_share",
        "neutral_play_share", "trail_play_share",
    }
    cols = []
    for c in frame.columns:
        if c in blocked:
            continue
        v = pd.to_numeric(frame[c], errors="coerce")
        if v.notna().any():
            cols.append(c)
    return sorted(cols)


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
                random_state=seed, n_estimators=160, learning_rate=0.03,
                max_depth=2, min_samples_leaf=10, loss="huber",
            )),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=seed, n_estimators=350, max_depth=5,
                min_samples_leaf=8, max_features=0.7, n_jobs=-1,
            )),
        ]),
    }


def _select_scalar_family(train: pd.DataFrame, hold: pd.DataFrame, features: list[str], target: str, seed: int) -> tuple[str, pd.DataFrame]:
    rows = []
    for name, model in _models(seed).items():
        model.fit(train[features], pd.to_numeric(train[target], errors="coerce"))
        p = model.predict(hold[features])
        m = _metrics(hold[target], pd.Series(p, index=hold.index))
        rows.append({"target": target, "model": name, "mae": m["mae"], "rmse": m["rmse"], "bias": m["bias"], "correlation": m["correlation"]})
    grid = pd.DataFrame(rows).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)
    return str(grid.iloc[0]["model"]), grid


def _select_state_family(train: pd.DataFrame, hold: pd.DataFrame, features: list[str]) -> tuple[str, pd.DataFrame]:
    rows = []
    for name in _models(944):
        maes = []
        rmses = []
        for i, state in enumerate(STATE_NAMES):
            model = _models(944 + i)[name]
            target = f"{state}_play_share"
            model.fit(train[features], pd.to_numeric(train[target], errors="coerce"))
            p = np.clip(model.predict(hold[features]), 0.0, 1.0)
            m = _metrics(hold[target], pd.Series(p, index=hold.index))
            maes.append(float(m["mae"]))
            rmses.append(float(m["rmse"]))
        rows.append({"model": name, "mean_state_share_mae": float(np.mean(maes)), "mean_state_share_rmse": float(np.mean(rmses))})
    grid = pd.DataFrame(rows).sort_values(["mean_state_share_mae", "mean_state_share_rmse", "model"]).reset_index(drop=True)
    return str(grid.iloc[0]["model"]), grid


def _fit_predict_states(train: pd.DataFrame, test: pd.DataFrame, features: list[str], family: str) -> pd.DataFrame:
    raw = []
    for i, state in enumerate(STATE_NAMES):
        model = _models(954 + i)[family]
        target = f"{state}_play_share"
        model.fit(train[features], pd.to_numeric(train[target], errors="coerce"))
        raw.append(np.clip(model.predict(test[features]), 0.001, 0.999))
    arr = np.column_stack(raw)
    arr = arr / arr.sum(axis=1, keepdims=True)
    out = test.copy()
    for i, state in enumerate(STATE_NAMES):
        out[f"pred_{state}_play_share"] = arr[:, i]
    return out


def _fit_predict_plays(train: pd.DataFrame, test: pd.DataFrame, features: list[str], family: str) -> pd.DataFrame:
    model = _models(964)[family]
    model.fit(train[features], pd.to_numeric(train["actual_off_plays"], errors="coerce"))
    out = test.copy()
    out["pred_off_plays"] = np.clip(model.predict(test[features]), 35.0, 90.0)
    return out


def _structured_team_rush(x: pd.DataFrame) -> pd.Series:
    rate = pd.Series(0.0, index=x.index, dtype=float)
    defaults = {"lead": 0.52, "neutral": 0.43, "trail": 0.33}
    for state in STATE_NAMES:
        state_rate = pd.to_numeric(x.get(f"gs_team_{state}_rush_rate_shrunk"), errors="coerce").fillna(defaults[state]).clip(0.15, 0.75)
        state_share = pd.to_numeric(x[f"pred_{state}_play_share"], errors="coerce").fillna(1 / 3)
        rate += state_rate * state_share
    return (pd.to_numeric(x["pred_off_plays"], errors="coerce") * rate).clip(8.0, 50.0)


def _add_oracles(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    pred_rate = pd.Series(0.0, index=out.index, dtype=float)
    actual_state_rate = pd.Series(0.0, index=out.index, dtype=float)
    defaults = {"lead": 0.52, "neutral": 0.43, "trail": 0.33}
    for state in STATE_NAMES:
        r = pd.to_numeric(out.get(f"gs_team_{state}_rush_rate_shrunk"), errors="coerce").fillna(defaults[state]).clip(0.15, 0.75)
        pred_share = pd.to_numeric(out[f"pred_{state}_play_share"], errors="coerce")
        actual_share = pd.to_numeric(out[f"{state}_play_share"], errors="coerce")
        pred_rate += r * pred_share
        actual_state_rate += r * actual_share
    out["oracle_actual_plays_team_rush"] = (pd.to_numeric(out["actual_off_plays"], errors="coerce") * pred_rate).clip(8.0, 50.0)
    out["oracle_actual_state_team_rush"] = (pd.to_numeric(out["pred_off_plays"], errors="coerce") * actual_state_rate).clip(8.0, 50.0)
    out["oracle_actual_plays_state_team_rush"] = (pd.to_numeric(out["actual_off_plays"], errors="coerce") * actual_state_rate).clip(8.0, 50.0)
    return out


def _team_summary(x: pd.DataFrame, candidate_col: str, season_scope: str) -> pd.DataFrame:
    actual = pd.to_numeric(x["actual_team_rush_att"], errors="coerce")
    masks = {
        "all_team_games": pd.Series(True, index=x.index),
        "actual_20_or_less": actual.le(20),
        "actual_21_29": actual.between(21, 29),
        "actual_30_plus": actual.ge(30),
        "actual_35_plus": actual.ge(35),
    }
    rows = []
    for slice_name, mask in masks.items():
        g = x.loc[mask]
        b = _metrics(g["actual_team_rush_att"], g["baseline_team_rush_att"])
        c = _metrics(g["actual_team_rush_att"], g[candidate_col])
        rows.append({
            "season_scope": season_scope, "slice": slice_name, "n": b["n"],
            "baseline_mae": b["mae"], "candidate_mae": c["mae"], "mae_gain": b["mae"] - c["mae"],
            "baseline_rmse": b["rmse"], "candidate_rmse": c["rmse"],
            "baseline_bias": b["bias"], "candidate_bias": c["bias"],
            "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
        })
    return pd.DataFrame(rows)


def _state_summary(x: pd.DataFrame, season_scope: str) -> pd.DataFrame:
    rows = []
    for state in STATE_NAMES:
        target = f"{state}_play_share"; pred = f"pred_{state}_play_share"
        m = _metrics(x[target], x[pred])
        rows.append({"season_scope": season_scope, "target": target, **m})
    pm = _metrics(x["actual_off_plays"], x["pred_off_plays"])
    rows.append({"season_scope": season_scope, "target": "actual_off_plays", **pm})
    return pd.DataFrame(rows)


def _oracle_summary(x: pd.DataFrame, season_scope: str) -> pd.DataFrame:
    rows = []
    for name, col in [
        ("baseline", "baseline_team_rush_att"),
        ("candidate", "candidate_team_rush_att"),
        ("actual_plays_oracle", "oracle_actual_plays_team_rush"),
        ("actual_state_oracle", "oracle_actual_state_team_rush"),
        ("actual_plays_state_oracle", "oracle_actual_plays_state_team_rush"),
    ]:
        m = _metrics(x["actual_team_rush_att"], x[col])
        rows.append({"season_scope": season_scope, "scenario": name, **m})
    return pd.DataFrame(rows)


def _choose_blend(hold: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    actual = pd.to_numeric(hold["actual_team_rush_att"], errors="coerce")
    for alpha in BLEND_GRID:
        cand = (1.0 - alpha) * pd.to_numeric(hold["baseline_team_rush_att"], errors="coerce") + alpha * pd.to_numeric(hold["structured_team_rush_att"], errors="coerce")
        cand = cand.clip(8.0, 50.0)
        all_m = _metrics(actual, cand)
        hi = actual.ge(30); lo = actual.le(20)
        hi_m = _metrics(actual.loc[hi], cand.loc[hi]); lo_m = _metrics(actual.loc[lo], cand.loc[lo])
        rows.append({
            "alpha": alpha, "all_mae": all_m["mae"], "all_rmse": all_m["rmse"],
            "all_bias": all_m["bias"], "rush_30_plus_mae": hi_m["mae"], "rush_20_or_less_mae": lo_m["mae"],
        })
    grid = pd.DataFrame(rows).sort_values(["all_mae", "rush_30_plus_mae", "rush_20_or_less_mae", "alpha"]).reset_index(drop=True)
    return float(grid.iloc[0]["alpha"]), grid


def _gain(table: pd.DataFrame, slice_name: str) -> float:
    q = table.loc[table["slice"].eq(slice_name), "mae_gain"]
    return float(q.iloc[0]) if len(q) else np.nan


def _rb_gain(table: pd.DataFrame, market: str, slice_name: str = "all_rb") -> float:
    q = table.loc[table["market"].eq(market) & table["slice"].eq(slice_name), "mae_gain"]
    return float(q.iloc[0]) if len(q) else np.nan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m94b"))
    args = p.parse_args()

    state_hist = load_game_state_observations([2023, 2024, 2025])
    x24, pred24 = build_state_features(args.m91_root, state_hist, 2024)
    x25, pred25 = build_state_features(args.m91_root, state_hist, 2025)
    features = sorted(set(_feature_cols(x24)) & set(_feature_cols(x25)))
    if "baseline_team_rush_att" not in features:
        raise RuntimeError("M94B feature set lost frozen M91 baseline")

    train24 = x24.loc[pd.to_numeric(x24["week"], errors="coerce").le(12)].copy()
    hold24 = x24.loc[pd.to_numeric(x24["week"], errors="coerce").ge(13)].copy()

    play_family, play_grid = _select_scalar_family(train24, hold24, features, "actual_off_plays", 941)
    state_family, state_grid = _select_state_family(train24, hold24, features)

    hold24 = _fit_predict_states(train24, hold24, features, state_family)
    hold24 = _fit_predict_plays(train24, hold24, features, play_family)
    hold24["structured_team_rush_att"] = _structured_team_rush(hold24)
    alpha, blend_grid = _choose_blend(hold24)
    hold24["candidate_team_rush_att"] = (
        (1.0 - alpha) * pd.to_numeric(hold24["baseline_team_rush_att"], errors="coerce")
        + alpha * hold24["structured_team_rush_att"]
    ).clip(8.0, 50.0)
    hold24 = _add_oracles(hold24)

    x25 = _fit_predict_states(x24, x25, features, state_family)
    x25 = _fit_predict_plays(x24, x25, features, play_family)
    x25["structured_team_rush_att"] = _structured_team_rush(x25)
    x25["candidate_team_rush_att"] = (
        (1.0 - alpha) * pd.to_numeric(x25["baseline_team_rush_att"], errors="coerce")
        + alpha * x25["structured_team_rush_att"]
    ).clip(8.0, 50.0)
    x25 = _add_oracles(x25)

    team24 = _team_summary(hold24, "candidate_team_rush_att", "2024_w13_18_holdout")
    team25 = _team_summary(x25, "candidate_team_rush_att", "2025_validation")
    team_summary = pd.concat([team24, team25], ignore_index=True)
    state_summary = pd.concat([
        _state_summary(hold24, "2024_w13_18_holdout"),
        _state_summary(x25, "2025_validation"),
    ], ignore_index=True)
    oracle_summary = pd.concat([
        _oracle_summary(hold24, "2024_w13_18_holdout"),
        _oracle_summary(x25, "2025_validation"),
    ], ignore_index=True)

    rb25 = _player_candidate(pred25, x25, "candidate_team_rush_att")
    rb_summary = _rb_summary(rb25)
    guard = _legacy_guard(pred25, rb25)

    pass_gate = (
        _gain(team24, "all_team_games") > 0
        and _gain(team25, "all_team_games") > 0
        and _gain(team25, "actual_20_or_less") > 0
        and _gain(team25, "actual_30_plus") > 0
        and _rb_gain(rb_summary, "rush_att") > 0
        and _rb_gain(rb_summary, "rush_yards") > 0
        and _rb_gain(rb_summary, "rush_att", "actual_20_plus") > 0
        and float(guard["mae_gain"].iloc[0]) >= 0
    )
    disposition = pd.DataFrame([{
        "play_model_family": play_family,
        "state_model_family": state_family,
        "frozen_blend_alpha": alpha,
        "feature_count": len(features),
        "state_threshold_points": STATE_THRESHOLD,
        "development_train_weeks": "2024_01-12",
        "development_holdout_weeks": "2024_13-18",
        "validation_season": 2025,
        "validation_pass": int(pass_gate),
        "disposition": "ADVANCE_EXPLICIT_GAME_STATE_SIGNAL" if pass_gate else "DO_NOT_ADVANCE_M94B_TO_PRODUCTION",
        "note": "Research only; no sportsbook inputs and no production change.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_hist.to_csv(args.out_dir / "m94b_game_state_history.csv", index=False)
    play_grid.to_csv(args.out_dir / "m94b_play_model_grid_2024_holdout.csv", index=False)
    state_grid.to_csv(args.out_dir / "m94b_state_model_grid_2024_holdout.csv", index=False)
    blend_grid.to_csv(args.out_dir / "m94b_blend_grid_2024_holdout.csv", index=False)
    team_summary.to_csv(args.out_dir / "m94b_team_volume_summary.csv", index=False)
    state_summary.to_csv(args.out_dir / "m94b_state_prediction_summary.csv", index=False)
    oracle_summary.to_csv(args.out_dir / "m94b_oracle_decomposition.csv", index=False)
    rb_summary.to_csv(args.out_dir / "m94b_rb_validation_summary.csv", index=False)
    guard.to_csv(args.out_dir / "m94b_legacy_rushing_guard.csv", index=False)
    disposition.to_csv(args.out_dir / "m94b_disposition.csv", index=False)
    hold24.to_csv(args.out_dir / "m94b_2024_holdout_trace.csv", index=False)
    x25.to_csv(args.out_dir / "m94b_2025_team_trace.csv", index=False)
    rb25.to_csv(args.out_dir / "m94b_2025_rb_trace.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.out_dir / "m94b_features.csv", index=False)

    print("[rb_m94b] selected architecture")
    print(disposition.to_string(index=False))
    print("\n[rb_m94b] team rushing volume")
    print(team_summary.to_string(index=False))
    print("\n[rb_m94b] explicit state / plays prediction")
    print(state_summary.to_string(index=False))
    print("\n[rb_m94b] opportunity oracles")
    print(oracle_summary.to_string(index=False))
    print("\n[rb_m94b] 2025 RB translation")
    print(rb_summary.loc[rb_summary["slice"].isin(["all_rb", "actual_0_5", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"])].to_string(index=False))
    print("\n[rb_m94b] legacy guard")
    print(guard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
