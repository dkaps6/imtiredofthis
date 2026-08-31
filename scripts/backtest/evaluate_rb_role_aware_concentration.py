"""Migration 93B: learn when backfield concentration should activate.

M93A showed that a fixed share-sharpening exponent improves high-workload RB
outcomes but hurts legitimate committee/middle-workload backs. M93B therefore
keeps the total M91 RB carry pool, rushing efficiency, receiving projection,
and every production coefficient frozen, while learning a *pregame* team-level
concentration gate from prior usage and current depth-chart role evidence.

Design:
- 2024 weeks 1-12 train a simple logistic concentration classifier.
- 2024 weeks 13-18 select among a small predeclared target/gate/gamma grid.
- Refit the selected classifier on all 2024.
- Score untouched 2025 as the validation season.

No sportsbook information is used. No production file is modified.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_rb_backfield_concentration import (
    KEYS,
    TEAM_KEYS,
    _legacy_guard,
    _metrics,
    _rb_frame,
    _read_predictions,
)

TARGET_THRESHOLDS = (0.65, 0.70, 0.75)
PROBABILITY_GATES = (0.50, 0.60, 0.70)
HIGH_GAMMAS = (1.10, 1.20, 1.30)
DEVELOPMENT_TRAIN_END_WEEK = 12
DEVELOPMENT_HOLDOUT_START_WEEK = 13


def _clean_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _read_logs(root: Path, season: int) -> pd.DataFrame:
    path = root / str(season) / "player_game_logs_history.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing historical player log file: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "position", "player", "rushes", "targets", "receptions"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"player logs missing columns: {sorted(missing)}")
    if "player_clean_key" not in x.columns:
        x["player_clean_key"] = x["player"].map(_clean_name)
    x["player_clean_key"] = x["player_clean_key"].fillna(x["player"].map(_clean_name)).astype(str)
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    for c in ["rushes", "targets", "receptions"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    return x


def _historical_rb_tables(logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = logs.loc[logs["position"].eq("RB")].copy()
    rb["rb_pool_carries"] = rb.groupby(["season", "week", "team"])["rushes"].transform("sum")
    rb["rb_pool_share_hist"] = np.where(
        rb["rb_pool_carries"].gt(0), rb["rushes"] / rb["rb_pool_carries"], 0.0
    )
    rows: list[dict] = []
    for (season, week, team), g in rb.groupby(["season", "week", "team"], dropna=False):
        carries = g["rushes"].to_numpy(dtype=float)
        total = float(np.nansum(carries))
        shares = carries / total if total > 0 else np.zeros(len(carries), dtype=float)
        rows.append({
            "season": int(season),
            "week": int(week),
            "team": team,
            "team_top1_share_hist": float(np.nanmax(shares)) if len(shares) else np.nan,
            "team_hhi_hist": float(np.nansum(np.square(shares))) if len(shares) else np.nan,
            "team_rb_used_hist": int(np.sum(carries > 0)),
            "team_rb_pool_hist": total,
        })
    return rb, pd.DataFrame(rows)


def _prediction_role_map(pred: pd.DataFrame) -> pd.DataFrame:
    cols = KEYS + ["role"]
    if "role" not in pred.columns:
        out = pred.loc[pred["market"].eq("rush_att"), KEYS].drop_duplicates().copy()
        out["role"] = ""
        return out
    return pred.loc[pred["market"].eq("rush_att"), cols].drop_duplicates(KEYS)


def _build_team_features(
    season: int,
    pred: pd.DataFrame,
    base: pd.DataFrame,
    logs: pd.DataFrame,
) -> pd.DataFrame:
    hist_rb, team_hist = _historical_rb_tables(logs)
    roles = _prediction_role_map(pred)
    x = base.merge(roles, on=KEYS, how="left", validate="one_to_one")
    pool = x.groupby(TEAM_KEYS)["base_rush_att"].transform(lambda s: s.sum(min_count=1))
    x["base_rb_pool"] = pool
    x["base_share"] = np.where(pool.gt(0), x["base_rush_att"] / pool, 0.0)
    actual_pool = x.groupby(TEAM_KEYS)["actual_rush_att"].transform(lambda s: s.sum(min_count=1))
    x["actual_rb_pool"] = actual_pool
    x["actual_rb_share"] = np.where(actual_pool.gt(0), x["actual_rush_att"] / actual_pool, 0.0)

    rows: list[dict] = []
    for (s, week, team), g in x.groupby(TEAM_KEYS, dropna=False):
        g = g.sort_values(["base_share", "base_rush_att"], ascending=False).copy()
        lead = g.iloc[0]
        second_share = float(g.iloc[1]["base_share"]) if len(g) > 1 else 0.0
        lead_key = str(lead["player_clean_key"])
        other_keys = set(g.iloc[1:]["player_clean_key"].astype(str))

        prior_mask = (
            (hist_rb["season"].lt(int(s)))
            | (hist_rb["season"].eq(int(s)) & hist_rb["week"].lt(int(week)))
        )
        prior_team = hist_rb.loc[prior_mask & hist_rb["team"].eq(team)].copy()
        prior_lead = prior_team.loc[prior_team["player_clean_key"].astype(str).eq(lead_key)].sort_values(
            ["season", "week"]
        )
        prior_team_games = team_hist.loc[
            (
                team_hist["season"].lt(int(s))
                | (team_hist["season"].eq(int(s)) & team_hist["week"].lt(int(week)))
            )
            & team_hist["team"].eq(team)
        ].sort_values(["season", "week"])

        competitor_stats: list[tuple[float, float, float]] = []
        for key in other_keys:
            ch = prior_team.loc[prior_team["player_clean_key"].astype(str).eq(key)].sort_values(
                ["season", "week"]
            ).tail(5)
            if len(ch):
                competitor_stats.append((
                    float(ch["rushes"].mean()),
                    float(ch["rb_pool_share_hist"].mean()),
                    float(ch["targets"].mean()),
                ))

        actual_top_share = float(g["actual_rb_share"].max()) if len(g) else np.nan
        rec: dict[str, object] = {
            "season": int(s),
            "week": int(week),
            "team": team,
            "lead_player": lead["player"],
            "lead_key": lead_key,
            "lead_role": str(lead.get("role", "")),
            "baseline_lead_share": float(lead["base_share"]),
            "baseline_gap12": float(lead["base_share"] - second_share),
            "baseline_hhi": float(np.square(g["base_share"].to_numpy(dtype=float)).sum()),
            "baseline_rb_count": int(len(g)),
            "baseline_pool": float(lead["base_rb_pool"]),
            "lead_role_rb1": int(str(lead.get("role", "")).upper() == "RB1"),
            "actual_team_top_share": actual_top_share,
        }

        for n in (1, 3, 5):
            ph = prior_lead.tail(n)
            th = prior_team_games.tail(n)
            rec[f"lead_carries_avg{n}"] = float(ph["rushes"].mean()) if len(ph) else np.nan
            rec[f"lead_share_avg{n}"] = float(ph["rb_pool_share_hist"].mean()) if len(ph) else np.nan
            rec[f"lead_targets_avg{n}"] = float(ph["targets"].mean()) if len(ph) else np.nan
            rec[f"lead_receptions_avg{n}"] = float(ph["receptions"].mean()) if len(ph) else np.nan
            rec[f"lead_15plus_rate{n}"] = float(ph["rushes"].ge(15).mean()) if len(ph) else np.nan
            rec[f"lead_20plus_rate{n}"] = float(ph["rushes"].ge(20).mean()) if len(ph) else np.nan
            rec[f"team_top1_share_avg{n}"] = float(th["team_top1_share_hist"].mean()) if len(th) else np.nan
            rec[f"team_hhi_avg{n}"] = float(th["team_hhi_hist"].mean()) if len(th) else np.nan
            rec[f"team_rb_used_avg{n}"] = float(th["team_rb_used_hist"].mean()) if len(th) else np.nan
            rec[f"team_rb_pool_avg{n}"] = float(th["team_rb_pool_hist"].mean()) if len(th) else np.nan

        rec["lead_games_prev5"] = int(len(prior_lead.tail(5)))
        if pd.notna(rec["lead_carries_avg1"]) and pd.notna(rec["lead_carries_avg5"]):
            rec["lead_carry_trend_1v5"] = float(rec["lead_carries_avg1"] - rec["lead_carries_avg5"])
        else:
            rec["lead_carry_trend_1v5"] = np.nan
        if pd.notna(rec["lead_share_avg1"]) and pd.notna(rec["lead_share_avg5"]):
            rec["lead_share_trend_1v5"] = float(rec["lead_share_avg1"] - rec["lead_share_avg5"])
        else:
            rec["lead_share_trend_1v5"] = np.nan
        rec["comp_max_carries_avg5"] = max((v[0] for v in competitor_stats), default=0.0)
        rec["comp_max_share_avg5"] = max((v[1] for v in competitor_stats), default=0.0)
        rec["comp_max_targets_avg5"] = max((v[2] for v in competitor_stats), default=0.0)
        rows.append(rec)

    return pd.DataFrame(rows)


def _feature_columns(features: pd.DataFrame) -> list[str]:
    blocked = {
        "season", "week", "team", "lead_player", "lead_key", "lead_role",
        "actual_team_top_share",
    }
    return [
        c for c in features.columns
        if c not in blocked and pd.api.types.is_numeric_dtype(features[c])
    ]


def _new_classifier() -> Pipeline:
    # Intentionally simple/regularized so M93B tests whether the signal exists,
    # rather than hiding a large hyperparameter search inside one season.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.20, max_iter=1000, random_state=93)),
    ])


def _apply_conditional_gamma(
    base: pd.DataFrame,
    team_scores: pd.DataFrame,
    probability_gate: float,
    high_gamma: float,
) -> pd.DataFrame:
    out = base.copy()
    scores = team_scores[TEAM_KEYS + ["concentration_probability"]].copy()
    out = out.merge(scores, on=TEAM_KEYS, how="left", validate="many_to_one")
    out["concentration_probability"] = out["concentration_probability"].fillna(0.0)
    out["gamma"] = np.where(out["concentration_probability"].ge(probability_gate), high_gamma, 1.0)

    pool = out.groupby(TEAM_KEYS)["base_rush_att"].transform(lambda s: s.sum(min_count=1))
    out["base_rb_pool_rush_att"] = pool
    base_share = np.where(pool.gt(0), out["base_rush_att"] / pool, 0.0)
    out["base_rb_pool_share"] = base_share
    raw = np.power(np.clip(base_share, 1e-12, None), out["gamma"].to_numpy(dtype=float))
    out["_raw_share"] = raw
    denom = out.groupby(TEAM_KEYS)["_raw_share"].transform("sum")
    out["candidate_rb_pool_share"] = np.where(denom.gt(0), out["_raw_share"] / denom, base_share)
    out["candidate_rush_att"] = out["candidate_rb_pool_share"] * pool

    implied_ypc = np.where(
        out["base_rush_att"].abs().gt(1e-9),
        out["base_rush_yards"] / out["base_rush_att"],
        np.nan,
    )
    out["candidate_rush_yards"] = np.where(
        np.isfinite(implied_ypc), out["candidate_rush_att"] * implied_ypc, out["base_rush_yards"]
    )
    out["candidate_rush_rec_yards"] = (
        out["base_rush_rec_yards"] + out["candidate_rush_yards"] - out["base_rush_yards"]
    )
    return out.drop(columns=["_raw_share"])


def _slice_mask(x: pd.DataFrame, name: str) -> pd.Series:
    a = pd.to_numeric(x["actual_rush_att"], errors="coerce")
    if name == "all_rb":
        return pd.Series(True, index=x.index)
    if name == "actual_0_5":
        return a.le(5)
    if name == "actual_6_10":
        return a.between(6, 10)
    if name == "actual_11_14":
        return a.between(11, 14)
    if name == "actual_15_plus":
        return a.ge(15)
    if name == "actual_20_plus":
        return a.ge(20)
    if name == "actual_25_plus":
        return a.ge(25)
    if name == "bellcow_60":
        return x["bellcow_60"].fillna(False)
    raise KeyError(name)


def _score_trace(trace: pd.DataFrame, season_label: object, candidate_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    slices = (
        "all_rb", "actual_0_5", "actual_6_10", "actual_11_14",
        "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60",
    )
    markets = (
        ("rush_att", "actual_rush_att", "base_rush_att", "candidate_rush_att"),
        ("rush_yards", "actual_rush_yards", "base_rush_yards", "candidate_rush_yards"),
        ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "candidate_rush_rec_yards"),
    )
    for slice_name in slices:
        g = trace.loc[_slice_mask(trace, slice_name)]
        for market, actual_col, base_col, cand_col in markets:
            b = _metrics(g, base_col, actual_col)
            c = _metrics(g, cand_col, actual_col)
            rows.append({
                "season": season_label,
                "candidate": candidate_name,
                "slice": slice_name,
                "market": market,
                "n": b["n"],
                "baseline_mae": b["mae"],
                "candidate_mae": c["mae"],
                "mae_gain": b["mae"] - c["mae"],
                "baseline_rmse": b["rmse"],
                "candidate_rmse": c["rmse"],
                "baseline_bias": b["bias"],
                "candidate_bias": c["bias"],
                "baseline_correlation": b["correlation"],
                "candidate_correlation": c["correlation"],
            })
    return pd.DataFrame(rows)


def _value(summary: pd.DataFrame, market: str, slice_name: str, field: str = "mae_gain") -> float:
    q = summary.loc[summary["market"].eq(market) & summary["slice"].eq(slice_name), field]
    return float(q.iloc[0]) if len(q) else np.nan


def _development_grid(
    base_2024: pd.DataFrame,
    features_2024: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    train = features_2024["week"].le(DEVELOPMENT_TRAIN_END_WEEK)
    holdout = features_2024["week"].ge(DEVELOPMENT_HOLDOUT_START_WEEK)
    holdout_keys = set(
        map(tuple, features_2024.loc[holdout, TEAM_KEYS].itertuples(index=False, name=None))
    )
    base_holdout = base_2024.loc[
        base_2024[TEAM_KEYS].apply(tuple, axis=1).isin(holdout_keys)
    ].copy()

    rows: list[dict] = []
    for target_threshold in TARGET_THRESHOLDS:
        y = features_2024["actual_team_top_share"].ge(target_threshold).astype(int)
        clf = _new_classifier()
        clf.fit(features_2024.loc[train, feature_cols], y.loc[train])
        proba = clf.predict_proba(features_2024.loc[holdout, feature_cols])[:, 1]
        scores = features_2024.loc[holdout, TEAM_KEYS].copy()
        scores["concentration_probability"] = proba
        auc = np.nan
        if y.loc[holdout].nunique() > 1:
            auc = float(roc_auc_score(y.loc[holdout], proba))

        for probability_gate in PROBABILITY_GATES:
            for high_gamma in HIGH_GAMMAS:
                trace = _apply_conditional_gamma(base_holdout, scores, probability_gate, high_gamma)
                summary = _score_trace(trace, "2024_holdout", "conditional")
                all_att_gain = _value(summary, "rush_att", "all_rb")
                all_ry_gain = _value(summary, "rush_yards", "all_rb")
                low_ry_gain = _value(summary, "rush_yards", "actual_0_5")
                mid6_gain = _value(summary, "rush_yards", "actual_6_10")
                mid11_gain = _value(summary, "rush_yards", "actual_11_14")
                tail15_gain = _value(summary, "rush_yards", "actual_15_plus")
                tail20_gain = _value(summary, "rush_yards", "actual_20_plus")
                rr_gain = _value(summary, "rush_rec_yards", "all_rb")

                # Selection is deliberately conservative. M93A's failure mode was
                # buying tail gains by badly damaging 6-14 carry committee backs.
                eligible = (
                    all_att_gain >= 0
                    and all_ry_gain > 0
                    and low_ry_gain >= 0
                    and tail15_gain >= 0
                    and mid6_gain >= -0.50
                    and mid11_gain >= -0.50
                )
                objective = all_ry_gain + 0.10 * tail20_gain + 0.25 * all_att_gain
                rows.append({
                    "target_threshold": target_threshold,
                    "probability_gate": probability_gate,
                    "high_gamma": high_gamma,
                    "holdout_auc": auc,
                    "eligible": int(eligible),
                    "objective": objective,
                    "all_rush_att_gain": all_att_gain,
                    "all_rush_yards_gain": all_ry_gain,
                    "all_rush_rec_yards_gain": rr_gain,
                    "actual_0_5_rush_yards_gain": low_ry_gain,
                    "actual_6_10_rush_yards_gain": mid6_gain,
                    "actual_11_14_rush_yards_gain": mid11_gain,
                    "actual_15_plus_rush_yards_gain": tail15_gain,
                    "actual_20_plus_rush_yards_gain": tail20_gain,
                    "fraction_sharpened": float(np.mean(proba >= probability_gate)),
                })

    grid = pd.DataFrame(rows)
    eligible_grid = grid.loc[grid["eligible"].eq(1)].copy()
    source = eligible_grid if len(eligible_grid) else grid
    selected = source.sort_values(
        ["objective", "all_rush_yards_gain", "all_rush_att_gain", "high_gamma"],
        ascending=[False, False, False, True],
    ).iloc[0]
    config = {
        "target_threshold": float(selected["target_threshold"]),
        "probability_gate": float(selected["probability_gate"]),
        "high_gamma": float(selected["high_gamma"]),
    }
    return grid, config


def _fit_and_score_validation(
    base_2025: pd.DataFrame,
    features_2024: pd.DataFrame,
    features_2025: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, float]:
    y24 = features_2024["actual_team_top_share"].ge(config["target_threshold"]).astype(int)
    clf = _new_classifier()
    clf.fit(features_2024[feature_cols], y24)
    p25 = clf.predict_proba(features_2025[feature_cols])[:, 1]
    scores = features_2025[TEAM_KEYS].copy()
    scores["concentration_probability"] = p25
    scores["gamma"] = np.where(
        scores["concentration_probability"].ge(config["probability_gate"]),
        config["high_gamma"],
        1.0,
    )
    y25 = features_2025["actual_team_top_share"].ge(config["target_threshold"]).astype(int)
    auc = float(roc_auc_score(y25, p25)) if y25.nunique() > 1 else np.nan
    trace = _apply_conditional_gamma(
        base_2025,
        scores,
        config["probability_gate"],
        config["high_gamma"],
    )
    return trace, scores, clf, auc


def _fixed_gamma_reference(base: pd.DataFrame, gamma: float = 1.20) -> pd.DataFrame:
    scores = base[TEAM_KEYS].drop_duplicates().copy()
    scores["concentration_probability"] = 1.0
    return _apply_conditional_gamma(base, scores, 0.0, gamma)


def _coefficient_table(clf: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    model = clf.named_steps["model"]
    coef = np.ravel(model.coef_)
    out = pd.DataFrame({"feature": feature_cols, "standardized_logit_coefficient": coef})
    out["abs_coefficient"] = out["standardized_logit_coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m93b"))
    args = p.parse_args()

    preds = {s: _read_predictions(args.m91_root, s) for s in (2024, 2025)}
    bases = {s: _rb_frame(preds[s], "ml") for s in (2024, 2025)}
    logs = {s: _read_logs(args.m91_root, s) for s in (2024, 2025)}
    features = {
        s: _build_team_features(s, preds[s], bases[s], logs[s])
        for s in (2024, 2025)
    }
    feature_cols = _feature_columns(features[2024])
    if not feature_cols:
        raise RuntimeError("M93B produced no numeric pregame features")

    grid, config = _development_grid(bases[2024], features[2024], feature_cols)
    validation_trace, validation_scores, clf, validation_auc = _fit_and_score_validation(
        bases[2025], features[2024], features[2025], feature_cols, config
    )
    conditional_summary = _score_trace(validation_trace, 2025, "m93b_conditional")
    fixed_trace = _fixed_gamma_reference(bases[2025], 1.20)
    fixed_summary = _score_trace(fixed_trace, 2025, "m93a_fixed_gamma_1_2")
    summary = pd.concat([conditional_summary, fixed_summary], ignore_index=True)

    conditional_guard = _legacy_guard(preds, validation_trace)
    conditional_guard = conditional_guard.loc[conditional_guard["season"].eq(2025)].copy()
    conditional_guard["candidate"] = "m93b_conditional"
    fixed_guard = _legacy_guard(preds, fixed_trace)
    fixed_guard = fixed_guard.loc[fixed_guard["season"].eq(2025)].copy()
    fixed_guard["candidate"] = "m93a_fixed_gamma_1_2"
    guard = pd.concat([conditional_guard, fixed_guard], ignore_index=True)

    cond = conditional_summary
    all_att_gain = _value(cond, "rush_att", "all_rb")
    all_ry_gain = _value(cond, "rush_yards", "all_rb")
    all_rr_gain = _value(cond, "rush_rec_yards", "all_rb")
    low_gain = _value(cond, "rush_yards", "actual_0_5")
    mid6_gain = _value(cond, "rush_yards", "actual_6_10")
    mid11_gain = _value(cond, "rush_yards", "actual_11_14")
    tail20_gain = _value(cond, "rush_yards", "actual_20_plus")
    legacy_gain = float(conditional_guard["mae_gain"].iloc[0])
    validation_pass = (
        all_att_gain > 0
        and all_ry_gain > 0
        and all_rr_gain >= 0
        and low_gain >= 0
        and tail20_gain > 0
        and mid6_gain >= -0.50
        and mid11_gain >= -0.50
        and legacy_gain >= 0
    )
    disposition = pd.DataFrame([{
        **config,
        "development_train_weeks": "2024_01-12",
        "development_holdout_weeks": "2024_13-18",
        "validation_season": 2025,
        "validation_auc": validation_auc,
        "validation_fraction_sharpened": float(validation_scores["gamma"].gt(1.0).mean()),
        "validation_pass": int(validation_pass),
        "disposition": (
            "ADVANCE_ROLE_AWARE_CONCENTRATION"
            if validation_pass else "DO_NOT_ADVANCE_M93B_TO_PRODUCTION"
        ),
        "note": "Allocation-only research; M91 pool/efficiency/receiving remain frozen.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features[2024].to_csv(args.out_dir / "m93b_features_2024.csv", index=False)
    features[2025].to_csv(args.out_dir / "m93b_features_2025.csv", index=False)
    grid.to_csv(args.out_dir / "m93b_development_grid.csv", index=False)
    validation_scores.to_csv(args.out_dir / "m93b_validation_team_scores.csv", index=False)
    validation_trace.to_csv(args.out_dir / "m93b_validation_row_trace.csv", index=False)
    summary.to_csv(args.out_dir / "m93b_validation_summary.csv", index=False)
    guard.to_csv(args.out_dir / "m93b_legacy_guard.csv", index=False)
    _coefficient_table(clf, feature_cols).to_csv(args.out_dir / "m93b_feature_coefficients.csv", index=False)
    disposition.to_csv(args.out_dir / "m93b_disposition.csv", index=False)

    print("[rb_m93b] selected 2024 development configuration")
    print(disposition.to_string(index=False))
    print("\n[rb_m93b] 2025 validation headline")
    print(summary.loc[
        summary["slice"].isin([
            "all_rb", "actual_0_5", "actual_6_10", "actual_11_14",
            "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60",
        ])
        & summary["market"].isin(["rush_att", "rush_yards", "rush_rec_yards"]),
        [
            "candidate", "slice", "market", "n", "baseline_mae", "candidate_mae",
            "mae_gain", "baseline_bias", "candidate_bias", "baseline_correlation",
            "candidate_correlation",
        ],
    ].to_string(index=False))
    print("\n[rb_m93b] legacy all-player guard")
    print(guard.to_string(index=False))
    print("\n[rb_m93b] strongest standardized role/concentration signals")
    print(_coefficient_table(clf, feature_cols).head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
