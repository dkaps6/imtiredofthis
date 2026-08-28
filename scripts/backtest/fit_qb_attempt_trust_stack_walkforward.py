#!/usr/bin/env python3
"""Migration 61A: leakage-safe selective-trust QB pass-attempt models.

Purpose
-------
Migration 58/59 showed that removing attempt shrinkage contains useful signal but
creates too many high-side tail misses. This migration does not sweep caps. It
builds a second-stage trust model using only *previously generated OOS base
predictions* plus pregame QB volume-history/context features.

For each target week:
1. fit the existing capped/shrunk and raw attempt models on earlier games only;
2. generate current-week OOS base predictions;
3. train meta-models only on prior weeks' OOS base predictions;
4. predict actual attempts for the target week;
5. retain the existing 18-48 outer plausibility bounds.

No sportsbook player-prop line is used as a feature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.diagnose_qb_gamescript_attribution import (
    FORCE_PASS,
    MARKET,
    OPP_OFFENSE,
    PACE,
    TEAM_TENDENCY,
    fit_residual,
    prepare,
)
from scripts.backtest.fit_qb_both_raw_walkforward import fit_raw_residual
from scripts.backtest.fit_qb_gamescript_attempts_walkforward import metrics, num, read
from scripts.backtest.fit_qb_joint_attempts_ypa_walkforward import add_qb_and_matchup_context

OUTER_ATTEMPT_BOUNDS = (18.0, 48.0)
META_MIN_ROWS = 60


def add_attempt_history_features(x: pd.DataFrame, logs: pd.DataFrame, season: int) -> pd.DataFrame:
    p = logs.copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    p["season"] = num(p.season)
    p["week"] = num(p.week)
    att_col = "pass_att" if "pass_att" in p.columns else "attempts"
    if att_col not in p.columns:
        raise RuntimeError("player logs require pass_att/attempts")

    rows = []
    for _, r in x.iterrows():
        prior = p[(p.season < season) | ((p.season == season) & (p.week < int(r.week)))]
        g = prior[prior.player_clean_key.astype(str).eq(str(r.player_clean_key))].sort_values(["season", "week"])
        vals = num(g[att_col]).dropna().tail(8).astype(float)
        rec = {"_row": r.name}
        if len(vals):
            a = vals.to_numpy(dtype=float)
            rec["qb_att_last1"] = float(a[-1])
            rec["qb_att_last3"] = float(np.mean(a[-3:]))
            rec["qb_att_mean8"] = float(np.mean(a))
            rec["qb_att_std8"] = float(np.std(a, ddof=0))
            rec["qb_att_iqr8"] = float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) >= 2 else 0.0
            rec["qb_att_min8"] = float(np.min(a))
            rec["qb_att_max8"] = float(np.max(a))
            rec["qb_att_games8"] = float(len(a))
            rec["qb_att_40plus_rate8"] = float(np.mean(a >= 40.0))
            if len(a) >= 2:
                t = np.arange(len(a), dtype=float)
                rec["qb_att_trend8"] = float(np.polyfit(t, a, 1)[0])
            else:
                rec["qb_att_trend8"] = 0.0
        else:
            for c in (
                "qb_att_last1", "qb_att_last3", "qb_att_mean8", "qb_att_std8", "qb_att_iqr8",
                "qb_att_min8", "qb_att_max8", "qb_att_games8", "qb_att_40plus_rate8", "qb_att_trend8",
            ):
                rec[c] = np.nan
        rows.append(rec)
    return x.join(pd.DataFrame(rows).set_index("_row"))


def fit_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=15.0),
    )
    model.fit(train[features], num(train.actual_pass_att))
    return pd.Series(model.predict(test[features]), index=test.index)


def fit_gbr(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            loss="huber",
            n_estimators=120,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=12,
            random_state=61,
        ),
    )
    model.fit(train[features], num(train.actual_pass_att))
    return pd.Series(model.predict(test[features]), index=test.index)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--market-trace", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-train", type=int, default=80)
    p.add_argument("--meta-min-rows", type=int, default=META_MIN_ROWS)
    a = p.parse_args()

    logs = read(a.player_logs)
    x = prepare(a.market_trace, a.team_weekly, a.season)
    x = add_qb_and_matchup_context(x, logs, read(a.team_weekly), read(a.weather), a.season)
    x = add_attempt_history_features(x, logs, a.season)

    attempt_features = MARKET + ["pred_attempts"] + PACE + TEAM_TENDENCY + OPP_OFFENSE + FORCE_PASS
    meta_features = [
        "pred_attempts", "attempts_capped", "attempts_raw", "raw_minus_capped", "abs_raw_minus_capped",
        "raw_minus_recent", "capped_minus_recent",
        "qb_att_last1", "qb_att_last3", "qb_att_mean8", "qb_att_std8", "qb_att_iqr8",
        "qb_att_min8", "qb_att_max8", "qb_att_games8", "qb_att_40plus_rate8", "qb_att_trend8",
        "market_spread", "market_abs_spread", "market_total", "market_team_implied",
        "pregame_win_probability", "expected_trailing_probability", "competitive_game",
        "team_neutral_pace", "team_plays_est", "opp_neutral_pace", "opp_plays_est",
        "team_dropback_rate", "team_proe", "team_success_rate_off", "opp_dropback_rate",
        "opp_success_rate_off", "opponent_force_pass",
    ]

    for c in (
        "attempts_capped", "attempts_raw", "attempts_stack_ridge", "attempts_stack_gbr", "attempts_stack_consensus",
        "raw_minus_capped", "abs_raw_minus_capped", "raw_minus_recent", "capped_minus_recent", "meta_history_rows",
    ):
        x[c] = np.nan

    scored: list[int] = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train = x[num(x.week) < week].copy()
        test = x[num(x.week) == week].copy()
        if len(train) < a.min_train or test.empty:
            continue

        capped = fit_residual(train, test, attempt_features)
        raw_delta = fit_raw_residual(
            train,
            test,
            attempt_features,
            num(train.actual_pass_att) - num(train.pred_attempts),
        )
        raw = (num(test.pred_attempts) + raw_delta).clip(*OUTER_ATTEMPT_BOUNDS)

        x.loc[test.index, "attempts_capped"] = capped
        x.loc[test.index, "attempts_raw"] = raw
        x.loc[test.index, "raw_minus_capped"] = raw - capped
        x.loc[test.index, "abs_raw_minus_capped"] = (raw - capped).abs()
        x.loc[test.index, "raw_minus_recent"] = raw - num(test.qb_att_mean8)
        x.loc[test.index, "capped_minus_recent"] = capped - num(test.qb_att_mean8)

        meta_train = x[(num(x.week) < week) & num(x.attempts_capped).notna() & num(x.attempts_raw).notna()].copy()
        x.loc[test.index, "meta_history_rows"] = len(meta_train)

        if len(meta_train) < int(a.meta_min_rows):
            # Frozen fallback: until enough prior OOS rows exist, use the safer
            # capped estimate. No current-week actual is consulted.
            ridge_pred = capped.copy()
            gbr_pred = capped.copy()
        else:
            ridge_pred = fit_ridge(meta_train, test, meta_features)
            gbr_pred = fit_gbr(meta_train, test, meta_features)

        ridge_pred = ridge_pred.clip(*OUTER_ATTEMPT_BOUNDS)
        gbr_pred = gbr_pred.clip(*OUTER_ATTEMPT_BOUNDS)
        consensus = ((ridge_pred + gbr_pred) / 2.0).clip(*OUTER_ATTEMPT_BOUNDS)
        x.loc[test.index, "attempts_stack_ridge"] = ridge_pred
        x.loc[test.index, "attempts_stack_gbr"] = gbr_pred
        x.loc[test.index, "attempts_stack_consensus"] = consensus
        scored.extend(test.index.tolist())
        print(
            f"[m61 fit] {a.season} W{week:02d} rows={len(test)} meta_history={len(meta_train)} "
            f"mean_gap={float((raw-capped).abs().mean()):.3f}"
        )

    x = x.loc[sorted(set(scored))].copy()
    if x.empty:
        raise RuntimeError("no OOS M61 attempt rows")

    rows = []
    for cand in ("attempts_capped", "attempts_raw", "attempts_stack_ridge", "attempts_stack_gbr", "attempts_stack_consensus"):
        rows.append({"season": a.season, "candidate": cand, **metrics(x.actual_pass_att, x[cand])})
    summary = pd.DataFrame(rows)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "qb_attempt_trust_stack_trace.csv", index=False)
    summary.to_csv(a.out_dir / "qb_attempt_trust_stack_attempt_metrics.csv", index=False)
    print("=== M61 ATTEMPT METRICS ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
