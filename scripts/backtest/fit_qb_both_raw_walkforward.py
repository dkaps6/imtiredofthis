#!/usr/bin/env python3
"""Migration 58A: rebuild the Migration 57 BOTH RAW QB candidate walk-forward.

Diagnostic/candidate-validation only. Uses the same Migration 53 feature sets,
Ridge alpha, and week-by-week fitting discipline, but removes only the inner
residual caps and 60% shrinkage. Existing outer plausibility bounds remain:
18-48 pass attempts and 4.5-10.5 YPA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    prepare,
)
from scripts.backtest.fit_qb_gamescript_attempts_walkforward import num, read
from scripts.backtest.fit_qb_joint_attempts_ypa_walkforward import add_qb_and_matchup_context


def fit_raw_residual(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    *,
    alpha: float = 30.0,
) -> pd.Series:
    usable = [
        f
        for f in features
        if f in train and num(train[f]).notna().sum() >= 10 and num(train[f]).nunique() > 1
    ]
    if not usable:
        return pd.Series(float(num(target).mean()), index=test.index)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(train[usable], num(target))
    return pd.Series(model.predict(test[usable]), index=test.index)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--market-trace", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--weather", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-train", type=int, default=80)
    a = p.parse_args()

    x = prepare(a.market_trace, a.team_weekly, a.season)
    x = add_qb_and_matchup_context(
        x,
        read(a.player_logs),
        read(a.team_weekly),
        read(a.weather),
        a.season,
    )

    attempt_features = MARKET + ["pred_attempts"] + PACE + TEAM_TENDENCY + OPP_OFFENSE + FORCE_PASS
    qb_features = [
        "pred_ypa",
        "qb_recent_ypa",
        "qb_recent_pass_att",
        "qb_recent_completion_pct",
        "qb_recent_td_rate",
        "qb_recent_int_rate",
        "qb_recent_epa_per_att",
    ]
    matchup_features = [
        "team_pressure_rate_allowed",
        "team_success_rate_off",
        "opp_pressure_rate_generated",
        "opp_def_pass_epa",
        "opp_success_rate_def",
        "opp_explosive_play_rate_allowed",
        "opp_coverage_man_rate",
        "opp_coverage_zone_rate",
        "market_total",
        "market_team_implied",
        "market_opp_implied",
        "market_spread",
        "controlled_environment",
    ]
    ypa_features = qb_features + matchup_features

    for c in ["attempts_raw", "ypa_raw", "att_raw_delta", "ypa_raw_delta"]:
        x[c] = np.nan

    scored: list[int] = []
    for week in sorted(num(x.week).dropna().astype(int).unique()):
        train = x[num(x.week) < week].copy()
        test = x[num(x.week) == week].copy()
        if len(train) < a.min_train or test.empty:
            continue

        att_delta = fit_raw_residual(
            train,
            test,
            attempt_features,
            num(train.actual_pass_att) - num(train.pred_attempts),
        )
        ypa_delta = fit_raw_residual(
            train,
            test,
            ypa_features,
            num(train.actual_ypa) - num(train.pred_ypa),
        )
        x.loc[test.index, "att_raw_delta"] = att_delta
        x.loc[test.index, "ypa_raw_delta"] = ypa_delta
        x.loc[test.index, "attempts_raw"] = (num(test.pred_attempts) + att_delta).clip(18, 48)
        x.loc[test.index, "ypa_raw"] = (num(test.pred_ypa) + ypa_delta).clip(4.5, 10.5)
        scored.extend(test.index.tolist())

    x = x.loc[sorted(set(scored))].copy()
    if x.empty:
        raise RuntimeError("no OOS BOTH RAW rows")

    # Sanity diagnostics only; canonical evaluation occurs in Migration 58B.
    x["raw_pass_yards_proxy"] = (
        num(x.mc_proj)
        * num(x.attempts_raw) / num(x.pred_attempts).replace(0, np.nan)
        * num(x.ypa_raw) / num(x.pred_ypa).replace(0, np.nan)
    )
    x["raw_pass_error_proxy"] = num(x.raw_pass_yards_proxy) - num(x.actual_pass_yards_raw)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "qb_both_raw_walkforward_trace.csv"
    x.to_csv(out, index=False)
    print(
        f"[m58 raw] season={a.season} rows={len(x)} "
        f"attempt_delta_sd={num(x.att_raw_delta).std(ddof=0):.4f} "
        f"ypa_delta_sd={num(x.ypa_raw_delta).std(ddof=0):.4f}"
    )
    print(f"[m58 raw] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
