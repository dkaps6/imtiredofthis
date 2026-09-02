#!/usr/bin/env python3
"""Mechanical correction wrapper for frozen RB STACK6I protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_rb_stack6i_team_rush_mechanics_oracle import (
    START_WEEK,
    EXPECTED_W6_18_N,
    EXPECTED_BASE_MAE,
    EXPECTED_BASE_RMSE,
    EXPECTED_BASE_BIAS,
    EXPECTED_BASE_CORR,
    one,
    build_trace,
    score_population,
    metric_pair,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    trace = build_trace(a.m94c_root, a.stack6h_root)
    w = trace.loc[trace.season.eq(2025) & trace.week.ge(START_WEEK)].copy()

    overall = score_population(w, "ALL_W6_18")
    masks = {
        "POOL_OVER_5": w.pool_over_5.eq(1),
        "POOL_UNDER_5": w.pool_under_5.eq(1),
        "POOL_ABS_5": w.pool_abs_5.eq(1),
        "NON_EXTREME_ABS_LT3": w.non_extreme_abs_lt3.eq(1),
    }
    bins = pd.concat([score_population(w.loc[mask], label) for label, mask in masks.items()], ignore_index=True)

    components = pd.DataFrame([
        {"component": "OFFENSIVE_PLAYS", **metric_pair(w.actual_off_plays, w.pred_off_plays)},
        {"component": "EFFECTIVE_RUSH_RATE", **metric_pair(w.actual_effective_rush_rate, w.pred_effective_rush_rate)},
    ])

    def recovery(table: pd.DataFrame, pop: str, arm: str) -> float:
        q = table.loc[table.population.eq(pop) & table.arm.eq(arm), "mae_recovery_vs_base"]
        return float(q.iloc[0]) if len(q) else np.nan

    play_rec = recovery(overall, "ALL_W6_18", "ORACLE_PLAYS")
    rate_rec = recovery(overall, "ALL_W6_18", "ORACLE_RUSH_RATE")
    play_over = recovery(bins, "POOL_OVER_5", "ORACLE_PLAYS")
    rate_over = recovery(bins, "POOL_OVER_5", "ORACLE_RUSH_RATE")
    play_under = recovery(bins, "POOL_UNDER_5", "ORACLE_PLAYS")
    rate_under = recovery(bins, "POOL_UNDER_5", "ORACLE_RUSH_RATE")

    if play_rec - rate_rec >= 0.50 and play_over >= rate_over and play_under >= rate_under:
        disposition = "PLAY_VOLUME_DOMINANT"
    elif rate_rec - play_rec >= 0.50 and rate_over >= play_over and rate_under >= play_under:
        disposition = "RUSH_RATE_DOMINANT"
    else:
        disposition = "MIXED_TOTAL_RUSH_MECHANICS"

    base = overall.loc[overall.arm.eq("BASE_M94C_TOTAL_RUSH")].iloc[0]
    base_identity_err = float((w.BASE_M94C_TOTAL_RUSH - (w.pred_off_plays * w.pred_effective_rush_rate)).abs().max())
    actual_identity_err = float((w.ORACLE_BOTH - w.actual_team_rush_att).abs().max())
    integrity_pass = int(
        len(w) == EXPECTED_W6_18_N
        and base_identity_err <= 1e-9
        and actual_identity_err <= 1e-9
        and abs(float(base["mae"]) - EXPECTED_BASE_MAE) <= 1e-9
        and abs(float(base["rmse"]) - EXPECTED_BASE_RMSE) <= 1e-9
        and abs(float(base["bias"]) - EXPECTED_BASE_BIAS) <= 1e-9
        and abs(float(base["corr"]) - EXPECTED_BASE_CORR) <= 1e-9
    )
    if not integrity_pass:
        disposition = "STACK6I_INTEGRITY_FAILURE_DO_NOT_INTERPRET"

    integrity = pd.DataFrame([{
        "m94c_rows": int(len(one(a.m94c_root, "m94c_2025_team_trace.csv"))),
        "stack6h_rows": int(len(one(a.stack6h_root, "stack6h_team_trace.csv"))),
        "joined_rows": int(len(trace)),
        "w6_18_team_games": int(len(w)),
        "base_factorization_max_abs_error": base_identity_err,
        "actual_factorization_max_abs_error": actual_identity_err,
        "expected_base_mae": EXPECTED_BASE_MAE,
        "observed_base_mae": float(base["mae"]),
        "expected_base_rmse": EXPECTED_BASE_RMSE,
        "observed_base_rmse": float(base["rmse"]),
        "expected_base_bias": EXPECTED_BASE_BIAS,
        "observed_base_bias": float(base["bias"]),
        "expected_base_corr": EXPECTED_BASE_CORR,
        "observed_base_corr": float(base["corr"]),
        "integrity_pass": integrity_pass,
        "fitted_models": 0,
        "hyperparameter_search": 0,
        "feature_search": 0,
        "threshold_search": 0,
        "sportsbook_inputs": 0,
        "actual_plays_used_as_oracle_only": 1,
        "actual_effective_rush_rate_used_as_oracle_only": 1,
    }])

    disposition_df = pd.DataFrame([{
        "oracle_plays_mae_recovery": play_rec,
        "oracle_rush_rate_mae_recovery": rate_rec,
        "rush_rate_minus_play_recovery": rate_rec - play_rec,
        "pool_over5_plays_recovery": play_over,
        "pool_over5_rush_rate_recovery": rate_over,
        "pool_under5_plays_recovery": play_under,
        "pool_under5_rush_rate_recovery": rate_under,
        "disposition": disposition,
        "production_change": 0,
        "predictive_model_authorized": 0,
    }])

    trace.to_csv(a.out_dir / "stack6i_team_trace.csv", index=False)
    components.to_csv(a.out_dir / "stack6i_component_scores.csv", index=False)
    overall.to_csv(a.out_dir / "stack6i_overall_oracle_scores.csv", index=False)
    bins.to_csv(a.out_dir / "stack6i_bin_oracle_scores.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6i_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6i_disposition.csv", index=False)

    print("=== STACK6I integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6I component scores ===")
    print(components.to_string(index=False))
    print("=== STACK6I overall oracle scores ===")
    print(overall.to_string(index=False))
    print("=== STACK6I bin oracle scores ===")
    print(bins.to_string(index=False))
    print("=== STACK6I disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6I_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
