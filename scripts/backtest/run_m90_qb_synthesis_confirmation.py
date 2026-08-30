#!/usr/bin/env python3
"""M90: exact M89 football-synthesis temporal-rotation confirmation.

No new feature engineering is allowed here. M90 imports M89's frozen football
feature list, history construction, preprocessing, Ridge fit, and ±45-yard
prediction correction cap, then rotates the chronology one year earlier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.run_m89_pregame_synthesis import (
    ALPHA,
    RESIDUAL_CAP,
    FOOTBALL_FEATURES,
    add_history_features,
    attach_market,
    bootstrap_prob,
    fit_candidate,
    load_market,
    load_player_logs,
    load_team_history,
    metric,
    normalize_trace,
    predict_candidate,
)

BOOTSTRAP_GATE = 0.90
MAE_GAIN_GATE = 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-season", type=int, required=True)
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--train-trace", required=True)
    ap.add_argument("--train-team", required=True)
    ap.add_argument("--train-logs", required=True)
    ap.add_argument("--test-trace", required=True)
    ap.add_argument("--test-team", required=True)
    ap.add_argument("--test-logs", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.test_season != args.train_season + 1:
        raise RuntimeError("M90 primary rotation must be one-season-forward train -> test")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = normalize_trace(Path(args.train_trace), args.train_season)
    test = normalize_trace(Path(args.test_trace), args.test_season)

    train = add_history_features(
        train,
        load_team_history(Path(args.train_team)),
        load_player_logs(Path(args.train_logs)),
    )
    test = add_history_features(
        test,
        load_team_history(Path(args.test_team)),
        load_player_logs(Path(args.test_logs)),
    )

    # M89 obtains controlled_environment from the schedule loader bundled with
    # the market table. The football model still uses only FOOTBALL_FEATURES;
    # no total/spread/moneyline/implied-points field is admitted here.
    market = load_market([args.train_season, args.test_season])
    train = attach_market(train, market)
    test = attach_market(test, market)

    missing_train = [c for c in FOOTBALL_FEATURES if c not in train.columns]
    missing_test = [c for c in FOOTBALL_FEATURES if c not in test.columns]
    if missing_train or missing_test:
        raise RuntimeError(
            f"exact M89 football feature contract unavailable; train={missing_train} test={missing_test}"
        )

    model, coverage = fit_candidate(train, FOOTBALL_FEATURES, "football_synthesis")
    pred, correction = predict_candidate(model, test, FOOTBALL_FEATURES)
    test = test.copy()
    test["football_synthesis"] = pred
    test["m90_correction"] = correction

    score = pd.DataFrame([
        metric(test, "base_proj", "base", str(args.test_season)),
        metric(test, "football_synthesis", "football_synthesis", str(args.test_season)),
    ])
    base = score.loc[score.model.eq("base")].iloc[0]
    cand = score.loc[score.model.eq("football_synthesis")].iloc[0]

    actual = pd.to_numeric(test["actual_pass_yards"], errors="coerce").to_numpy(float)
    base_pred = pd.to_numeric(test["base_proj"], errors="coerce").to_numpy(float)
    cand_pred = pd.to_numeric(test["football_synthesis"], errors="coerce").to_numpy(float)
    base_abs = np.abs(base_pred - actual)
    cand_abs = np.abs(cand_pred - actual)
    p_improve = bootstrap_prob(base_abs, cand_abs)

    gates = {
        "mae_gain": float(base.mae - cand.mae),
        "mae_gain_gate": bool((base.mae - cand.mae) >= MAE_GAIN_GATE),
        "rmse_nonworse": bool(cand.rmse <= base.rmse),
        "correlation_nonworse": bool(cand.correlation >= base.correlation),
        "tails_nonincrease": bool(cand.tails100 <= base.tails100),
        "bootstrap_p_improve": float(p_improve),
        "bootstrap_gate": bool(p_improve >= BOOTSTRAP_GATE),
        "sportsbook_features_in_football_model": False,
        "postgame_casebook_features_used_for_prediction": False,
    }
    gates["all_gates_pass"] = bool(
        gates["mae_gain_gate"]
        and gates["rmse_nonworse"]
        and gates["correlation_nonworse"]
        and gates["tails_nonincrease"]
        and gates["bootstrap_gate"]
    )

    score.to_csv(out_dir / "m90_rotated_confirmation_scoreboard.csv", index=False)
    coverage.to_csv(out_dir / "m90_feature_coverage.csv", index=False)

    keep = [
        "season", "week", "team", "opponent", "player_clean_key",
        "actual_pass_yards", "base_proj", "football_synthesis", "m90_correction",
    ]
    trace = test[[c for c in keep if c in test.columns]].copy()
    trace["base_error"] = trace["base_proj"] - trace["actual_pass_yards"]
    trace["synthesis_error"] = trace["football_synthesis"] - trace["actual_pass_yards"]
    trace.to_csv(out_dir / "m90_rotated_confirmation_trace.csv", index=False)

    decision = {
        "migration": "M90",
        "train_season": args.train_season,
        "test_season": args.test_season,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "ridge_alpha": float(ALPHA),
        "residual_cap": float(RESIDUAL_CAP),
        "football_features": FOOTBALL_FEATURES,
        "m89_feature_contract_reused_exactly": True,
        "base": base.to_dict(),
        "football_synthesis": cand.to_dict(),
        "gates": gates,
        "directional_tail_diagnostic": {
            "base_under100": int(base.under100),
            "base_over100": int(base.over100),
            "synthesis_under100": int(cand.under100),
            "synthesis_over100": int(cand.over100),
        },
        "catastrophic_casebook_reopened": False,
        "single_largest_completion_retested": False,
        "disposition": (
            "PROMOTE_M89_FOOTBALL_SYNTHESIS"
            if gates["all_gates_pass"]
            else "DO_NOT_PROMOTE_M89_SYNTHESIS"
        ),
    }
    (out_dir / "m90_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    print("=== M90 ROTATED CONFIRMATION ===")
    print(score.to_string(index=False))
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
