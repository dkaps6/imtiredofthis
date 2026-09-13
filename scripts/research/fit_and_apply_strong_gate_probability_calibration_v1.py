#!/usr/bin/env python3
"""Fit and apply isotonic probability calibration to the STRONG/LEAN gate.

Frozen plan: docs/research/STRONG_GATE_PROBABILITY_CALIBRATION_V1_PLAN.md

Diagnosed defects (Issue #535 checkpoints 30/45): the empirical-MC fair
probability is severely overconfident -- calibration bins show realized
win rate stuck near 50-55% across nearly the entire claimed-probability
range 0.55-0.95 -- and the STRONG gate's second condition (prob_edge>=3pp)
is algebraically redundant with the first (EV>=5%) under realistic vig.
This fixes the root cause (the probability itself) rather than retuning
the EV/edge threshold constants, which stay exactly as in production
(0.05, 0.03).

Genuine two-directional holdout: fit an isotonic regression per market on
one season's decided (non-push) rows, mapping raw empirical p_over to
realized over-rate; freeze; apply blind to the other season. Recompute
best_model_p/prob_edge/best_ev/signal from the calibrated probability
using the unchanged threshold constants and the exact same, unmodified
implied_prob/no_vig/ev_roi/signal functions used everywhere else in this
repo's grading.

Research only. No production/model/weight/threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import ev_roi, signal
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side

MARKETS = ["pass_yards", "rec_yards", "receptions", "rush_rec_yards", "rush_yards"]
MIN_CALIBRATION_ROWS = 100


def fit_calibrators(train: pd.DataFrame) -> dict:
    calibrators: dict[str, IsotonicRegression | None] = {}
    decided = train.loc[train["actual_side"].ne("PUSH")].copy()
    for market in MARKETS:
        g = decided.loc[decided["market"].eq(market)]
        if len(g) < MIN_CALIBRATION_ROWS:
            calibrators[market] = None
            continue
        x = num(g["p_over"]).to_numpy(dtype=float)
        y = (num(g["actual"]) > num(g["line"])).astype(float).to_numpy()
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, y)
        calibrators[market] = iso
    return calibrators


def apply_calibration(test: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    z = test.copy()
    calibrated_p_over = np.full(len(z), np.nan)
    for market, iso in calibrators.items():
        mask = z["market"].eq(market).to_numpy()
        if iso is None or not mask.any():
            continue
        raw = num(z.loc[mask, "p_over"]).to_numpy(dtype=float)
        calibrated_p_over[mask] = iso.predict(raw)

    z["calibrated_p_over"] = calibrated_p_over
    scoreable = pd.notna(z["calibrated_p_over"])
    z = z.loc[scoreable].copy()

    z["p_over_new"] = z["calibrated_p_over"]
    z["p_under_new"] = 1.0 - z["p_over_new"]
    z["ev_over_new"] = [ev_roi(p, o) for p, o in zip(z["p_over_new"], z["over_odds"])]
    z["ev_under_new"] = [ev_roi(p, o) for p, o in zip(z["p_under_new"], z["under_odds"])]

    best_over = z["ev_under_new"].isna() | (
        z["ev_over_new"].fillna(-np.inf) >= z["ev_under_new"].fillna(-np.inf)
    )
    z["side_new"] = np.where(best_over, "OVER", "UNDER")
    z["best_ev_new"] = np.where(best_over, z["ev_over_new"], z["ev_under_new"])
    z["best_model_p_new"] = np.where(best_over, z["p_over_new"], z["p_under_new"])
    z["best_market_p_new"] = np.where(best_over, z["over_novig"], z["under_novig"])
    z["prob_edge_new"] = z["best_model_p_new"] - z["best_market_p_new"]
    z["chosen_odds_new"] = np.where(best_over, z["over_odds"], z["under_odds"])
    z["signal_new"] = [signal(e, q) for e, q in zip(z["best_ev_new"], z["prob_edge_new"])]

    z["bet_result_new"] = np.select(
        [z["actual_side"].eq("PUSH"), z["side_new"].eq(z["actual_side"])],
        ["PUSH", "WIN"],
        default="LOSS",
    )
    z["unit_result_new"] = np.where(
        z["bet_result_new"].eq("WIN"),
        [american_profit(o) for o in z["chosen_odds_new"]],
        np.where(z["bet_result_new"].eq("LOSS"), -1.0, 0.0),
    )
    return z


def _tier_row(g: pd.DataFrame, tier: str, market: str) -> dict:
    decided = g.loc[g["bet_result_new"].isin(["WIN", "LOSS"])]
    return {
        "market": market,
        "tier": tier,
        "matched_rows": int(len(g)),
        "decided_bets": int(len(decided)),
        "win_rate": float(decided["bet_result_new"].eq("WIN").mean()) if len(decided) else np.nan,
        "roi_per_unit": float(decided["unit_result_new"].mean()) if len(decided) else np.nan,
        "mean_calibrated_p_over": float(num(g["p_over_new"]).mean()) if len(g) else np.nan,
    }


def summarize(calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = list(calibrated["market"].unique()) + ["ALL_MARKETS"]
    for market in scopes:
        g = calibrated if market == "ALL_MARKETS" else calibrated.loc[calibrated["market"].eq(market)]
        rows.append(_tier_row(g, "ALL_NO_FILTER", market))
        rows.append(_tier_row(g.loc[g["signal_new"].eq("STRONG_EDGE")], "STRONG_ONLY_PLAY_TIER", market))
    return pd.DataFrame(rows)


def run_direction(detail: pd.DataFrame, *, fit_season: int, test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = detail.loc[detail["season"].eq(fit_season)]
    test = detail.loc[detail["season"].eq(test_season)]
    calibrators = fit_calibrators(train)
    calibrated = apply_calibration(test, calibrators)
    calibrated["fit_season"] = fit_season
    calibrated["test_season"] = test_season
    summary = summarize(calibrated)
    summary["fit_season"] = fit_season
    summary["test_season"] = test_season
    coverage_rows = [
        {"market": m, "fit_season": fit_season, "calibrator_fit": calibrators.get(m) is not None}
        for m in MARKETS
    ]
    return calibrated, summary, pd.DataFrame(coverage_rows)  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail", type=Path, required=True, help="empirical_fair_prob_detail.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if not a.detail.exists() or not a.detail.stat().st_size:
        raise RuntimeError(f"missing detail file: {a.detail}")
    detail = pd.read_csv(a.detail, low_memory=False)
    detail.columns = [str(c).strip().lower() for c in detail.columns]
    detail["season"] = pd.to_numeric(detail["season"], errors="raise").astype(int)
    seasons = sorted(detail["season"].unique().tolist())
    if len(seasons) != 2:
        raise RuntimeError(f"expected exactly 2 seasons in detail file, found {seasons}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_coverage = []
    for fit_season, test_season in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
        calibrated, summary, coverage = run_direction(detail, fit_season=fit_season, test_season=test_season)
        calibrated.to_csv(a.out_dir / f"calibrated_detail_fit{fit_season}_test{test_season}.csv", index=False)
        all_summaries.append(summary)
        all_coverage.append(coverage)

    out = pd.concat(all_summaries, ignore_index=True)
    out.to_csv(a.out_dir / "strong_gate_calibration_summary.csv", index=False)
    pd.concat(all_coverage, ignore_index=True).to_csv(a.out_dir / "strong_gate_calibration_coverage.csv", index=False)

    print("=== STRONG-GATE PROBABILITY CALIBRATION, TWO-DIRECTIONAL HOLDOUT ===")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
