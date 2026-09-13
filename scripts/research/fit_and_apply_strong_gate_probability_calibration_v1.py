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
    """Recompute the new (calibrated) arm alongside the untouched original arm.

    Rows whose market had too few training rows to fit a calibrator are kept
    (not dropped) with new-arm columns left NaN/None -- summarize() reports
    those markets explicitly as INSUFFICIENT_ROWS rather than silently
    omitting them, and every OK row is a genuine same-row old-vs-new
    comparison rather than only reporting the calibrated side.
    """
    z = test.copy()
    calibrated_p_over = np.full(len(z), np.nan)
    for market, iso in calibrators.items():
        mask = z["market"].eq(market).to_numpy()
        if iso is None or not mask.any():
            continue
        raw = num(z.loc[mask, "p_over"]).to_numpy(dtype=float)
        calibrated_p_over[mask] = iso.predict(raw)
    z["calibrated_p_over"] = calibrated_p_over
    z["calibration_available"] = pd.notna(z["calibrated_p_over"])

    scored = z.loc[z["calibration_available"]].copy()
    scored["p_over_new"] = scored["calibrated_p_over"]
    scored["p_under_new"] = 1.0 - scored["p_over_new"]
    scored["ev_over_new"] = [ev_roi(p, o) for p, o in zip(scored["p_over_new"], scored["over_odds"])]
    scored["ev_under_new"] = [ev_roi(p, o) for p, o in zip(scored["p_under_new"], scored["under_odds"])]

    best_over = scored["ev_under_new"].isna() | (
        scored["ev_over_new"].fillna(-np.inf) >= scored["ev_under_new"].fillna(-np.inf)
    )
    scored["side_new"] = np.where(best_over, "OVER", "UNDER")
    scored["best_ev_new"] = np.where(best_over, scored["ev_over_new"], scored["ev_under_new"])
    scored["best_model_p_new"] = np.where(best_over, scored["p_over_new"], scored["p_under_new"])
    scored["best_market_p_new"] = np.where(best_over, scored["over_novig"], scored["under_novig"])
    scored["prob_edge_new"] = scored["best_model_p_new"] - scored["best_market_p_new"]
    scored["chosen_odds_new"] = np.where(best_over, scored["over_odds"], scored["under_odds"])
    scored["signal_new"] = [signal(e, q) for e, q in zip(scored["best_ev_new"], scored["prob_edge_new"])]

    scored["bet_result_new"] = np.select(
        [scored["actual_side"].eq("PUSH"), scored["side_new"].eq(scored["actual_side"])],
        ["PUSH", "WIN"],
        default="LOSS",
    )
    scored["unit_result_new"] = np.where(
        scored["bet_result_new"].eq("WIN"),
        [american_profit(o) for o in scored["chosen_odds_new"]],
        np.where(scored["bet_result_new"].eq("LOSS"), -1.0, 0.0),
    )

    new_cols = [
        "p_over_new", "p_under_new", "ev_over_new", "ev_under_new", "side_new", "best_ev_new",
        "best_model_p_new", "best_market_p_new", "prob_edge_new", "chosen_odds_new",
        "signal_new", "bet_result_new", "unit_result_new",
    ]
    return z.join(scored[new_cols])


def _side_metrics(g: pd.DataFrame, *, bet_col: str, unit_col: str) -> dict:
    decided = g.loc[g[bet_col].isin(["WIN", "LOSS"])]
    return {
        "matched_rows": int(len(g)),
        "decided_bets": int(len(decided)),
        "win_rate": float(decided[bet_col].eq("WIN").mean()) if len(decided) else np.nan,
        "roi_per_unit": float(decided[unit_col].mean()) if len(decided) else np.nan,
    }


def _insufficient_row(market: str, tier: str, matched_rows: int) -> dict:
    return {
        "market": market, "tier": tier, "status": "INSUFFICIENT_ROWS",
        "matched_rows": matched_rows, "decided_bets": np.nan,
        "old_win_rate": np.nan, "old_roi_per_unit": np.nan,
        "new_win_rate": np.nan, "new_roi_per_unit": np.nan, "roi_delta": np.nan,
    }


def summarize(calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in MARKETS + ["ALL_MARKETS"]:
        g = calibrated if market == "ALL_MARKETS" else calibrated.loc[calibrated["market"].eq(market)]
        if market != "ALL_MARKETS" and not g["calibration_available"].any():
            rows.append(_insufficient_row(market, "ALL_NO_FILTER", int(len(g))))
            rows.append(_insufficient_row(market, "STRONG_ONLY_PLAY_TIER", 0))
            continue

        # Same fixed row set for old vs new (only rows a calibrator actually
        # covered) -- an apples-to-apples comparison, not old-on-everything
        # vs new-on-a-narrower-subset.
        scoreable = g.loc[g["calibration_available"]]

        old_all = _side_metrics(scoreable, bet_col="bet_result", unit_col="unit_result")
        new_all = _side_metrics(scoreable, bet_col="bet_result_new", unit_col="unit_result_new")
        roi_delta_all = (
            new_all["roi_per_unit"] - old_all["roi_per_unit"]
            if np.isfinite(new_all["roi_per_unit"]) and np.isfinite(old_all["roi_per_unit"])
            else np.nan
        )
        rows.append({
            "market": market, "tier": "ALL_NO_FILTER", "status": "OK",
            "matched_rows": old_all["matched_rows"], "decided_bets": old_all["decided_bets"],
            "old_win_rate": old_all["win_rate"], "old_roi_per_unit": old_all["roi_per_unit"],
            "new_win_rate": new_all["win_rate"], "new_roi_per_unit": new_all["roi_per_unit"],
            "roi_delta": roi_delta_all,
        })

        old_strong = scoreable.loc[scoreable["signal"].eq("STRONG_EDGE")]
        new_strong = scoreable.loc[scoreable["signal_new"].eq("STRONG_EDGE")]
        old_s = _side_metrics(old_strong, bet_col="bet_result", unit_col="unit_result")
        new_s = _side_metrics(new_strong, bet_col="bet_result_new", unit_col="unit_result_new")
        roi_delta_strong = (
            new_s["roi_per_unit"] - old_s["roi_per_unit"]
            if np.isfinite(new_s["roi_per_unit"]) and np.isfinite(old_s["roi_per_unit"])
            else np.nan
        )
        rows.append({
            "market": market, "tier": "STRONG_ONLY_PLAY_TIER", "status": "OK",
            # matched_rows/decided_bets describe the NEW (calibrated) tier --
            # the tier a live gate change would actually produce. The OLD
            # tier's own row count is old_matched_rows, since STRONG
            # membership is a translator-dependent outcome, not a fixed set.
            "matched_rows": new_s["matched_rows"], "decided_bets": new_s["decided_bets"],
            "old_matched_rows": old_s["matched_rows"],
            "old_win_rate": old_s["win_rate"], "old_roi_per_unit": old_s["roi_per_unit"],
            "new_win_rate": new_s["win_rate"], "new_roi_per_unit": new_s["roi_per_unit"],
            "roi_delta": roi_delta_strong,
        })
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
