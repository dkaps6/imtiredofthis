import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import ev_roi, signal as prod_signal
from scripts.operations.grade_market_track_record_v1 import american_profit
from scripts.research.fit_and_apply_strong_gate_probability_calibration_v1 import (
    apply_calibration,
    fit_calibrators,
    main,
    run_direction,
    summarize,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _grade_row(p_over, over_odds, under_odds, over_novig, under_novig, actual, line):
    """Mirrors grade_empirical_fair_prob_v1.py's own side/signal/bet_result
    derivation, so the fixture's original ("old") arm columns are exactly
    what the real empirical_fair_prob_detail.csv would contain, not stand-ins.
    """
    p_under = 1.0 - p_over
    ev_over = ev_roi(p_over, over_odds)
    ev_under = ev_roi(p_under, under_odds)
    best_over = ev_under is None or (ev_over is not None and ev_over >= ev_under)
    side = "OVER" if best_over else "UNDER"
    best_ev = ev_over if best_over else ev_under
    best_model_p = p_over if best_over else p_under
    best_market_p = over_novig if best_over else under_novig
    prob_edge = best_model_p - best_market_p
    sig = prod_signal(best_ev, prob_edge)
    chosen_odds = over_odds if best_over else under_odds
    actual_side = "OVER" if actual > line else ("UNDER" if actual < line else "PUSH")
    bet_result = "PUSH" if actual_side == "PUSH" else ("WIN" if side == actual_side else "LOSS")
    unit_result = american_profit(chosen_odds) if bet_result == "WIN" else (-1.0 if bet_result == "LOSS" else 0.0)
    return {
        "p_over": p_over, "side": side, "best_ev": best_ev, "prob_edge": prob_edge,
        "signal": sig, "actual_side": actual_side, "bet_result": bet_result, "unit_result": unit_result,
    }


def _fixture(n_per_season=200, seed=5, extra_market_rows=0):
    """A market whose raw p_over is overconfident: a true win probability in
    a narrow band around 0.5 gets stretched 3x away from 0.5 into the raw
    claimed probability, while the realized outcome is drawn from the true
    (much less extreme) probability. A calibrator fit on (raw p_over,
    realized outcome) should learn to compress the stretch back out.

    Optionally adds a second market (pass_yards) with too few rows to
    calibrate, to exercise the INSUFFICIENT_ROWS path.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2024, 2025):
        for i in range(n_per_season):
            true_p = rng.uniform(0.40, 0.60)
            raw_p_over = float(np.clip(0.5 + 3.0 * (true_p - 0.5), 0.0, 1.0))  # overconfident stretch
            line = 50.0
            is_over = rng.uniform() < true_p
            actual = line + (10.0 if is_over else -10.0)
            over_odds, under_odds = -110, -110
            over_novig, under_novig = 0.5, 0.5
            graded = _grade_row(raw_p_over, over_odds, under_odds, over_novig, under_novig, actual, line)
            rows.append({
                "season": season, "market": "rec_yards",
                "over_odds": over_odds, "under_odds": under_odds,
                "over_novig": over_novig, "under_novig": under_novig,
                "actual": actual, "line": line,
                **graded,
            })
        for i in range(extra_market_rows):
            raw_p_over = 0.6
            line = 200.0
            actual = line + (5.0 if rng.uniform() < 0.5 else -5.0)
            over_odds, under_odds = -110, -110
            over_novig, under_novig = 0.5, 0.5
            graded = _grade_row(raw_p_over, over_odds, under_odds, over_novig, under_novig, actual, line)
            rows.append({
                "season": season, "market": "pass_yards",
                "over_odds": over_odds, "under_odds": under_odds,
                "over_novig": over_novig, "under_novig": under_novig,
                "actual": actual, "line": line,
                **graded,
            })
    return pd.DataFrame(rows)


def test_fit_calibrators_fails_closed_below_min_rows():
    train = pd.DataFrame({
        "season": [2024] * 10, "market": ["pass_yards"] * 10,
        "p_over": np.linspace(0.5, 0.9, 10), "actual": [60.0] * 10, "line": [55.0] * 10,
        "actual_side": ["OVER"] * 10,
    })
    calibrators = fit_calibrators(train)
    assert calibrators["pass_yards"] is None
    assert calibrators["rec_yards"] is None


def test_fit_calibrators_fits_when_enough_rows():
    detail = _fixture()
    train = detail.loc[detail.season.eq(2024)]
    calibrators = fit_calibrators(train)
    assert calibrators["rec_yards"] is not None
    assert calibrators["pass_yards"] is None


def test_calibration_pulls_overconfident_probability_toward_realized_rate():
    detail = _fixture()
    train = detail.loc[detail.season.eq(2024)]
    test = detail.loc[detail.season.eq(2025)]
    calibrators = fit_calibrators(train)
    out = apply_calibration(test, calibrators)
    assert len(out) == len(test)
    # The raw probabilities are a stretched (overconfident) transform of the
    # true probability; a fitted isotonic calibrator trained on the realized
    # outcomes should compress that range back down, not preserve or widen it.
    assert out["calibrated_p_over"].std() < out["p_over"].std()
    # Required downstream columns for re-grading exist and are finite.
    for col in ["best_ev_new", "prob_edge_new", "signal_new", "unit_result_new"]:
        assert col in out.columns
    assert out["signal_new"].isin(["STRONG_EDGE", "LEAN_EDGE", "NO_EDGE"]).all()


def test_summarize_produces_all_no_filter_and_strong_tier():
    detail = _fixture()
    train = detail.loc[detail.season.eq(2024)]
    test = detail.loc[detail.season.eq(2025)]
    calibrated = apply_calibration(test, fit_calibrators(train))
    summary = summarize(calibrated)
    assert set(summary["tier"]) == {"ALL_NO_FILTER", "STRONG_ONLY_PLAY_TIER"}
    all_row = summary.loc[(summary.market.eq("rec_yards")) & (summary.tier.eq("ALL_NO_FILTER"))].iloc[0]
    assert int(all_row["matched_rows"]) == len(test.loc[test.market.eq("rec_yards")])


def test_summarize_reports_both_old_and_new_arms_on_the_same_rows():
    """Codex P1: the summary must let a reader actually compare the frozen
    criterion (does the calibrated tier beat the current tier on the same
    held-out season), not just report post-calibration numbers alone.
    """
    detail = _fixture()
    train = detail.loc[detail.season.eq(2024)]
    test = detail.loc[detail.season.eq(2025)]
    calibrated = apply_calibration(test, fit_calibrators(train))
    summary = summarize(calibrated)
    for col in ["old_win_rate", "old_roi_per_unit", "new_win_rate", "new_roi_per_unit", "roi_delta"]:
        assert col in summary.columns
    all_row = summary.loc[(summary.market.eq("rec_yards")) & (summary.tier.eq("ALL_NO_FILTER"))].iloc[0]
    # Both arms must be computed from the identical scoreable row set --
    # old_roi_per_unit here is NOT simply copied from the pre-existing
    # detail columns wholesale, it is re-derived only over rows the new arm
    # could also score, so the delta is a genuine same-row comparison.
    assert np.isfinite(all_row["old_roi_per_unit"])
    assert np.isfinite(all_row["new_roi_per_unit"])
    assert np.isfinite(all_row["roi_delta"])
    strong_row = summary.loc[(summary.market.eq("rec_yards")) & (summary.tier.eq("STRONG_ONLY_PLAY_TIER"))].iloc[0]
    assert "old_matched_rows" in strong_row.index


def test_summarize_surfaces_insufficient_rows_explicitly_instead_of_omitting():
    """Codex P2: a market with too few fit-season rows to calibrate must
    appear in the summary with an explicit INSUFFICIENT_ROWS status, not
    silently vanish while the CLI still exits 0.
    """
    detail = _fixture(extra_market_rows=5)  # well under MIN_CALIBRATION_ROWS
    train = detail.loc[detail.season.eq(2024)]
    test = detail.loc[detail.season.eq(2025)]
    calibrated = apply_calibration(test, fit_calibrators(train))
    summary = summarize(calibrated)
    pass_rows = summary.loc[summary.market.eq("pass_yards")]
    assert len(pass_rows) == 2  # ALL_NO_FILTER + STRONG_ONLY_PLAY_TIER, both explicit
    assert (pass_rows["status"] == "INSUFFICIENT_ROWS").all()
    assert pass_rows["old_roi_per_unit"].isna().all()
    # The healthy market must still be graded normally in the same summary.
    rec_rows = summary.loc[summary.market.eq("rec_yards")]
    assert (rec_rows["status"] == "OK").all()


def test_run_direction_produces_both_seasons_worth_of_rows():
    detail = _fixture()
    calibrated, summary, coverage = run_direction(detail, fit_season=2024, test_season=2025)
    assert (calibrated["fit_season"] == 2024).all()
    assert (calibrated["test_season"] == 2025).all()
    assert len(coverage) >= 1


def test_cli_end_to_end_via_subprocess(tmp_path):
    detail = _fixture()
    detail_path = tmp_path / "empirical_fair_prob_detail.csv"
    detail.to_csv(detail_path, index=False)
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(
                _REPO_ROOT,
                "scripts/research/fit_and_apply_strong_gate_probability_calibration_v1.py",
            ),
            "--detail", str(detail_path),
            "--out-dir", str(out_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(out_dir / "strong_gate_calibration_summary.csv")
    assert set(summary["fit_season"].unique()) == {2024, 2025}
    assert set(summary["tier"]) == {"ALL_NO_FILTER", "STRONG_ONLY_PLAY_TIER"}


def test_main_fails_closed_on_more_than_two_seasons(tmp_path):
    detail = _fixture()
    extra = detail.loc[detail.season.eq(2024)].copy()
    extra["season"] = 2023
    detail = pd.concat([detail, extra], ignore_index=True)
    detail_path = tmp_path / "detail.csv"
    detail.to_csv(detail_path, index=False)
    out_dir = tmp_path / "out"

    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["prog", "--detail", str(detail_path), "--out-dir", str(out_dir)]
    try:
        with pytest.raises(RuntimeError, match="expected exactly 2 seasons"):
            main()
    finally:
        _sys.argv = old_argv
