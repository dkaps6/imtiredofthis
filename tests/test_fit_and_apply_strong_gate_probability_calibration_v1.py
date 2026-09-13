import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.research.fit_and_apply_strong_gate_probability_calibration_v1 import (
    apply_calibration,
    fit_calibrators,
    main,
    run_direction,
    summarize,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fixture(n_per_season=200, seed=5):
    """A market whose raw p_over is overconfident: a true win probability in
    a narrow band around 0.5 gets stretched 3x away from 0.5 into the raw
    claimed probability, while the realized outcome is drawn from the true
    (much less extreme) probability. A calibrator fit on (raw p_over,
    realized outcome) should learn to compress the stretch back out.
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
            rows.append({
                "season": season, "market": "rec_yards",
                "p_over": raw_p_over, "over_odds": over_odds, "under_odds": under_odds,
                "over_novig": 0.5, "under_novig": 0.5,
                "actual": actual, "line": line,
                "actual_side": "OVER" if actual > line else ("UNDER" if actual < line else "PUSH"),
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
