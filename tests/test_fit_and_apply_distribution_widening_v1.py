import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.research.fit_and_apply_distribution_widening_v1 import (
    _row_arrays,
    _summarize,
    apply_and_grade,
    fit_widening_factors,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]
NARROW_SD = 8.0
REALIZED_SD = 20.0
N_PLAYERS = 30
DRAWS = 2000


def _build_fixture(tmp_path):
    rng = np.random.default_rng(0)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    proj_rows, meta_rows, props_rows = [], [], []
    for season, true_sd in ((2024, NARROW_SD), (2025, NARROW_SD * 1.1)):
        arrays = {}
        for i in range(N_PLAYERS):
            mean = 50.0 + i * 0.1
            arr = rng.normal(mean, true_sd, DRAWS)
            proj = float(np.mean(arr))
            actual = proj + rng.normal(0, REALIZED_SD)
            player = f"player{i}"
            game_id = f"{season}_01_KC_BAL"
            key = (season, 1, "KC", "BAL", player, "rec_yards")
            array_key = f"a{i:06d}"
            arrays[array_key] = arr

            proj_rows.append(
                {
                    "season": season, "week": 1, "team": "KC", "opponent": "BAL",
                    "player_clean_key": player, "market": "rec_yards",
                    "game_id": game_id, "proj": proj, "ensemble_proj": proj, "actual": actual,
                }
            )
            meta_rows.append(
                {
                    "season": season, "week": 1, "team": "KC", "opponent": "BAL",
                    "player": player, "player_clean_key": player, "market": "rec_yards",
                    "event_id": "e1", "array_key": array_key, "draws": DRAWS,
                    "mc_mean": proj, "mc_sd": float(np.std(arr, ddof=1)),
                    "npz_file": f"{season}_week_01.npz",
                }
            )
            props_rows.append(
                {
                    "season": season, "week": 1, "game_id": game_id,
                    "player_clean_key": player, "market": "rec_yards",
                    "book": "draftkings", "line": 49.5,
                    "over_odds": -110, "under_odds": -110, "player": player,
                }
            )
        np.savez(dist_dir / f"{season}_week_01.npz", **arrays)

    proj = pd.DataFrame(proj_rows)
    meta = pd.DataFrame(meta_rows)
    props = pd.DataFrame(props_rows)
    return proj, meta, props, dist_dir


def test_fit_widening_factors_recovers_true_dispersion_ratio(tmp_path):
    proj, meta, _props, dist_dir = _build_fixture(tmp_path)
    factors = fit_widening_factors(proj, meta, dist_dir, fit_season=2024)
    assert set(factors) == {"rec_yards"}
    assert factors["rec_yards"] == pytest.approx(REALIZED_SD / NARROW_SD, rel=0.15)


def test_apply_and_grade_never_moves_the_mean(tmp_path):
    proj, meta, props, dist_dir = _build_fixture(tmp_path)
    factors = fit_widening_factors(proj, meta, dist_dir, fit_season=2024)
    summary, _merged = apply_and_grade(proj, meta, props, dist_dir, test_season=2025, widening_factors=factors)

    unwidened = summary.loc[summary.variant.eq("empirical_unwidened") & summary.tier.eq("ALL_NO_FILTER") & summary.market.eq("ALL_MARKETS")]
    widened = summary.loc[summary.variant.eq("empirical_widened") & summary.tier.eq("ALL_NO_FILTER") & summary.market.eq("ALL_MARKETS")]
    assert len(unwidened) == 1
    assert len(widened) == 1
    assert float(unwidened["model_mae"].iloc[0]) == pytest.approx(float(widened["model_mae"].iloc[0]), abs=1e-9)
    assert float(unwidened["vegas_mae"].iloc[0]) == pytest.approx(float(widened["vegas_mae"].iloc[0]), abs=1e-9)


def test_widening_factor_of_one_is_a_no_op(tmp_path):
    proj, meta, props, dist_dir = _build_fixture(tmp_path)
    summary, _merged = apply_and_grade(
        proj, meta, props, dist_dir, test_season=2025, widening_factors={"rec_yards": 1.0}
    )
    unwidened = summary.loc[summary.variant.eq("empirical_unwidened") & summary.tier.eq("ALL_NO_FILTER") & summary.market.eq("ALL_MARKETS")]
    widened = summary.loc[summary.variant.eq("empirical_widened") & summary.tier.eq("ALL_NO_FILTER") & summary.market.eq("ALL_MARKETS")]
    assert float(unwidened["brier"].iloc[0]) == pytest.approx(float(widened["brier"].iloc[0]), abs=1e-9)
    assert float(unwidened["roi_per_unit"].iloc[0]) == pytest.approx(float(widened["roi_per_unit"].iloc[0]), abs=1e-9)


def test_row_arrays_rejects_path_traversal_shard_names(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    meta = pd.DataFrame(
        [
            {
                "season": 2024, "week": 1, "team": "KC", "opponent": "BAL",
                "player_clean_key": "player0", "market": "rec_yards",
                "array_key": "a0", "npz_file": "../escape.npz",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="invalid shard path"):
        _row_arrays(meta, dist_dir)


def test_fit_widening_factors_requires_matched_rows(tmp_path):
    proj, meta, _props, dist_dir = _build_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="no rows to fit widening factors"):
        fit_widening_factors(proj, meta, dist_dir, fit_season=2099)


def test_fit_widening_factors_fails_closed_on_zero_mean_mc_array(tmp_path):
    # rescale_outcomes intentionally leaves an all-zero MC array untouched
    # (it can't rescale toward a nonzero proj by any finite multiplier). A
    # zero-allocation player (e.g. a non-rusher in rush_yards) can produce
    # exactly this array. Silently proceeding would compute row_sd from a
    # distribution that was never actually aligned to proj -- this must
    # fail loud instead, per the same contract grade_empirical_fair_prob_v1
    # already enforces.
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    np.savez(dist_dir / "2024_week_01.npz", a000000=np.zeros(2000))

    proj = pd.DataFrame(
        [
            {
                "season": 2024, "week": 1, "team": "KC", "opponent": "BAL",
                "player_clean_key": "player0", "market": "rush_yards",
                "game_id": "2024_01_KC_BAL", "proj": 12.0, "ensemble_proj": 12.0, "actual": 8.0,
            }
        ]
    )
    meta = pd.DataFrame(
        [
            {
                "season": 2024, "week": 1, "team": "KC", "opponent": "BAL",
                "player": "player0", "player_clean_key": "player0", "market": "rush_yards",
                "array_key": "a000000", "npz_file": "2024_week_01.npz",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="failed to align mean to proj"):
        fit_widening_factors(proj, meta, dist_dir, fit_season=2024)


def test_summarize_excludes_pushes_from_brier_and_log_loss(tmp_path):
    # A row where actual lands exactly on the line is a PUSH -- it must not
    # be scored as a "not over" (y=0) outcome in calibration metrics, the
    # same exclusion grade_empirical_fair_prob_v1's diagnostics already apply.
    detail = pd.DataFrame(
        [
            {
                "market": "rec_yards", "actual": 50.0, "line": 50.0, "p_over": 0.5, "p_under": 0.5,
                "signal": "NO_EDGE", "bet_result": "PUSH", "unit_result": 0.0,
                "model_error": 1.0, "vegas_error": 1.0,
            },
            {
                "market": "rec_yards", "actual": 60.0, "line": 50.0, "p_over": 0.9, "p_under": 0.1,
                "signal": "NO_EDGE", "bet_result": "WIN", "unit_result": 0.9,
                "model_error": 1.0, "vegas_error": 1.0,
            },
        ]
    )
    summary = _summarize(detail)
    row = summary.loc[summary.market.eq("rec_yards") & summary.tier.eq("ALL_NO_FILTER")].iloc[0]
    # Brier/log-loss computed on the single decided (non-push) row only:
    # y=1 (actual > line), p=0.9 -> brier=(0.9-1)**2=0.01
    assert row["brier"] == pytest.approx(0.01, abs=1e-9)
    assert row["log_loss"] == pytest.approx(-np.log(0.9), abs=1e-9)


def test_cli_rejects_identical_fit_and_test_season(tmp_path):
    proj, meta, props, dist_dir = _build_fixture(tmp_path)
    proj_path = tmp_path / "projection_trace.csv"
    props_path = tmp_path / "props.csv"
    proj.to_csv(proj_path, index=False)
    props.to_csv(props_path, index=False)
    meta.to_csv(dist_dir / "combined_metadata.csv", index=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(_REPO_ROOT, "scripts/research/fit_and_apply_distribution_widening_v1.py"),
            "--projection-file", str(proj_path),
            "--distribution-dir", str(dist_dir),
            "--props", str(props_path),
            "--fit-season", "2024",
            "--test-season", "2024",
            "--out-dir", str(tmp_path / "out"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must differ" in result.stderr


def test_cli_ignores_out_of_scope_markets_with_degenerate_mc_arrays(tmp_path):
    # Reproduces the exact real-data CI failure on PR #548: the clean
    # projection trace also carries rush_att (an ensemble-weight
    # consistency-check market, never one of the 5 markets this experiment
    # is scoped to). A zero-carry player there has an all-zero MC array
    # with a nonzero calibrated proj -- rescale_outcomes can't align it,
    # correctly tripping the mean-alignment guard. The fix is for main() to
    # filter to MARKETS before fitting/applying, not to weaken the guard.
    proj, meta, props, dist_dir = _build_fixture(tmp_path)

    out_of_scope_arr = np.zeros(2000)
    np.savez(dist_dir / "2024_week_02.npz", z000000=out_of_scope_arr)
    proj = pd.concat(
        [
            proj,
            pd.DataFrame(
                [
                    {
                        "season": 2024, "week": 2, "team": "KC", "opponent": "BAL",
                        "player_clean_key": "zerocarry", "market": "rush_att",
                        "game_id": "2024_02_KC_BAL", "proj": 0.0535, "ensemble_proj": 0.0535,
                        "actual": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    meta = pd.concat(
        [
            meta,
            pd.DataFrame(
                [
                    {
                        "season": 2024, "week": 2, "team": "KC", "opponent": "BAL",
                        "player": "zerocarry", "player_clean_key": "zerocarry", "market": "rush_att",
                        "event_id": "e2", "array_key": "z000000", "draws": 2000,
                        "mc_mean": 0.0, "mc_sd": 0.0, "npz_file": "2024_week_02.npz",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    proj_path = tmp_path / "projection_trace.csv"
    props_path = tmp_path / "props.csv"
    proj.to_csv(proj_path, index=False)
    props.to_csv(props_path, index=False)
    meta.to_csv(dist_dir / "combined_metadata.csv", index=False)

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(_REPO_ROOT, "scripts/research/fit_and_apply_distribution_widening_v1.py"),
            "--projection-file", str(proj_path),
            "--distribution-dir", str(dist_dir),
            "--props", str(props_path),
            "--fit-season", "2024",
            "--test-season", "2025",
            "--out-dir", str(out_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = pd.read_csv(out_dir / "widening_test2025_fit2024_summary.csv")
    assert "rush_att" not in set(summary["market"])
