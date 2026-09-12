import numpy as np
import pandas as pd
import pytest

from scripts.research.fit_and_apply_distribution_widening_v1 import (
    _row_arrays,
    apply_and_grade,
    fit_widening_factors,
)

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
