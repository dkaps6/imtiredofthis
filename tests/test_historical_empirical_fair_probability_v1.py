from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.backtest.grade_empirical_fair_probability_v1 import grade_empirical_ab


def _proj(*, ensemble_proj: float = 275.0, mc_proj: float = 275.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "team": "KC",
                "opponent": "BAL",
                "player": "Patrick Mahomes",
                "player_clean_key": "patrickmahomes",
                "position": "QB",
                "market": "pass_yards",
                "game_id": "2024_01_KC_BAL",
                "mc_proj": mc_proj,
                "ml_proj": mc_proj + 2.0,
                "state_proj": mc_proj - 2.0,
                "ensemble_proj": ensemble_proj,
                "actual": 300.0,
            }
        ]
    )


def _props(*, line: float = 265.5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_id": "2024_01_KC_BAL",
                "player_clean_key": "patrickmahomes",
                "market": "pass_yards",
                "book": "draftkings",
                "line": line,
                "over_odds": -110,
                "under_odds": -110,
                "player": "p.mahomes",
            }
        ]
    )


def _write_distribution(
    root: Path,
    outcomes: np.ndarray,
    *,
    opponent: str = "BAL",
    canonical_mc_proj: float = 275.0,
    duplicate_manifest: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray([outcomes], dtype=np.float32)
    npz_name = "sim_distribution_2024_week_01.npz"
    np.savez_compressed(root / npz_name, outcomes=matrix)
    stored = matrix[0].astype(np.float64)
    row = {
        "season": 2024,
        "week": 1,
        "team": "KC",
        "opponent": opponent,
        "player_clean_key": "patrickmahomes",
        "market": "pass_yards",
        "event_id": "BAL|KC",
        "array_row": 0,
        "distribution_file": npz_name,
        "simulation_iterations": len(outcomes),
        "simulation_seed": 43,
        "canonical_mc_proj": canonical_mc_proj,
        "stored_mc_proj": float(stored.mean()),
        "stored_model_sd": float(stored.std(ddof=1)),
        "serialization_abs_error": abs(float(stored.mean()) - canonical_mc_proj),
    }
    rows = [row, dict(row)] if duplicate_manifest else [row]
    pd.DataFrame(rows).to_csv(root / "sim_distribution_2024_week_01_manifest.csv", index=False)


def test_empirical_translator_uses_saved_distribution_and_deflates_fake_strong(tmp_path: Path):
    outcomes = np.array([200.0, 250.0, 300.0, 350.0], dtype=float)
    _write_distribution(tmp_path, outcomes)
    detail, *_ = grade_empirical_ab(
        _proj(), _props(), tmp_path, expected_iterations=4
    )
    row = detail.iloc[0]
    assert row.old_signal == "STRONG_EDGE"
    assert row.new_p_over == pytest.approx(0.5)
    assert row.new_signal == "NO_EDGE"
    assert row.signal_changed == 1


def test_empirical_distribution_is_aligned_to_same_frozen_final_mean(tmp_path: Path):
    outcomes = np.array([200.0, 250.0, 300.0, 350.0], dtype=float)
    _write_distribution(tmp_path, outcomes)
    detail, *_ = grade_empirical_ab(
        _proj(ensemble_proj=330.0),
        _props(line=300.5),
        tmp_path,
        expected_iterations=4,
    )
    row = detail.iloc[0]
    assert row.target_mean == pytest.approx(330.0)
    assert row.new_model_proj == pytest.approx(330.0, abs=1e-3)
    assert row.mean_alignment_abs_error < 1e-3


def test_distribution_opponent_key_mismatch_fails_closed(tmp_path: Path):
    outcomes = np.array([200.0, 250.0, 300.0, 350.0], dtype=float)
    _write_distribution(tmp_path, outcomes, opponent="BUF")
    with pytest.raises(RuntimeError, match="missing empirical distribution"):
        grade_empirical_ab(_proj(), _props(), tmp_path, expected_iterations=4)


def test_distribution_mean_mismatch_fails_closed(tmp_path: Path):
    outcomes = np.array([250.0, 300.0, 325.0, 325.0], dtype=float)  # mean=300, not 275
    _write_distribution(tmp_path, outcomes, canonical_mc_proj=275.0)
    with pytest.raises(RuntimeError, match="stored empirical distribution does not reproduce graded mc_proj"):
        grade_empirical_ab(_proj(), _props(), tmp_path, expected_iterations=4)


def test_duplicate_distribution_identity_fails_closed(tmp_path: Path):
    outcomes = np.array([200.0, 250.0, 300.0, 350.0], dtype=float)
    _write_distribution(tmp_path, outcomes, duplicate_manifest=True)
    with pytest.raises(RuntimeError, match="duplicate identity rows"):
        grade_empirical_ab(_proj(), _props(), tmp_path, expected_iterations=4)
