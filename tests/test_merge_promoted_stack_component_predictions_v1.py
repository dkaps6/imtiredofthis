from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.research.merge_promoted_stack_component_predictions_v1 import merge_qb_synthesis

KEYS = ["season", "week", "team", "opponent", "player_clean_key"]


def _component() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "mahomes",
         "market": "pass_yards", "mc_proj": 250.0, "ml_proj": 240.0, "state_proj": 245.0, "actual": 230.0},
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "kelce",
         "market": "rec_yards", "mc_proj": 50.0, "ml_proj": 55.0, "state_proj": 45.0, "actual": 60.0},
        {"season": 2024, "week": 2, "team": "KC", "opponent": "BAL", "player_clean_key": "mahomes",
         "market": "pass_yards", "mc_proj": 260.0, "ml_proj": 250.0, "state_proj": 255.0, "actual": 240.0},
    ])


def _qb_trace() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "mahomes",
         "actual_pass_yards": 230.0, "base_proj": 250.0, "football_synthesis": 235.0, "m90_correction": -15.0},
    ])


def test_matched_pass_yards_row_uses_football_synthesis():
    merged, stats = merge_qb_synthesis(_component(), _qb_trace())
    row = merged.loc[(merged.week == 1) & (merged.player_clean_key == "mahomes")].iloc[0]
    assert row["mc_proj"] == 235.0
    assert row["qb_m90_synthesis_applied"] == 1
    assert stats["pass_yards_rows_matched_to_qb_trace"] == 1


def test_unmatched_pass_yards_row_stays_on_generic_ensemble():
    merged, _ = merge_qb_synthesis(_component(), _qb_trace())
    row = merged.loc[(merged.week == 2) & (merged.player_clean_key == "mahomes")].iloc[0]
    assert row["mc_proj"] == 260.0
    assert row["qb_m90_synthesis_applied"] == 0


def test_non_qb_row_untouched():
    merged, _ = merge_qb_synthesis(_component(), _qb_trace())
    row = merged.loc[merged.player_clean_key == "kelce"].iloc[0]
    assert row["mc_proj"] == 50.0
    assert row["market"] == "rec_yards"


def test_missing_key_column_raises():
    bad = _component().drop(columns=["opponent"])
    try:
        merge_qb_synthesis(bad, _qb_trace())
    except RuntimeError as exc:
        assert "opponent" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing key column")


def test_duplicate_qb_trace_identity_raises():
    dup = pd.concat([_qb_trace(), _qb_trace()], ignore_index=True)
    try:
        merge_qb_synthesis(_component(), dup)
    except RuntimeError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for duplicate QB trace identity")


def test_cli_round_trip(tmp_path: Path):
    component_path = tmp_path / "component.csv"
    qb_trace_path = tmp_path / "qb_trace.csv"
    out_path = tmp_path / "out.csv"
    _component().to_csv(component_path, index=False)
    _qb_trace().to_csv(qb_trace_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.merge_promoted_stack_component_predictions_v1",
            "--component-file", str(component_path),
            "--qb-trace", str(qb_trace_path),
            "--out", str(out_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    out = pd.read_csv(out_path)
    matched = out.loc[(out.week == 1) & (out.player_clean_key == "mahomes")].iloc[0]
    assert matched["mc_proj"] == 235.0
