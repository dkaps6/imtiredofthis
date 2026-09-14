from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.research.merge_promoted_stack_component_predictions_v1 import merge_qb_synthesis

KEYS = ["season", "week", "team", "opponent", "player_clean_key"]


def _projection() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "mahomes",
         "market": "pass_yards", "mc_proj": 250.0, "ml_proj": 240.0, "state_proj": 245.0,
         "ensemble_proj": 243.0, "actual": 230.0},
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "kelce",
         "market": "rec_yards", "mc_proj": 50.0, "ml_proj": 55.0, "state_proj": 45.0,
         "ensemble_proj": 51.0, "actual": 60.0},
        {"season": 2024, "week": 2, "team": "KC", "opponent": "BAL", "player_clean_key": "mahomes",
         "market": "pass_yards", "mc_proj": 260.0, "ml_proj": 250.0, "state_proj": 255.0,
         "ensemble_proj": 255.0, "actual": 240.0},
    ])


def _qb_trace() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "opponent": "DEN", "player_clean_key": "mahomes",
         "actual_pass_yards": 230.0, "base_proj": 243.0, "football_synthesis": 235.0,
         "football_residual_correction": -8.0},
    ])


def test_matched_pass_yards_row_replaces_ensemble_proj_not_mc_proj():
    merged, stats = merge_qb_synthesis(_projection(), _qb_trace())
    row = merged.loc[(merged.week == 1) & (merged.player_clean_key == "mahomes")].iloc[0]
    assert row["ensemble_proj"] == 235.0
    assert row["mc_proj"] == 250.0  # untouched -- QB synthesis replaces the final mean, not the MC component
    assert row["qb_m89_synthesis_applied"] == 1
    assert stats["qb_trace_rows_with_finite_synthesis"] == 1
    assert stats["qb_trace_rows_missing_from_projection"] == 0
    assert stats["pass_yards_rows_m89_authorized"] == 1
    assert stats["pass_yards_rows_excluded_no_m89_authority"] == 1


def test_unmatched_pass_yards_row_is_excluded_from_promoted_qb_benchmark():
    merged, stats = merge_qb_synthesis(_projection(), _qb_trace())
    assert merged.loc[(merged.week == 2) & (merged.player_clean_key == "mahomes")].empty
    assert stats["pass_yards_rows_excluded_no_m89_authority"] == 1
    assert len(merged) == 2  # matched QB + untouched non-QB row


def test_non_qb_row_untouched():
    merged, _ = merge_qb_synthesis(_projection(), _qb_trace())
    row = merged.loc[merged.player_clean_key == "kelce"].iloc[0]
    assert row["ensemble_proj"] == 51.0
    assert row["market"] == "rec_yards"


def test_missing_ensemble_proj_column_raises():
    bad = _projection().drop(columns=["ensemble_proj"])
    try:
        merge_qb_synthesis(bad, _qb_trace())
    except RuntimeError as exc:
        assert "ensemble_proj" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing ensemble_proj")


def test_missing_key_column_raises():
    bad = _projection().drop(columns=["opponent"])
    try:
        merge_qb_synthesis(bad, _qb_trace())
    except RuntimeError as exc:
        assert "opponent" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing key column")


def test_duplicate_qb_trace_identity_raises():
    dup = pd.concat([_qb_trace(), _qb_trace()], ignore_index=True)
    try:
        merge_qb_synthesis(_projection(), dup)
    except RuntimeError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for duplicate QB trace identity")


def test_missing_qb_authority_row_from_projection_raises():
    qb = pd.concat([
        _qb_trace(),
        pd.DataFrame([{
            "season": 2024, "week": 3, "team": "KC", "opponent": "BUF", "player_clean_key": "mahomes",
            "actual_pass_yards": 240.0, "base_proj": 250.0, "football_synthesis": 248.0,
            "football_residual_correction": -2.0,
        }]),
    ], ignore_index=True)
    try:
        merge_qb_synthesis(_projection(), qb)
    except RuntimeError as exc:
        assert "authority rows missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing QB authority row")


def test_duplicate_projection_pass_yards_identity_raises():
    dup = pd.concat([_projection(), _projection().iloc[[0]]], ignore_index=True)
    try:
        merge_qb_synthesis(dup, _qb_trace())
    except RuntimeError as exc:
        assert "duplicate pass_yards identities" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for duplicate pass_yards projection identity")


def test_cli_round_trip(tmp_path: Path):
    projection_path = tmp_path / "projection.csv"
    qb_trace_path = tmp_path / "qb_trace.csv"
    out_path = tmp_path / "out.csv"
    _projection().to_csv(projection_path, index=False)
    _qb_trace().to_csv(qb_trace_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.merge_promoted_stack_component_predictions_v1",
            "--projection-file", str(projection_path),
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
    assert matched["ensemble_proj"] == 235.0
    assert matched["mc_proj"] == 250.0
    assert out.loc[(out.week == 2) & (out.player_clean_key == "mahomes")].empty
