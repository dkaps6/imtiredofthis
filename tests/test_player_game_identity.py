import pandas as pd
import pytest

from scripts.run_player_form_v2_loader import attach_schedule_with_game_identity


def test_game_id_collision_is_resolved_without_suffix_columns():
    logs = pd.DataFrame({
        "season": [2025],
        "week": [1],
        "team": ["DAL"],
        "player": ["Example Player"],
        "game_id": ["provider-game-123"],
    })
    schedule = pd.DataFrame({
        "season": [2025],
        "week": [1],
        "team": ["DAL"],
        "opponent": ["PHI"],
        "game_id": ["2025_01_DAL_PHI"],
    })

    out = attach_schedule_with_game_identity(logs, schedule)

    assert out.loc[0, "game_id"] == "2025_01_DAL_PHI"
    assert out.loc[0, "source_game_id"] == "provider-game-123"
    assert out.loc[0, "schedule_game_id"] == "2025_01_DAL_PHI"
    assert "game_id_x" not in out.columns
    assert "game_id_y" not in out.columns


def test_fallback_game_id_is_symmetric_for_both_teams():
    logs = pd.DataFrame({
        "season": [2025, 2025],
        "week": [1, 1],
        "team": ["DAL", "PHI"],
        "player": ["Dallas Player", "Philly Player"],
    })
    schedule = pd.DataFrame({
        "season": [2025, 2025],
        "week": [1, 1],
        "team": ["DAL", "PHI"],
        "opponent": ["PHI", "DAL"],
    })

    out = attach_schedule_with_game_identity(logs, schedule)

    assert out["game_id"].nunique() == 1
    assert out["game_id"].iloc[0] == "2025_01_DAL_PHI"


def test_duplicate_schedule_grain_hard_fails():
    logs = pd.DataFrame({
        "season": [2025],
        "week": [1],
        "team": ["DAL"],
        "player": ["Example Player"],
    })
    schedule = pd.DataFrame({
        "season": [2025, 2025],
        "week": [1, 1],
        "team": ["DAL", "DAL"],
        "opponent": ["PHI", "NYG"],
    })

    with pytest.raises(RuntimeError, match="not unique"):
        attach_schedule_with_game_identity(logs, schedule)
