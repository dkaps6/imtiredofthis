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
    assert out.loc[0, "season_type"] == "REG"
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


def test_postseason_rows_are_explicitly_excluded_by_schedule_scope():
    logs = pd.DataFrame({
        "season": [2025, 2025],
        "week": [18, 19],
        "team": ["PIT", "PIT"],
        "player": ["Regular Season Player", "Postseason Player"],
        "game_id": ["reg-provider", "post-provider"],
    })
    schedule = pd.DataFrame({
        "season": [2025],
        "week": [18],
        "team": ["PIT"],
        "opponent": ["BAL"],
        "game_id": ["2025_18_BAL_PIT"],
    })

    out = attach_schedule_with_game_identity(logs, schedule)

    assert len(out) == 1
    assert out.iloc[0]["week"] == 18
    assert out.iloc[0]["player"] == "Regular Season Player"
    assert out.iloc[0]["season_type"] == "REG"


def test_in_scope_team_mismatch_still_hard_fails():
    logs = pd.DataFrame({
        "season": [2025],
        "week": [1],
        "team": ["XXX"],
        "player": ["Bad Team Mapping"],
    })
    schedule = pd.DataFrame({
        "season": [2025, 2025],
        "week": [1, 1],
        "team": ["DAL", "PHI"],
        "opponent": ["PHI", "DAL"],
    })

    with pytest.raises(RuntimeError, match="inside the authoritative regular-season scope"):
        attach_schedule_with_game_identity(logs, schedule)
