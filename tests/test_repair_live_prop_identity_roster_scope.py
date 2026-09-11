import pandas as pd
import pytest

from scripts._opponent_map import CANON_TEAM_CODES
from scripts.repair_live_prop_identity_v1 import _build_roster_index


def _roles_for(teams: set[str]) -> pd.DataFrame:
    ordered = sorted(teams)
    return pd.DataFrame(
        {
            "team": ordered,
            "player": [f"Player {team}" for team in ordered],
        }
    )


REMAINING_SLATE_TEAMS = set(CANON_TEAM_CODES) - {"LAR", "NE", "SEA", "SF"}


def test_28_team_remaining_slate_passes_when_all_event_teams_present():
    roster = _build_roster_index(
        _roles_for(REMAINING_SLATE_TEAMS),
        required_teams=REMAINING_SLATE_TEAMS,
    )
    assert set(roster) == REMAINING_SLATE_TEAMS


def test_missing_current_event_team_fails_closed():
    with pytest.raises(RuntimeError, match="missing current-event teams: \\['WAS'\\]"):
        _build_roster_index(
            _roles_for(REMAINING_SLATE_TEAMS - {"WAS"}),
            required_teams=REMAINING_SLATE_TEAMS,
        )


def test_full_32_team_roster_still_passes():
    roster = _build_roster_index(
        _roles_for(set(CANON_TEAM_CODES)),
        required_teams=REMAINING_SLATE_TEAMS,
    )
    assert set(roster) == set(CANON_TEAM_CODES)


def test_absent_non_event_teams_are_ignored():
    current_event = {"ARI", "ATL"}
    roster = _build_roster_index(
        _roles_for(current_event),
        required_teams=current_event,
    )
    assert set(roster) == current_event
