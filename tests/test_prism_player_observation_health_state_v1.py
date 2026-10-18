import pandas as pd

from scripts.research.audit_prism_player_observation_health_state_v1 import (
    STATE_NO_REPORT_PARTICIPATED,
    STATE_REPORTED_NO_SNAPS,
    STATE_REPORTED_PARTICIPATED,
    construct_states,
    normalize_rosters,
    normalize_schedule,
)


def test_schedule_gate_excludes_bye_week_roster_rows():
    schedule = pd.DataFrame([
        {"season": 2024, "week": 1, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"},
        {"season": 2024, "week": 2, "game_type": "REG", "home_team": "AAA", "away_team": "CCC"},
    ])
    scheduled, _ = normalize_schedule(schedule)

    roster = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "AAA", "gsis_id": "p1", "player_name": "Alpha One", "position": "WR", "status": "ACT"},
        {"season": 2024, "week": 2, "team": "AAA", "gsis_id": "p1", "player_name": "Alpha One", "position": "WR", "status": "ACT"},
        # BBB is on bye in W2 and must not become a false no-snap observation.
        {"season": 2024, "week": 2, "team": "BBB", "gsis_id": "p2", "player_name": "Beta Two", "position": "TE", "status": "ACT"},
    ])
    out, _ = normalize_rosters([(2024, roster)], scheduled)
    assert set(zip(out["week"], out["team"])) == {(1, "AAA"), (2, "AAA")}


def test_observation_states_and_return_flag():
    roster = pd.DataFrame([
        {
            "season": 2024, "week": 1, "team": "AAA", "stable_id": "p1",
            "player": "Alpha One", "name_key": "alphaone", "position_raw": "WR",
            "family": "WR", "roster_status": "", "scheduled_team_week": True,
            "history_key": "ID:p1", "roster_row_id": 0,
        },
        {
            "season": 2024, "week": 2, "team": "AAA", "stable_id": "p1",
            "player": "Alpha One", "name_key": "alphaone", "position_raw": "WR",
            "family": "WR", "roster_status": "", "scheduled_team_week": True,
            "history_key": "ID:p1", "roster_row_id": 1,
        },
        {
            "season": 2024, "week": 3, "team": "AAA", "stable_id": "p1",
            "player": "Alpha One", "name_key": "alphaone", "position_raw": "WR",
            "family": "WR", "roster_status": "", "scheduled_team_week": True,
            "history_key": "ID:p1", "roster_row_id": 2,
        },
    ])
    snaps = pd.DataFrame([
        {"roster_row_id": 1, "matched": True, "offense_snaps": 0.0, "offense_pct": 0.0, "positive_participation": False},
        {"roster_row_id": 2, "matched": True, "offense_snaps": 40.0, "offense_pct": 0.70, "positive_participation": True},
    ])
    injuries = pd.DataFrame([
        {"roster_row_id": 1, "matched": True},
        {"roster_row_id": 2, "matched": True},
    ])

    out, audit = construct_states(roster, snaps, injuries)
    by_week = out.set_index("week")

    assert by_week.loc[1, "observation_state"] != STATE_NO_REPORT_PARTICIPATED
    assert by_week.loc[2, "observation_state"] == STATE_REPORTED_NO_SNAPS
    assert by_week.loc[3, "observation_state"] == STATE_REPORTED_PARTICIPATED
    assert bool(by_week.loc[3, "return_after_reported_nonparticipation"])
    assert audit["chronology_violations"] == 0


def test_positive_no_report_is_ordinary_participation_state():
    roster = pd.DataFrame([{
        "season": 2024, "week": 1, "team": "AAA", "stable_id": "p1",
        "player": "Alpha One", "name_key": "alphaone", "position_raw": "WR",
        "family": "WR", "roster_status": "", "scheduled_team_week": True,
        "history_key": "ID:p1", "roster_row_id": 0,
    }])
    snaps = pd.DataFrame([{
        "roster_row_id": 0, "matched": True, "offense_snaps": 55.0,
        "offense_pct": 0.90, "positive_participation": True,
    }])
    injuries = pd.DataFrame(columns=["roster_row_id", "matched"])
    out, _ = construct_states(roster, snaps, injuries)
    assert out.iloc[0]["observation_state"] == STATE_NO_REPORT_PARTICIPATED
