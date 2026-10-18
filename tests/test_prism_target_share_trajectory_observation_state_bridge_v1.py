import pandas as pd

from scripts.research.audit_prism_target_share_trajectory_observation_state_bridge_v1 import (
    NOT_ON_SCOPED_ROSTER,
    PARTICIPATION_CERTIFIED,
    REPORTED_NO_SNAPS,
    UNRESOLVED_NO_SNAPS,
    build_bridge,
    read_health,
)


def _trajectory():
    return pd.DataFrame([{
        "season": 2024,
        "week": 5,
        "team": "AAA",
        "player_clean_key": "alphaone",
        "receiver_id": "p1",
        "position_group": "WR",
        "prior_team_games": 4,
        "recent2_team_targets": 60.0,
        "recent2_player_targets": 8.0,
        "recent2_share": 8/60,
        "earlier_share": 6/60,
        "trajectory_delta": 2/60,
        "feature_max_week": 4,
    }])


def _health(states):
    rows = []
    # Include another player every week so the team-week calendar exists even
    # when p1 is not on the scoped roster.
    for week in range(1, 5):
        rows.append({
            "season": 2024, "week": week, "team": "AAA", "stable_id": "other",
            "player": "Other Player", "family": "WR",
            "observation_state": "NO_REPORT_PARTICIPATED",
            "positive_participation": True,
        })
    for week, state, pos in states:
        rows.append({
            "season": 2024, "week": week, "team": "AAA", "stable_id": "p1",
            "player": "Alpha One", "family": "WR",
            "observation_state": state,
            "positive_participation": pos,
        })
    return pd.DataFrame(rows)


def test_recent2_clean_when_both_recent_games_participated():
    health = _health([
        (3, "NO_REPORT_PARTICIPATED", True),
        (4, "REPORTED_INJURY_PARTICIPATED", True),
    ])
    bridge, audit = build_bridge(_trajectory(), health)
    row = bridge.iloc[0]
    assert audit["feature_max_week_mismatches"] == 0
    assert row["recent2_older_state"] == PARTICIPATION_CERTIFIED
    assert row["recent1_state"] == PARTICIPATION_CERTIFIED
    assert bool(row["recent2_participation_certified"])
    assert not bool(row["recent2_observation_contaminated"])


def test_reported_no_snap_marks_recent_window_contaminated():
    health = _health([
        (3, "NO_REPORT_PARTICIPATED", True),
        (4, "REPORTED_INJURY_NO_OFFENSIVE_SNAPS", False),
    ])
    bridge, _ = build_bridge(_trajectory(), health)
    row = bridge.iloc[0]
    assert row["recent1_state"] == REPORTED_NO_SNAPS
    assert bool(row["recent2_observation_contaminated"])
    assert bool(row["contamination_recent1_only"])
    assert bool(row["reported_no_snap_present"])


def test_unresolved_no_snap_is_not_treated_as_healthy_role_evidence():
    health = _health([
        (3, "NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED", False),
        (4, "NO_REPORT_PARTICIPATED", True),
    ])
    bridge, _ = build_bridge(_trajectory(), health)
    row = bridge.iloc[0]
    assert row["recent2_older_state"] == UNRESOLVED_NO_SNAPS
    assert bool(row["recent2_observation_contaminated"])
    assert bool(row["contamination_recent2_older_only"])


def test_missing_player_roster_row_is_explicit_source_state():
    health = _health([
        (4, "NO_REPORT_PARTICIPATED", True),
    ])
    bridge, _ = build_bridge(_trajectory(), health)
    row = bridge.iloc[0]
    assert row["recent2_older_state"] == NOT_ON_SCOPED_ROSTER
    assert bool(row["not_on_scoped_roster_present"])
    assert bool(row["recent2_observation_contaminated"])
