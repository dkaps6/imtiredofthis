import pandas as pd

from scripts.research.audit_prism_playerform_participation_state_bridge_v1 import (
    admitted_weekly,
    raw_weekly_identity,
)


def _raw_weekly():
    return pd.DataFrame([
        {
            "season": 2024,
            "week": 5,
            "player_id": "00-A",
            "player_display_name": "Player A",
            "recent_team": "CAR",
            "position": "WR",
            "targets": 4,
            "receptions": 2,
            "receiving_yards": 18,
            "rushing_attempts": 0,
            "rushing_yards": 0,
            "attempts": 0,
            "passing_yards": 0,
        },
        {
            "season": 2024,
            "week": 5,
            "player_id": "00-B",
            "player_display_name": "Player B",
            "recent_team": "CAR",
            "position": "WR",
            "targets": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "rushing_attempts": 0,
            "rushing_yards": 0,
            "attempts": 0,
            "passing_yards": 0,
        },
    ])


def test_raw_identity_contains_zero_usage_player_but_playerform_does_not():
    raw=_raw_weekly()
    provider=raw_weekly_identity(raw,2024)
    admitted=admitted_weekly(raw,2024)

    assert set(provider["stable_id"]) == {"00-A","00-B"}
    assert set(admitted["stable_id"]) == {"00-A"}


def test_positive_usage_row_is_admitted_under_exact_playerform_gate():
    admitted=admitted_weekly(_raw_weekly(),2024)
    row=admitted.iloc[0]
    assert row["stable_id"] == "00-A"
    assert row["team"] == "CAR"
    assert int(row["week"]) == 5
