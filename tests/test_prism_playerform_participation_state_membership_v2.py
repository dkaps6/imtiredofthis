import pandas as pd

from scripts.research.audit_prism_playerform_participation_state_membership_v2 import (
    build,
)


def test_exact_membership_is_stable_id_season_week_team_only():
    anomaly = pd.DataFrame([
        {
            "season": 2024,
            "week": 5,
            "team": "CAR",
            "stable_id": "00-A",
            "family": "WR",
            "participation_anomaly_state": "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
            "corroboration_source": "NEXT_SCHEDULED_GAME_INJURY",
            "corroboration_week": 6,
            "player_prior_snap_pct": 0.80,
            "offense_pct": 0.30,
            "snap_retention_ratio": 0.375,
        },
        {
            "season": 2024,
            "week": 5,
            "team": "CAR",
            "stable_id": "00-B",
            "family": "WR",
            "participation_anomaly_state": "PARTICIPATION_COLLAPSE_UNRESOLVED",
            "corroboration_source": "NONE",
            "corroboration_week": None,
            "player_prior_snap_pct": 0.70,
            "offense_pct": 0.25,
            "snap_retention_ratio": 0.357,
        },
    ])
    admitted = pd.DataFrame([
        {
            "season": 2024,
            "week": 5,
            "team": "CAR",
            "stable_id": "00-A",
            "playerform_position": "WR",
        }
    ])

    out = build(anomaly, admitted)
    a = out.loc[out["stable_id"].eq("00-A")].iloc[0]
    b = out.loc[out["stable_id"].eq("00-B")].iloc[0]

    assert bool(a["playerform_row_present"]) is True
    assert a["playerform_position"] == "WR"
    assert bool(b["playerform_row_present"]) is False


def test_same_id_wrong_week_or_team_does_not_match():
    anomaly = pd.DataFrame([
        {
            "season": 2024,
            "week": 5,
            "team": "CAR",
            "stable_id": "00-A",
            "family": "WR",
            "participation_anomaly_state": "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
            "corroboration_source": "CURRENT_WEEK_INJURY",
            "corroboration_week": 5,
            "player_prior_snap_pct": 0.80,
            "offense_pct": 0.30,
            "snap_retention_ratio": 0.375,
        }
    ])
    admitted = pd.DataFrame([
        {
            "season": 2024,
            "week": 6,
            "team": "CAR",
            "stable_id": "00-A",
            "playerform_position": "WR",
        },
        {
            "season": 2024,
            "week": 5,
            "team": "ATL",
            "stable_id": "00-A",
            "playerform_position": "WR",
        },
    ])

    out = build(anomaly, admitted)
    assert bool(out.iloc[0]["playerform_row_present"]) is False
