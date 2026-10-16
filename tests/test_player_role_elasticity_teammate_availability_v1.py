from scripts.research.audit_player_role_elasticity_teammate_availability_v1 import (
    family, gate_metrics
)


def test_position_family_mapping():
    assert family("RB") == "RB"
    assert family("FB") == "RB"
    assert family("HB") == "RB"
    assert family("WR") == "WR"
    assert family("SWR") == "WR"
    assert family("TE") == "TE"
    assert family("QB") == ""


def test_gate_metrics_pass():
    m = {
        "family": "RB",
        "inactive_roster_identity_bridge": 0.95,
        "position_family_mapping": 1.0,
        "candidate_stable_identity": 0.95,
        "strict_prior_usage_coverage": 0.92,
        "strict_prior_snap_coverage": 0.90,
        "event_team_games": 25,
        "distinct_successors": 35,
        "two_prior_room_state_players": 22,
        "chronology_violations": 0,
    }
    assert all(gate_metrics(m).values())


def test_gate_metrics_fail_closed():
    m = {
        "family": "WR",
        "inactive_roster_identity_bridge": 0.95,
        "position_family_mapping": 1.0,
        "candidate_stable_identity": 0.95,
        "strict_prior_usage_coverage": 0.89,
        "strict_prior_snap_coverage": 0.90,
        "event_team_games": 45,
        "distinct_successors": 55,
        "two_prior_room_state_players": 25,
        "chronology_violations": 0,
    }
    assert gate_metrics(m)["strict_prior_usage_ge_90"] is False
