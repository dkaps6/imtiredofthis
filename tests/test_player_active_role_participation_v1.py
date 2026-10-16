# Frozen active-role source-audit safeguards.
import pandas as pd

from scripts.research.audit_player_active_role_participation_v1 import (
    family,
    gates,
)


def test_family_mapping():
    assert family("QB") == "QB"
    assert family("RB") == "RB"
    assert family("FB") == "RB"
    assert family("WR") == "WR"
    assert family("SWR") == "WR"
    assert family("TE") == "TE"
    assert family("CB") == ""


def test_source_gate_pass_shape():
    m = {
        "family":"WR",
        "official_teamweeks":544,
        "roster_teamweek_coverage":1.0,
        "stable_roster_identity_rate":1.0,
        "inactive_roster_identity_bridge":0.98,
        "snap_teamweek_coverage":1.0,
        "positive_snap_roster_bridge":0.99,
        "pregame_universe_rows":3000,
        "distinct_players":250,
        "players_with_prior4_same_team_universe_games":180,
        "chronology_violations":0,
    }
    assert all(gates(m).values())


def test_source_gate_fails_closed_on_snap_coverage():
    m = {
        "family":"TE",
        "official_teamweeks":544,
        "roster_teamweek_coverage":1.0,
        "stable_roster_identity_rate":1.0,
        "inactive_roster_identity_bridge":0.98,
        "snap_teamweek_coverage":0.98,
        "positive_snap_roster_bridge":0.99,
        "pregame_universe_rows":1500,
        "distinct_players":150,
        "players_with_prior4_same_team_universe_games":100,
        "chronology_violations":0,
    }
    assert gates(m)["snap_teamweek_coverage_ge_99"] is False
