import numpy as np
import pandas as pd

from scripts.research import audit_prism_room_competition_state_v1 as audit


def test_prior_snap_mean3_is_same_team_and_strict_prior():
    h = pd.DataFrame(
        [
            {"player_id":"p1","team":"A","season":2024,"week":1,"off_pct":0.40,"ordinal":202401},
            {"player_id":"p1","team":"A","season":2024,"week":2,"off_pct":0.50,"ordinal":202402},
            {"player_id":"p1","team":"A","season":2024,"week":3,"off_pct":0.60,"ordinal":202403},
            {"player_id":"p1","team":"A","season":2024,"week":4,"off_pct":0.70,"ordinal":202404},
            {"player_id":"p1","team":"B","season":2024,"week":4,"off_pct":0.99,"ordinal":202404},
            {"player_id":"p1","team":"A","season":2024,"week":5,"off_pct":1.00,"ordinal":202405},
        ]
    )
    idx={(pid,team):g for (pid,team),g in h.groupby(["player_id","team"])}
    mean,n,bad=audit.prior_snap_mean3(idx,"p1","A",2024,5)
    assert n == 3
    assert bad == 0
    assert abs(mean - 0.60) < 1e-12


def test_room_state_measures_competition_not_transfer():
    active=pd.DataFrame(
        [
            {
                "season":2025,"week":7,"team":"A","family":"RB",
                "player_id":"p1","player_name_key":"one","stable_roster_identity":1,
                "prior_snap_mean3":0.60,"prior_snap_games":3,"chronology_violations":0,
            },
            {
                "season":2025,"week":7,"team":"A","family":"RB",
                "player_id":"p2","player_name_key":"two","stable_roster_identity":1,
                "prior_snap_mean3":0.30,"prior_snap_games":3,"chronology_violations":0,
            },
        ]
    )
    inactive=pd.DataFrame(
        [
            {
                "season":2025,"week":7,"team":"A","family":"RB",
                "inactive_name_key":"three","player_id":"p3",
                "stable_inactive_identity":1,"prior_snap_mean3":0.45,
                "prior_snap_games":3,"chronology_violations":0,
            }
        ]
    )
    out=audit.build_state_rows(active,inactive).sort_values("player_id")
    assert len(out) == 2
    assert np.allclose(out["active_room_prior_snap_mass"],0.90)
    assert np.allclose(out["inactive_room_prior_snap_mass"],0.45)
    assert np.allclose(out["competition_release_fraction"],1/3)
    assert np.all(out["finite_room_state"])
    p1=out.loc[out.player_id.eq("p1")].iloc[0]
    assert abs(p1.player_active_room_snap_share - (0.60/0.90)) < 1e-12
    # No output field allocates the inactive player's workload to a successor.
    assert "transferred_opportunity" not in out.columns
    assert "vacated_share" not in out.columns


def test_source_gate_requires_frozen_coverage():
    m={
        "family":"WR",
        "inactive_stable_identity":0.95,
        "active_stable_identity":0.99,
        "inactive_prior_snap_coverage":0.90,
        "active_prior_snap_coverage":0.95,
        "finite_room_state_coverage":0.90,
        "event_team_games":100,
        "distinct_active_players":100,
        "chronology_violations":0,
    }
    assert all(audit.gates(m).values())
    m["active_prior_snap_coverage"]=0.899
    assert audit.gates(m)["active_prior_snap_ge_90"] is False
