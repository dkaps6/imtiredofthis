import pandas as pd

from scripts.research.evaluate_player_high_value_opportunity_persistence_v1 import (
    attach_predictors,
    fam,
)


def test_family_mapping():
    assert fam("RB") == "RB"
    assert fam("FB") == "RB"
    assert fam("WR") == "WR"
    assert fam("SWR") == "WR"
    assert fam("TE") == "TE"
    assert fam("QB") == ""


def test_predictors_are_strictly_prior_and_same_team():
    u = pd.DataFrame([
        {"player_id":"p1","team":"AAA","season":2025,"week":1,"opportunities":4,"team_opp":10,"target_hv":1,"target_team_hv":2,"actual_hv_share":0.5},
        {"player_id":"p1","team":"AAA","season":2025,"week":2,"opportunities":5,"team_opp":10,"target_hv":2,"target_team_hv":2,"actual_hv_share":1.0},
        {"player_id":"p1","team":"AAA","season":2025,"week":3,"opportunities":3,"team_opp":10,"target_hv":1,"target_team_hv":2,"actual_hv_share":0.5},
        {"player_id":"p1","team":"AAA","season":2025,"week":4,"opportunities":2,"team_opp":10,"target_hv":0,"target_team_hv":2,"actual_hv_share":0.0},
    ])
    z = attach_predictors(u)
    assert len(z) == 1
    r = z.iloc[0]
    assert int(r["week"]) == 4
    assert abs(float(r["baseline_pred"]) - (12.0/30.0)) < 1e-12
    assert abs(float(r["candidate_pred"]) - (4.0/6.0)) < 1e-12


def test_prior_high_value_support_gate_is_enforced():
    u = pd.DataFrame([
        {"player_id":"p1","team":"AAA","season":2025,"week":1,"opportunities":4,"team_opp":10,"target_hv":0,"target_team_hv":1,"actual_hv_share":0.0},
        {"player_id":"p1","team":"AAA","season":2025,"week":2,"opportunities":4,"team_opp":10,"target_hv":0,"target_team_hv":1,"actual_hv_share":0.0},
        {"player_id":"p1","team":"AAA","season":2025,"week":3,"opportunities":4,"team_opp":10,"target_hv":0,"target_team_hv":1,"actual_hv_share":0.0},
        {"player_id":"p1","team":"AAA","season":2025,"week":4,"opportunities":4,"team_opp":10,"target_hv":1,"target_team_hv":1,"actual_hv_share":1.0},
    ])
    z = attach_predictors(u)
    assert z.empty
