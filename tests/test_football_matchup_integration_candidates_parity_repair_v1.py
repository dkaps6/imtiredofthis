import pandas as pd
import numpy as np

import scripts.research.score_football_matchup_integration_candidates_v1 as core
import scripts.research.rescore_football_matchup_integration_candidates_v1 as m


def _row(season, baseline, position="RB", market="rush_yards"):
    return {
        "season":season,"week":2,"game_id":f"{season}_02_ATL_NO",
        "team":"ATL","opponent":"NO","player_clean_key":"p1",
        "player_identity_key":"id1","position":position,"market":market,
        "actual":baseline+1.0,"baseline_projection":baseline,
    }


def test_hybrid_projection_uses_current_only_for_2022_2023_and_parent_for_secondary():
    current=pd.DataFrame([
        _row(2022,10.0),_row(2023,20.0),_row(2024,999.0),_row(2025,999.0)
    ])
    parent=pd.DataFrame([_row(2024,30.0),_row(2025,40.0)])
    out,drift=m.build_hybrid_projection(current,parent)
    by=dict(zip(out.season,out.baseline_projection))
    assert by=={2022:10.0,2023:20.0,2024:30.0,2025:40.0}
    src=dict(zip(out.season,out.baseline_source))
    assert src[2024]=="FROZEN_RIGHT_TAIL_PARENT"
    assert src[2025]=="FROZEN_RIGHT_TAIL_PARENT"
    assert drift["independent_rebuild_drift_detected"] is True
    assert drift["max_abs_gap"]==969.0


def test_hybrid_team_features_replaces_secondary_with_frozen_authority(monkeypatch):
    rebuilt=pd.DataFrame([
        {"season":2022,"week":2,"team":"ATL","opponent":"NO",
         "off_true_proe":1.0,"off_true_proe__z":1.0,
         "def_pass_rate_faced":1.0,"def_pass_rate_faced__z":1.0,
         "def_pass_success_allowed":1.0,"def_pass_success_allowed__z":1.0},
        {"season":2023,"week":2,"team":"ATL","opponent":"NO",
         "off_true_proe":2.0,"off_true_proe__z":2.0,
         "def_pass_rate_faced":2.0,"def_pass_rate_faced__z":2.0,
         "def_pass_success_allowed":2.0,"def_pass_success_allowed__z":2.0},
        {"season":2024,"week":2,"team":"ATL","opponent":"NO",
         "off_true_proe":999.0,"off_true_proe__z":999.0,
         "def_pass_rate_faced":999.0,"def_pass_rate_faced__z":999.0,
         "def_pass_success_allowed":999.0,"def_pass_success_allowed__z":999.0},
    ])
    monkeypatch.setattr(core,"build_team_features",lambda team,schedule: rebuilt.copy())
    frozen=pd.DataFrame([
        {"season":2024,"week":2,"team":"ATL","opponent":"NO",
         "off_true_proe":3.0,"off_true_proe__z":3.0,
         "def_pass_rate_faced":3.0,"def_pass_rate_faced__z":3.0,
         "def_pass_success_allowed":3.0,"def_pass_success_allowed__z":3.0},
        {"season":2025,"week":2,"team":"ATL","opponent":"NO",
         "off_true_proe":4.0,"off_true_proe__z":4.0,
         "def_pass_rate_faced":4.0,"def_pass_rate_faced__z":4.0,
         "def_pass_success_allowed":4.0,"def_pass_success_allowed__z":4.0},
    ])
    out=m.build_hybrid_team_features(pd.DataFrame(),pd.DataFrame(),frozen)
    z=out.set_index("season")
    assert z.loc[2024,"off_true_proe"]==3.0
    assert z.loc[2025,"def_pass_success_allowed__z"]==4.0
    assert z.loc[2024,"feature_source"]=="FROZEN_PHASE_BC_AUTHORITY"
    assert 999.0 not in out.select_dtypes(include=[np.number]).to_numpy()
