from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.research.lock_player_target_share_trajectory_future_week_v1 as m


def parent_meta():
    return {
        "season":2026,
        "target_week":6,
        "disposition":"PLAYER_STATE_FUTURE_WEEK_READY",
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "production_changed":False,
        "prior_week_source_completion":{"complete":True},
    }


def test_parent_state_must_be_ready_complete_and_clean(tmp_path):
    p=tmp_path/"parent.json"
    p.write_text(json.dumps(parent_meta()))
    got=m._certify_parent_state(p,2026,6)
    assert got["disposition"]=="PLAYER_STATE_FUTURE_WEEK_READY"

    bad=parent_meta()
    bad["prior_week_source_completion"]={"complete":False}
    p.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        m._certify_parent_state(p,2026,6)


def test_trajectory_feature_preserves_frozen_recent2_vs_earlier_rule():
    old=m.WEEK
    m.WEEK=6
    try:
        team_idx={"IND":pd.DataFrame([
            {"week":1,"game_id":"g1","team_targets":20.0},
            {"week":2,"game_id":"g2","team_targets":20.0},
            {"week":3,"game_id":"g3","team_targets":20.0},
            {"week":4,"game_id":"g4","team_targets":20.0},
            {"week":5,"game_id":"g5","team_targets":20.0},
        ])}
        lookup={
            (1,"g1","IND","P"):2.0,
            (2,"g2","IND","P"):2.0,
            (3,"g3","IND","P"):2.0,
            (4,"g4","IND","P"):6.0,
            (5,"g5","IND","P"):6.0,
        }
        st=m.trajectory_state(team_idx,lookup,"IND","P")
        assert st is not None
        assert abs(st["recent2_share"]-0.30)<1e-12
        assert abs(st["earlier_share"]-0.10)<1e-12
        assert abs(st["trajectory_delta"]-0.20)<1e-12
        assert st["trajectory_feature_max_week"]==5
    finally:
        m.WEEK=old


def test_apply_shadow_preserves_wr1_and_room_mass():
    frame=pd.DataFrame([
        {"event_id":"E","team":"IND","position":"WR","wr_r15_anchor":True,
         "entitlement_tgt_share":0.25,"trajectory_delta":0.20},
        {"event_id":"E","team":"IND","position":"WR","wr_r15_anchor":False,
         "entitlement_tgt_share":0.15,"trajectory_delta":0.20},
        {"event_id":"E","team":"IND","position":"WR","wr_r15_anchor":False,
         "entitlement_tgt_share":0.10,"trajectory_delta":-0.20},
        {"event_id":"E","team":"IND","position":"TE","wr_r15_anchor":False,
         "entitlement_tgt_share":0.12,"trajectory_delta":0.10},
        {"event_id":"E","team":"IND","position":"TE","wr_r15_anchor":False,
         "entitlement_tgt_share":0.08,"trajectory_delta":-0.10},
    ])
    out,audit,team_gap=m.apply_shadow(frame)
    wr=out.loc[out["position_family"].eq("WR")]
    te=out.loc[out["position_family"].eq("TE")]
    anchor=wr.loc[wr["wr_r15_anchor"]].iloc[0]
    assert abs(anchor["shadow_entitlement_tgt_share"]-0.25)<1e-12
    assert abs(float(wr["shadow_entitlement_tgt_share"].sum())-0.50)<1e-12
    assert abs(float(te["shadow_entitlement_tgt_share"].sum())-0.20)<1e-12
    assert team_gap<=1e-12
    assert float(audit["pool_gap"].max())<=1e-12
