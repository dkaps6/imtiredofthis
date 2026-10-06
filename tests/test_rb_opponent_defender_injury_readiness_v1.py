import pandas as pd
import scripts.research.audit_rb_opponent_defender_injury_readiness_v1 as m


def test_position_groups():
    assert m.classify_defense_position("DE")=="DL"
    assert m.classify_defense_position("DT")=="DL"
    assert m.classify_defense_position("ILB")=="LB"
    assert m.classify_defense_position("CB")=="DB"
    assert m.classify_defense_position("FS")=="DB"
    assert m.classify_defense_position("QB")==""


def test_latest_prior_snap_forbids_same_target_week():
    snaps=pd.DataFrame([
        {"season":2025,"week":3,"team":"PIT","name_key":"x","pfr_id":"p1","defense_snaps":50,"defense_pct":0.8},
        {"season":2025,"week":4,"team":"PIT","name_key":"x","pfr_id":"p1","defense_snaps":60,"defense_pct":0.9},
    ])
    row=pd.Series({"season":2025,"week":4,"team":"PIT","name_key":"x","pfr_id":"p1"})
    got=m.latest_prior_snap(row,snaps)
    assert got["snap_source_week"]==3
    assert got["chronology_valid"] is True


def test_missing_required_injury_season_closes_not_ready():
    injuries={
        2024:pd.DataFrame([{"season":2024,"week":1,"team":"PIT","player":"A","status":"Out"}]),
        2025:pd.DataFrame(),
        2026:pd.DataFrame(),
    }
    snaps={s:pd.DataFrame([{"season":s,"week":1,"team":"PIT","player":"A","defense_pct":.5}]) for s in m.SNAP_SEASONS}
    rosters={s:pd.DataFrame() for s in m.SNAP_SEASONS}
    src=[]
    for s in m.INJURY_SEASONS:
        src.append({"source":"injuries","season":s,"rows":len(injuries[s]),"error":""})
    for s in m.SNAP_SEASONS:
        src.append({"source":"snap_counts","season":s,"rows":len(snaps[s]),"error":""})
        src.append({"source":"weekly_rosters","season":s,"rows":0,"error":""})
    _,result=m.audit(injuries,snaps,rosters,pd.DataFrame(src))
    assert result["disposition"]=="RB_OPPONENT_DEFENDER_INJURY_SOURCE_NOT_READY"
    assert result["target_game_outcomes_read"]==0
    assert result["sportsbook_inputs_used"]==0
