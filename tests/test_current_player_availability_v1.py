import pandas as pd
from scripts.build.build_current_player_availability_v1 import build


def depth(rows):
    return pd.DataFrame(rows,columns=["team","player","status","role","position","position_group","depth_index"])

def test_rb1_out_promotes_rb2_and_zeroes_old_lead_eligibility():
    d=depth([["IND","Alpha Back","active","RB1","RB","RB",1],["IND","Beta Back","active","RB2","RB","RB",2]])
    inj=pd.DataFrame([["Alpha Back","IND","OUT","OUT"]],columns=["player","team","status","designation"])
    out,_=build(d,inj,pd.DataFrame())
    a=out[out.player.eq("Alpha Back")].iloc[0]; b=out[out.player.eq("Beta Back")].iloc[0]
    assert a.definitive_unavailable==1 and a.eligible_for_opportunity==0 and a.role_after_availability==""
    assert b.definitive_unavailable==0 and b.role_after_availability=="RB1" and b.role_rank_after_availability==1

def test_questionable_starter_remains_eligible():
    d=depth([["IND","Alpha Back","active","RB1","RB","RB",1]])
    inj=pd.DataFrame([["Alpha Back","IND","QUESTIONABLE","QUESTIONABLE"]],columns=["player","team","status","designation"])
    out,_=build(d,inj,pd.DataFrame()); r=out.iloc[0]
    assert r.final_availability_state=="UNCERTAIN" and r.definitive_unavailable==0 and r.role_after_availability=="RB1"

def test_ourlads_inactive_survives_when_injury_report_empty():
    d=depth([["IND","Alpha Back","inactive","RB1","RB","RB",1],["IND","Beta Back","active","RB2","RB","RB",2]])
    out,_=build(d,pd.DataFrame(),pd.DataFrame()); a=out[out.player.eq("Alpha Back")].iloc[0]; b=out[out.player.eq("Beta Back")].iloc[0]
    assert a.final_availability_state=="UNAVAILABLE_DEPTH_SOURCE" and a.definitive_unavailable==1
    assert b.role_after_availability=="RB1"

def test_complete_official_section_inactive_is_strongest_authority():
    d=depth([["IND","Alpha Back","active","RB1","RB","RB",1],["IND","Beta Back","active","RB2","RB","RB",2]])
    inj=pd.DataFrame([["Alpha Back","IND","QUESTIONABLE","QUESTIONABLE"]],columns=["player","team","status","designation"])
    off=pd.DataFrame([["IND","Alpha Back",1],["IND","",1]],columns=["team","player","section_complete"])
    out,_=build(d,inj,off); a=out[out.player.eq("Alpha Back")].iloc[0]
    assert a.final_availability_state=="UNAVAILABLE_OFFICIAL_INACTIVE" and a.availability_authority=="official_inactive" and a.definitive_unavailable==1

def test_absence_from_complete_official_section_is_active_evidence():
    d=depth([["IND","Alpha Back","active","RB1","RB","RB",1]])
    off=pd.DataFrame([["IND","Some Other Player",1]],columns=["team","player","section_complete"])
    out,_=build(d,pd.DataFrame(),off); r=out.iloc[0]
    assert r.official_inactive is False or r.official_inactive==False
    assert r.final_availability_state=="AVAILABLE_OFFICIAL_ACTIVE" and r.definitive_unavailable==0

def test_incomplete_official_section_cannot_clear_ourlads_inactive():
    d=depth([["IND","Alpha Back","inactive","RB1","RB","RB",1]])
    off=pd.DataFrame([["IND","Some Other Player",0]],columns=["team","player","section_complete"])
    out,_=build(d,pd.DataFrame(),off); r=out.iloc[0]
    assert r.official_inactive_section_complete==0
    assert r.final_availability_state=="UNAVAILABLE_DEPTH_SOURCE"

def test_qb1_official_inactive_promotes_qb2():
    d=depth([["IND","Alpha QB","active","QB1","QB","QB",1],["IND","Beta QB","active","QB2","QB","QB",2]])
    off=pd.DataFrame([["IND","Alpha QB",1]],columns=["team","player","section_complete"])
    out,_=build(d,pd.DataFrame(),off); q2=out[out.player.eq("Beta QB")].iloc[0]
    assert q2.role_after_availability=="QB1"

def test_no_sportsbook_dependency_and_metadata_reports_zero():
    d=depth([["IND","Alpha Back","active","RB1","RB","RB",1]])
    _,meta=build(d,pd.DataFrame(),pd.DataFrame())
    assert meta["sportsbook_inputs_used"]==0 and meta["production_wired"] is False
