import pandas as pd
from scripts.validate_current_player_availability_timing_v1 import certify

BASE=pd.Timestamp("2026-09-13T16:00:00Z")

def sched(kickoffs):
    rows=[]
    for i,(mins,away,home) in enumerate(kickoffs,1):
        rows.append({"season":2026,"week":1,"game_id":f"g{i}","away_team":away,"home_team":home,"kickoff_utc":BASE+pd.Timedelta(minutes=mins)})
    return pd.DataFrame(rows)

def official(teams, snap="2026-09-13T15:00:00Z"):
    return pd.DataFrame([{"team":t,"player":"","section_complete":1,"source_asof_utc":snap} for t in teams])

def one(minutes,off=None,asof=BASE):
    s=sched([(minutes,"IND","JAX")]); o=pd.DataFrame() if off is None else off
    out,_=certify(s,o,asof_utc=asof); return out.iloc[0]

def test_120_minutes_missing_not_yet_required():
    r=one(120); assert r.certification_state=="NOT_YET_REQUIRED" and bool(r.production_eligible)

def test_76_minutes_missing_not_yet_required():
    r=one(76); assert r.certification_state=="NOT_YET_REQUIRED" and bool(r.production_eligible)

def test_exact_75_missing_fails_closed():
    r=one(75); assert r.certification_state=="REQUIRED_MISSING_FAIL_CLOSED" and not bool(r.production_eligible)

def test_60_minutes_both_complete_pre_kickoff_certified():
    r=one(60,official(["IND","JAX"],"2026-09-13T15:30:00Z")); assert r.certification_state=="REQUIRED_AND_CERTIFIED" and bool(r.production_eligible)

def test_60_minutes_one_section_missing_fails_closed():
    r=one(60,official(["IND"],"2026-09-13T15:30:00Z")); assert r.certification_state=="REQUIRED_MISSING_FAIL_CLOSED" and "JAX" in r.failure_reason

def test_post_kickoff_snapshot_cannot_certify():
    r=one(60,official(["IND","JAX"],"2026-09-13T17:00:00Z")); assert r.certification_state=="REQUIRED_MISSING_FAIL_CLOSED" and "snapshot_not_pre_kickoff" in r.failure_reason

def test_kicked_off_locked():
    s=sched([(-1,"IND","JAX")]); out,_=certify(s,pd.DataFrame(),asof_utc=BASE); r=out.iloc[0]
    assert r.certification_state=="KICKED_OFF_LOCKED" and not bool(r.production_eligible)

def test_multiple_windows_fail_only_imminent_game():
    s=sched([(60,"IND","JAX"),(180,"BUF","NYJ")]); out,meta=certify(s,pd.DataFrame(),asof_utc=BASE)
    g1=out[out.game_id.eq("g1")].iloc[0]; g2=out[out.game_id.eq("g2")].iloc[0]
    assert g1.certification_state=="REQUIRED_MISSING_FAIL_CLOSED" and not bool(g1.production_eligible)
    assert g2.certification_state=="NOT_YET_REQUIRED" and bool(g2.production_eligible)
    assert set(meta["withheld_teams"])=={"IND","JAX"}
    assert meta["sportsbook_inputs_used"]==0 and meta["require_minutes_before_kickoff"]==75.0
