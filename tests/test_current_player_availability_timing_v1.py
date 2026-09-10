import pandas as pd
from scripts.validate_current_player_availability_timing_v1 import certify

def sched():
    return pd.DataFrame([
        {'season':2026,'week':1,'game_id':'g1','home_team':'IND','away_team':'HOU','kickoff_utc':'2026-09-13T17:00:00Z'},
        {'season':2026,'week':1,'game_id':'g2','home_team':'KC','away_team':'LV','kickoff_utc':'2026-09-13T20:25:00Z'},
    ])

def off(rows=()):
    return pd.DataFrame(list(rows),columns=['team','player','section_complete','source_asof_utc'])

def game(df,gid): return df.loc[df.game_id.eq(gid)].iloc[0]

def test_120_minutes_no_sections_not_yet_required():
    x,m=certify(sched().iloc[[0]],off(),asof_utc='2026-09-13T15:00:00Z'); r=game(x,'g1'); assert r.certification_state=='NOT_YET_REQUIRED' and bool(r.production_eligible); assert m['sportsbook_inputs_used']==0

def test_76_minutes_no_sections_not_yet_required():
    x,_=certify(sched().iloc[[0]],off(),asof_utc='2026-09-13T15:44:00Z'); assert game(x,'g1').certification_state=='NOT_YET_REQUIRED'

def test_exactly_75_minutes_missing_fails_closed():
    x,_=certify(sched().iloc[[0]],off(),asof_utc='2026-09-13T15:45:00Z'); r=game(x,'g1'); assert r.certification_state=='REQUIRED_MISSING_FAIL_CLOSED' and not bool(r.production_eligible)

def test_60_minutes_both_complete_certified():
    o=off([('IND','',1,'2026-09-13T15:50:00Z'),('HOU','',1,'2026-09-13T15:50:00Z')]); x,_=certify(sched().iloc[[0]],o,asof_utc='2026-09-13T16:00:00Z'); r=game(x,'g1'); assert r.certification_state=='REQUIRED_AND_CERTIFIED' and bool(r.production_eligible)

def test_60_minutes_one_section_fails_closed():
    o=off([('IND','',1,'2026-09-13T15:50:00Z')]); x,_=certify(sched().iloc[[0]],o,asof_utc='2026-09-13T16:00:00Z'); assert game(x,'g1').certification_state=='REQUIRED_MISSING_FAIL_CLOSED'

def test_postkickoff_snapshot_fails_closed():
    o=off([('IND','',1,'2026-09-13T17:01:00Z'),('HOU','',1,'2026-09-13T17:01:00Z')]); x,_=certify(sched().iloc[[0]],o,asof_utc='2026-09-13T16:00:00Z'); assert game(x,'g1').certification_state=='REQUIRED_MISSING_FAIL_CLOSED'

def test_kicked_off_locked():
    x,_=certify(sched().iloc[[0]],off(),asof_utc='2026-09-13T17:00:00Z'); r=game(x,'g1'); assert r.certification_state=='KICKED_OFF_LOCKED' and not bool(r.production_eligible)

def test_multiple_windows_scope_failure():
    x,_=certify(sched(),off(),asof_utc='2026-09-13T16:00:00Z'); assert game(x,'g1').certification_state=='REQUIRED_MISSING_FAIL_CLOSED'; assert game(x,'g2').certification_state=='NOT_YET_REQUIRED'
