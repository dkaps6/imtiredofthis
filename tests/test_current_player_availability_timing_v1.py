import pandas as pd
from scripts.build.certify_current_player_availability_timing_v1 import cert

def sched(): return pd.DataFrame([{'season':2026,'week':1,'game_id':'g1','home_team':'IND','away_team':'HOU','kickoff_utc':'2026-09-13T17:00:00Z'},{'season':2026,'week':1,'game_id':'g2','home_team':'KC','away_team':'LV','kickoff_utc':'2026-09-13T20:25:00Z'}])
def off(rows=()): return pd.DataFrame(list(rows),columns=['team','player','section_complete','source_asof_utc'])
def state(df,team): return df.loc[df.team.eq(team),'certification_state'].iloc[0]
def price(df,team): return int(df.loc[df.team.eq(team),'priceable_now'].iloc[0])

def test_tminus91_not_required():
    x,m=cert(sched(),off(),'2026-09-13T15:29:00Z'); assert state(x,'IND')=='NOT_YET_AVAILABLE' and price(x,'IND')==1 and m['sportsbook_inputs_used']==0

def test_tminus90_missing_fails_closed():
    x,_=cert(sched(),off(),'2026-09-13T15:30:00Z'); assert state(x,'IND')=='REQUIRED_MISSING_FAIL_CLOSED' and price(x,'IND')==0

def test_complete_section_certifies():
    x,_=cert(sched(),off([('IND','',1,'2026-09-13T15:31:00Z')]),'2026-09-13T16:30:00Z'); assert state(x,'IND')=='CERTIFIED_OFFICIAL_SECTION' and price(x,'IND')==1

def test_incomplete_section_fails_closed():
    x,_=cert(sched(),off([('IND','',0,'2026-09-13T15:31:00Z')]),'2026-09-13T16:30:00Z'); assert state(x,'IND')=='REQUIRED_MISSING_FAIL_CLOSED'

def test_early_window_cannot_certify_late_window():
    x,_=cert(sched(),off([('IND','',1,'2026-09-13T15:31:00Z')]),'2026-09-13T19:00:00Z'); assert state(x,'IND')=='POST_KICKOFF_NOT_PRICEABLE'; assert state(x,'KC')=='NOT_YET_AVAILABLE'

def test_postkickoff_not_priceable():
    x,_=cert(sched(),off([('IND','',1,'2026-09-13T15:31:00Z')]),'2026-09-13T17:01:00Z'); assert state(x,'IND')=='POST_KICKOFF_NOT_PRICEABLE' and price(x,'IND')==0

def test_postkickoff_snapshot_cannot_certify_pregame():
    x,_=cert(sched(),off([('IND','',1,'2026-09-13T17:01:00Z')]),'2026-09-13T16:30:00Z'); assert state(x,'IND')=='REQUIRED_MISSING_FAIL_CLOSED'
