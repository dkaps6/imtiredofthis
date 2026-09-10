#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import timedelta
from pathlib import Path
import pandas as pd
from scripts._opponent_map import canon_team

LEAD_MINUTES=90

def utc(v): return pd.to_datetime(v,utc=True,errors='coerce')
def cert(schedule:pd.DataFrame, official:pd.DataFrame, asof, source_status:dict|None=None)->tuple[pd.DataFrame,dict]:
    s=schedule.copy(); s.columns=[str(c).lower() for c in s.columns]
    home=next((c for c in ['home_team','home_team_abbr'] if c in s),None); away=next((c for c in ['away_team','away_team_abbr'] if c in s),None)
    if not home or not away or 'kickoff_utc' not in s: raise RuntimeError('schedule missing home/away/kickoff_utc')
    if 'season' not in s or 'week' not in s: raise RuntimeError('schedule missing season/week')
    a=utc(asof)
    if pd.isna(a): raise RuntimeError('invalid asof')
    o=official.copy() if official is not None else pd.DataFrame();
    if not o.empty:
        o.columns=[str(c).lower() for c in o.columns]; o['team']=o.team.map(canon_team); o['snap']=utc(o.get('source_asof_utc'))
        comp=o[pd.to_numeric(o.get('section_complete'),errors='coerce').fillna(0).eq(1)]
        comp_by={t:g.snap.dropna().max() for t,g in comp.groupby('team')}
    else: comp_by={}
    rows=[]
    for _,r in s.iterrows():
        ko=utc(r.kickoff_utc); req=ko-pd.Timedelta(minutes=LEAD_MINUTES)
        for team,opp in [(canon_team(r[home]),canon_team(r[away])),(canon_team(r[away]),canon_team(r[home]))]:
            snap=comp_by.get(team,pd.NaT); complete=pd.notna(snap) and snap<=ko
            if a>=ko: state='POST_KICKOFF_NOT_PRICEABLE'; price=0; reason='as-of at/after kickoff'
            elif a<req: state='NOT_YET_AVAILABLE'; price=1; reason='official inactive section not yet required'
            elif complete: state='CERTIFIED_OFFICIAL_SECTION'; price=1; reason='complete validated team section available in required window'
            else: state='REQUIRED_MISSING_FAIL_CLOSED'; price=0; reason='official inactive section required but not complete/valid'
            rows.append({'season':int(r.season),'week':int(r.week),'game_id':r.get('game_id',''),'team':team,'opponent':opp,'kickoff_utc':ko.isoformat(),'official_required_from_utc':req.isoformat(),'asof_utc':a.isoformat(),'official_section_complete':int(complete),'official_snapshot_utc':'' if pd.isna(snap) else snap.isoformat(),'certification_state':state,'priceable_now':price,'certification_reason':reason})
    out=pd.DataFrame(rows)
    meta={'scheduled_teams':int(out.team.nunique()),'rows':int(len(out)),'not_yet_required':int(out.certification_state.eq('NOT_YET_AVAILABLE').sum()),'certified':int(out.certification_state.eq('CERTIFIED_OFFICIAL_SECTION').sum()),'required_missing_fail_closed':int(out.certification_state.eq('REQUIRED_MISSING_FAIL_CLOSED').sum()),'post_kickoff':int(out.certification_state.eq('POST_KICKOFF_NOT_PRICEABLE').sum()),'sportsbook_inputs_used':0,'lead_minutes':LEAD_MINUTES}
    if source_status:
        meta['official_endpoint_reachable']=bool(source_status.get('endpoint_reachable',False)); meta['official_payload_valid']=bool(source_status.get('payload_valid',False))
    return out,meta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--schedule',type=Path,required=True); ap.add_argument('--official',type=Path,required=True); ap.add_argument('--official-status',type=Path); ap.add_argument('--asof',required=True); ap.add_argument('--out',type=Path,default=Path('data/current_player_availability_timing.csv')); ap.add_argument('--status',type=Path,default=Path('data/current_player_availability_timing_status.json')); a=ap.parse_args()
    sched=pd.read_csv(a.schedule); off=pd.read_csv(a.official); st=json.loads(a.official_status.read_text()) if a.official_status and a.official_status.exists() else {}
    out,meta=cert(sched,off,a.asof,st); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); a.status.write_text(json.dumps(meta,indent=2,sort_keys=True)); print(json.dumps(meta,indent=2,sort_keys=True))
if __name__=='__main__': main()
