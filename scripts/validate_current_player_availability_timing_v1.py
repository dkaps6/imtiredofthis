#!/usr/bin/env python3
"""Timing-aware official-inactive certification for current pregame slates.

Frozen rule: official inactive sections become REQUIRED at T-75 minutes.
Missing/incomplete required sections fail closed only the affected game. Games at
or after kickoff are locked against new pricing. Endpoint reachability is not a
certification input.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.build._schedule_utils import get_nfl_schedule

REQUIRE_MINUTES=75.0

def ts(v): return pd.to_datetime(v,utc=True,errors="coerce")

def certify(schedule:pd.DataFrame, official:pd.DataFrame, *, asof_utc)->tuple[pd.DataFrame,dict]:
    a=ts(asof_utc)
    if pd.isna(a): raise RuntimeError("asof_utc is not parseable")
    s=schedule.copy(); s.columns=[str(c).lower() for c in s.columns]
    rename={}
    if "home" in s and "home_team" not in s: rename["home"]="home_team"
    if "away" in s and "away_team" not in s: rename["away"]="away_team"
    s=s.rename(columns=rename)
    req={"season","week","home_team","away_team","kickoff_utc"}; miss=req-set(s.columns)
    if miss: raise RuntimeError(f"schedule missing {sorted(miss)}")
    s["home_team"]=s.home_team.map(canon_team); s["away_team"]=s.away_team.map(canon_team); s["kickoff_utc"]=pd.to_datetime(s.kickoff_utc,utc=True,errors="coerce")
    if s.kickoff_utc.isna().any(): raise RuntimeError("schedule contains unparseable kickoff_utc")
    o=official.copy() if official is not None else pd.DataFrame()
    complete=set(); snapshot_by_team={}
    if not o.empty:
        o.columns=[str(c).lower() for c in o.columns]
        if "team" not in o or "section_complete" not in o: raise RuntimeError("official source missing team/section_complete")
        o["team"]=o.team.map(canon_team)
        if "source_asof_utc" not in o: o["source_asof_utc"]=pd.NaT
        o["source_asof_utc"]=pd.to_datetime(o.source_asof_utc,utc=True,errors="coerce")
        good=pd.to_numeric(o.section_complete,errors="coerce").fillna(0).eq(1)
        complete=set(o.loc[good,"team"])
        for team,g in o[good].groupby("team"):
            vals=g.source_asof_utc.dropna()
            snapshot_by_team[team]=vals.max() if len(vals) else pd.NaT
    rows=[]; withheld=[]
    for idx,r in s.reset_index(drop=True).iterrows():
        ko=r.kickoff_utc; mins=(ko-a).total_seconds()/60.0
        hc=r.home_team in complete; ac=r.away_team in complete
        hts=snapshot_by_team.get(r.home_team,pd.NaT); ats=snapshot_by_team.get(r.away_team,pd.NaT)
        snap=max([x for x in [hts,ats] if not pd.isna(x)],default=pd.NaT)
        reasons=[]
        if mins<=0:
            state="KICKED_OFF_LOCKED"; eligible=False; reasons.append("kickoff_at_or_before_asof")
        elif mins>REQUIRE_MINUTES:
            state="NOT_YET_REQUIRED"; eligible=True
        else:
            if not hc: reasons.append(f"missing_complete_section:{r.home_team}")
            if not ac: reasons.append(f"missing_complete_section:{r.away_team}")
            if hc and pd.isna(hts): reasons.append(f"missing_snapshot_timestamp:{r.home_team}")
            if ac and pd.isna(ats): reasons.append(f"missing_snapshot_timestamp:{r.away_team}")
            if hc and not pd.isna(hts) and hts>=ko: reasons.append(f"snapshot_not_pre_kickoff:{r.home_team}")
            if ac and not pd.isna(ats) and ats>=ko: reasons.append(f"snapshot_not_pre_kickoff:{r.away_team}")
            eligible=not reasons
            state="REQUIRED_AND_CERTIFIED" if eligible else "REQUIRED_MISSING_FAIL_CLOSED"
        if not eligible: withheld.extend([r.away_team,r.home_team])
        rows.append({"season":int(r.season),"week":int(r.week),"game_id":str(r.get("game_id",f"{int(r.season)}_{int(r.week)}_{r.away_team}_{r.home_team}")),"away_team":r.away_team,"home_team":r.home_team,"kickoff_utc":ko.isoformat(),"asof_utc":a.isoformat(),"minutes_to_kickoff":float(mins),"official_required":bool(0<mins<=REQUIRE_MINUTES),"away_official_section_complete":bool(ac),"home_official_section_complete":bool(hc),"official_snapshot_asof_utc":"" if pd.isna(snap) else snap.isoformat(),"certification_state":state,"production_eligible":bool(eligible),"failure_reason":"|".join(reasons)})
    out=pd.DataFrame(rows)
    counts=out.certification_state.value_counts().to_dict() if len(out) else {}
    meta={"require_minutes_before_kickoff":REQUIRE_MINUTES,"asof_utc":a.isoformat(),"games":int(len(out)),"eligible_games":int(out.production_eligible.sum()) if len(out) else 0,"withheld_games":int((~out.production_eligible).sum()) if len(out) else 0,"state_counts":{str(k):int(v) for k,v in counts.items()},"withheld_teams":sorted(set(withheld)),"sportsbook_inputs_used":0}
    return out,meta

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--season",type=int,required=True); ap.add_argument("--week",type=int,required=True); ap.add_argument("--official",type=Path,default=Path("data/official_inactives_v1.csv")); ap.add_argument("--asof-utc",default=""); ap.add_argument("--out",type=Path,default=Path("data/current_player_availability_game_certification.csv")); ap.add_argument("--status",type=Path,default=Path("data/current_player_availability_game_certification.json")); a=ap.parse_args()
    sched=get_nfl_schedule(a.season); sched=sched[pd.to_numeric(sched.week,errors="coerce").eq(a.week)].copy()
    if sched.empty: raise RuntimeError(f"no schedule rows for season={a.season} week={a.week}")
    official=pd.read_csv(a.official) if a.official.exists() and a.official.stat().st_size else pd.DataFrame()
    asof=a.asof_utc or datetime.now(timezone.utc).isoformat(); out,meta=certify(sched,official,asof_utc=asof)
    a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); a.status.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(meta,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
