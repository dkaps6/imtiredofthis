#!/usr/bin/env python3
"""Small exact-article Wayback availability-API provenance pilot.

Source metadata only. No archive body retrieval, current page fetch, sportsbook,
outcomes, fitting, or 2025 result access. Week 1 2024 is the positive control.
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd, requests
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

API="https://archive.org/wayback/available"
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 exact-url metadata"}
KNOWN_POSITIVE=("2024",1,"20240907005714")

def parse_api(payload):
    if not isinstance(payload,dict): return None,"INVALID_SCHEMA"
    closest=((payload.get("archived_snapshots") or {}).get("closest"))
    if not isinstance(closest,dict) or not closest.get("available"):
        return None,"NO_AVAILABLE_SNAPSHOT_NOT_PROOF_OF_ABSENCE"
    if str(closest.get("status",""))!="200":
        return None,"CLOSEST_NOT_HTTP_200"
    raw=str(closest.get("timestamp",""))
    when=pd.to_datetime(raw,format="%Y%m%d%H%M%S",utc=True,errors="coerce")
    if pd.isna(when): return None,"INVALID_TIMESTAMP"
    return {
      "timestamp":raw,
      "url":str(closest.get("url","")),
      "status":str(closest.get("status","")),
    },"AVAILABLE"

def query(url,target_utc):
    target=pd.Timestamp(target_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S")
    try:
        r=requests.get(API,params={"url":url,"timestamp":target},headers=HEADERS,timeout=12)
        if r.status_code!=200: return {"query_target":target,"query_status":f"HTTP_{r.status_code}","snapshot":None}
        try: payload=r.json()
        except ValueError: return {"query_target":target,"query_status":"NOT_JSON","snapshot":None}
        snap,status=parse_api(payload)
        return {"query_target":target,"query_status":status,"snapshot":snap}
    except requests.RequestException as exc:
        return {"query_target":target,"query_status":"NETWORK_"+type(exc).__name__,"snapshot":None}

def classify(raw,first,last):
    t=pd.to_datetime(raw,format="%Y%m%d%H%M%S",utc=True,errors="coerce")
    if pd.isna(t):return "INVALID_TIMESTAMP"
    t=t.to_pydatetime()
    if t<first:return "FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    if t<last:return "PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    return "POST_WEEK_INDEX_ONLY"

def run(targets,out_dir):
    rows=pd.read_csv(targets)
    schedule=_load_schedule(sorted(rows.season.unique().tolist()))
    tasks=[]
    meta={}
    for rec in rows.itertuples(index=False):
        season,week=int(rec.season),int(rec.week)
        ks=pd.to_datetime(schedule.loc[schedule.season.eq(season)&schedule.week.eq(week),"kickoff_utc"],utc=True,errors="coerce").dropna().sort_values()
        if ks.empty:raise RuntimeError(f"schedule missing {season} W{week}")
        first,last=ks.iloc[0].to_pydatetime(),ks.iloc[-1].to_pydatetime()
        pub=pd.to_datetime(rec.published_at_utc,utc=True,errors="raise").to_pydatetime()
        # Three bounded targets centered on source publication and useful game window.
        qtimes=sorted(set([
          pub + timedelta(minutes=60),
          pub + timedelta(hours=6),
          last - timedelta(hours=6),
        ]))
        key=(season,week)
        meta[key]={"season":season,"week":week,"source_url":str(rec.url),
                   "published_at_utc":pub.isoformat(),"first_game_utc":first.isoformat(),
                   "last_game_utc":last.isoformat(),"lookups":[]}
        for qt in qtimes: tasks.append((key,str(rec.url),qt))
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(query,url,qt):(key,qt) for key,url,qt in tasks}
        for fut in as_completed(futs):
            key,_=futs[fut]; meta[key]["lookups"].append(fut.result())
    results=[]
    for key in sorted(meta):
        item=meta[key]
        first=pd.to_datetime(item["first_game_utc"],utc=True).to_pydatetime()
        last=pd.to_datetime(item["last_game_utc"],utc=True).to_pydatetime()
        seen={}
        for q in item["lookups"]:
            s=q.get("snapshot")
            if not s:continue
            s={**s,"time_class":classify(s["timestamp"],first,last)}
            seen[(s["timestamp"],s["url"])]=s
        snaps=sorted(seen.values(),key=lambda x:x["timestamp"])
        useful=[s for s in snaps if s["time_class"]!="POST_WEEK_INDEX_ONLY"]
        item["distinct_snapshots"]=snaps
        item["useful_snapshot_candidates"]=useful
        item["has_useful_candidate"]=bool(useful)
        results.append(item)
    pc=next(x for x in results if x["season"]==2024 and x["week"]==1)
    positive_control_recovered=any(s["timestamp"]=="20240907005714" for s in pc["distinct_snapshots"])
    output={
      "contract":"WR_CB_WAYBACK_AVAILABLE_API_PILOT_V1",
      "target_articles":len(results),
      "positive_control_expected_timestamp":"20240907005714",
      "positive_control_recovered":positive_control_recovered,
      "targets_with_useful_snapshot_candidate":sum(x["has_useful_candidate"] for x in results),
      "archive_bodies_fetched":0,"current_articles_fetched":0,
      "target_game_outcomes":False,"sportsbook_inputs":False,"parameters_fit":0,
      "source_model_gate_cleared":False,"results":results,
      "note":"Availability closest-snapshot metadata is only a candidate; exact body+digest+rowwise kickoff+two-sided roster verification still required.",
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"wayback_available_api_pilot.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k!="results"},indent=2,sort_keys=True))
    for x in results:
        print(f'{x["season"]} W{x["week"]}: candidates={[s["timestamp"] for s in x["useful_snapshot_candidates"]]}')
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--targets",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();run(a.targets,a.out_dir)
