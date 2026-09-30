#!/usr/bin/env python3
"""Sparse exact-URL CDX probe for WR/CB archive discovery.

Metadata only. Queries a tiny target set sequentially with exact source URLs and
bounded publication-to-week windows. Empty/blocked responses are access states,
never proof of snapshot absence.
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

CDX="https://web.archive.org/cdx/search/cdx"
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 sparse-exact-cdx"}

def same_url(a:str,b:str)->bool:
    pa,pb=urlparse(a),urlparse(b)
    return ((pa.hostname or "").lower().removeprefix("www.") ==
            (pb.hostname or "").lower().removeprefix("www.")
            and pa.path.rstrip("/")==pb.path.rstrip("/"))

def parse_ts(raw:str):
    try:
        return datetime.strptime(str(raw),"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError,TypeError):
        return None

def classify(when:datetime, first:datetime, last:datetime)->str:
    if when < first: return "FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    if when < last: return "PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    return "POST_WEEK_INDEX_ONLY"

def parse_payload(payload, source_url:str)->list[dict]:
    if not isinstance(payload,list) or not payload or not isinstance(payload[0],list):
        return []
    header=payload[0]
    needed={"timestamp","original","statuscode","digest"}
    if not needed.issubset(set(header)):
        return []
    out=[]
    for row in payload[1:]:
        if not isinstance(row,list) or len(row)!=len(header):
            continue
        x=dict(zip(header,row))
        if str(x.get("statuscode",""))!="200":
            continue
        if not same_url(source_url,str(x.get("original",""))):
            continue
        if parse_ts(str(x.get("timestamp",""))) is None:
            continue
        out.append({
          "timestamp":str(x["timestamp"]),
          "digest":str(x.get("digest","")),
          "original":str(x.get("original","")),
        })
    return out

def query_exact(source_url:str, start_utc, end_utc)->tuple[list[dict],str]:
    start=pd.Timestamp(start_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S")
    end=pd.Timestamp(end_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S")
    params={
      "url":source_url,"output":"json",
      "fl":"timestamp,original,statuscode,digest",
      "filter":"statuscode:200","from":start,"to":end,"limit":"20",
    }
    last=""
    for attempt in range(3):
        try:
            r=requests.get(CDX,params=params,headers=HEADERS,timeout=15)
            if r.status_code!=200:
                last=f"HTTP_{r.status_code}"
            else:
                try: payload=r.json()
                except ValueError:
                    last="NOT_JSON"
                else:
                    rows=parse_payload(payload,source_url)
                    return rows,("MATCHING_INDEX_RECORDS" if rows
                                 else "NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE")
        except requests.RequestException as exc:
            last="NETWORK_"+type(exc).__name__
        if attempt<2:
            time.sleep(1.0)
    return [],last or "UNKNOWN_QUERY_FAILURE"

def execute(targets:Path,out_dir:Path)->dict:
    df=pd.read_csv(targets,dtype={"control_expected_timestamp":"string"})
    seasons=sorted(df.season.astype(int).unique().tolist())
    schedule=_load_schedule(seasons)
    results=[]
    for i,rec in enumerate(df.itertuples(index=False)):
        season,week=int(rec.season),int(rec.week)
        ks=pd.to_datetime(
          schedule.loc[schedule.season.eq(season)&schedule.week.eq(week),"kickoff_utc"],
          utc=True,errors="coerce").dropna().sort_values()
        if ks.empty:
            item={"season":season,"week":week,"source_url":str(rec.url),
                  "query_status":"SCHEDULE_MISSING","captures":[]}
        else:
            pub=pd.to_datetime(rec.published_at_utc,utc=True,errors="raise")
            first,last=ks.iloc[0],ks.iloc[-1]
            captures,status=query_exact(str(rec.url),pub,last+pd.Timedelta(minutes=1))
            enriched=[]
            for c in captures:
                enriched.append({**c,"time_class":classify(
                  parse_ts(c["timestamp"]),first.to_pydatetime(),last.to_pydatetime())})
            useful=[x for x in enriched if x["time_class"]!="POST_WEEK_INDEX_ONLY"]
            full=[x for x in useful if x["time_class"]=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"]
            partial=[x for x in useful if x["time_class"]=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"]
            item={
              "season":season,"week":week,"source_url":str(rec.url),
              "published_at_utc":pub.isoformat(),
              "first_game_utc":first.isoformat(),"last_game_utc":last.isoformat(),
              "query_status":status,"captures":enriched,
              "useful_snapshot_candidates":useful,
              "preferred_candidate":(
                sorted(full,key=lambda x:x["timestamp"])[-1] if full
                else sorted(partial,key=lambda x:x["timestamp"])[0] if partial else None),
            }
        results.append(item)
        if i < len(df)-1: time.sleep(1.5)
    expected=""
    if "control_expected_timestamp" in df.columns:
        vals=[str(v) for v in df.control_expected_timestamp.fillna("") if str(v).strip()]
        expected=vals[0] if vals else ""
    recovered=any(
      c.get("timestamp")==expected
      for x in results for c in x.get("captures",[])
    ) if expected else None
    out={
      "contract":"WR_CB_SPARSE_EXACT_CDX_DISCOVERY_V1",
      "exact_url_targets":len(results),
      "sequential_requests":True,
      "control_expected_timestamp":expected,
      "control_recovered":recovered,
      "archive_bodies_fetched":0,"current_articles_fetched":0,
      "target_game_outcomes":False,"sportsbook_inputs":False,
      "parameters_fit":0,"source_model_gate_cleared":False,
      "empty_index_is_not_absence_evidence":True,
      "results":results,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"sparse_exact_cdx_probe.json").write_text(
      json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="results"},indent=2,sort_keys=True))
    for x in results:
        p=x.get("preferred_candidate")
        print(f'{x["season"]} W{x["week"]}: status={x.get("query_status")} preferred={p}')
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--targets",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); execute(a.targets,a.out_dir)
