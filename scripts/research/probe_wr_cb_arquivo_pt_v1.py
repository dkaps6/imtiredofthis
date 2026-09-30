#!/usr/bin/env python3
"""Sparse exact-URL Arquivo.pt metadata probe for WR/CB historical sources.

Free public archive metadata only. No archived body retrieval, current-page
reacquisition, sportsbook data, target outcomes, editorial grades, or fitting.
A zero-result response is not proof that no historical copy exists elsewhere.
"""
from __future__ import annotations

import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

API="https://arquivo.pt/textsearch"
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 sparse-exact-arquivo"}

def same_url(a:str,b:str)->bool:
    pa,pb=urlparse(a),urlparse(b)
    ha=(pa.hostname or "").lower().removeprefix("www.")
    hb=(pb.hostname or "").lower().removeprefix("www.")
    return ha==hb and pa.path.rstrip("/")==pb.path.rstrip("/")

def parse_ts(raw:str):
    try:
        return datetime.strptime(str(raw),"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (TypeError,ValueError):
        return None

def classify(when:datetime,first:datetime,last:datetime)->str:
    if when < first: return "FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    if when < last: return "PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    return "POST_WEEK_INDEX_ONLY"

def parse_payload(payload,source_url:str)->list[dict]:
    if not isinstance(payload,dict):
        return []
    items=payload.get("response_items")
    if not isinstance(items,list):
        return []
    out=[]
    for item in items:
        if not isinstance(item,dict):
            continue
        original=str(item.get("originalURL",""))
        when=parse_ts(item.get("tstamp"))
        if not same_url(source_url,original) or when is None:
            continue
        out.append({
          "timestamp":str(item.get("tstamp","")),
          "digest":str(item.get("digest","")),
          "original":original,
          "status":str(item.get("status","")),
          "link_to_archive":str(item.get("linkToArchive","")),
          "collection":str(item.get("collection","")),
        })
    return out

def query_exact(source_url:str,start_utc,end_utc)->tuple[list[dict],str,int|None]:
    params={
      "versionHistory":source_url,
      "from":pd.Timestamp(start_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S"),
      "to":pd.Timestamp(end_utc).tz_convert("UTC").strftime("%Y%m%d%H%M%S"),
      "maxItems":"50",
      "prettyPrint":"false",
    }
    last=""
    for attempt in range(2):
        try:
            r=requests.get(API,params=params,headers=HEADERS,timeout=20)
            if r.status_code!=200:
                last=f"HTTP_{r.status_code}"
            else:
                try: payload=r.json()
                except ValueError:
                    last="NOT_JSON"
                else:
                    rows=parse_payload(payload,source_url)
                    estimated=payload.get("estimated_nr_results")
                    try: estimated=int(estimated)
                    except (TypeError,ValueError): estimated=None
                    return rows,("MATCHING_ARQUIVO_RECORDS" if rows
                                 else "NO_EXACT_ARQUIVO_MATCH_NOT_PROOF_OF_ABSENCE"),estimated
        except requests.RequestException as exc:
            last="NETWORK_"+type(exc).__name__
        if attempt==0: time.sleep(1.0)
    return [],last or "UNKNOWN_QUERY_FAILURE",None

def execute(targets:Path,out_dir:Path)->dict:
    df=pd.read_csv(targets)
    seasons=sorted(df.season.astype(int).unique().tolist())
    schedule=_load_schedule(seasons)
    results=[]
    for i,rec in enumerate(df.itertuples(index=False)):
        season,week=int(rec.season),int(rec.week)
        ks=pd.to_datetime(
          schedule.loc[schedule.season.eq(season)&schedule.week.eq(week),"kickoff_utc"],
          utc=True,errors="coerce").dropna().sort_values()
        pub=pd.to_datetime(rec.published_at_utc,utc=True,errors="raise")
        if ks.empty:
            item={"season":season,"week":week,"source_url":str(rec.url),
                  "query_status":"SCHEDULE_MISSING","captures":[]}
        else:
            first,last=ks.iloc[0],ks.iloc[-1]
            captures,status,estimated=query_exact(str(rec.url),pub,last+pd.Timedelta(minutes=1))
            enriched=[{**c,"time_class":classify(
                parse_ts(c["timestamp"]),first.to_pydatetime(),last.to_pydatetime())}
                for c in captures]
            useful=[x for x in enriched if x["time_class"]!="POST_WEEK_INDEX_ONLY"]
            full=[x for x in useful if x["time_class"]=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"]
            partial=[x for x in useful if x["time_class"]=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"]
            preferred=(sorted(full,key=lambda x:x["timestamp"])[-1] if full
                       else sorted(partial,key=lambda x:x["timestamp"])[0] if partial else None)
            item={
              "season":season,"week":week,"source_url":str(rec.url),
              "published_at_utc":pub.isoformat(),
              "first_game_utc":first.isoformat(),"last_game_utc":last.isoformat(),
              "query_status":status,"estimated_nr_results":estimated,
              "captures":enriched,"useful_snapshot_candidates":useful,
              "preferred_candidate":preferred,
            }
        results.append(item)
        if i < len(df)-1: time.sleep(1.0)
    out={
      "contract":"WR_CB_ARQUIVO_PT_SPARSE_EXACT_URL_V1",
      "archive_provider":"Arquivo.pt",
      "exact_url_targets":len(results),
      "metadata_only":True,
      "archive_bodies_fetched":0,
      "current_articles_fetched":0,
      "target_game_outcomes":False,
      "confirmation_outcomes_accessed":False,
      "sportsbook_inputs":False,
      "editorial_grade_used":False,
      "parameters_fit":0,
      "source_model_gate_cleared":False,
      "zero_results_are_not_global_absence_evidence":True,
      "results":results,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"arquivo_pt_sparse_exact_url.json").write_text(
      json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="results"},indent=2,sort_keys=True))
    for x in results:
        print(f'{x["season"]} W{x["week"]}: status={x.get("query_status")} preferred={x.get("preferred_candidate")}')
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--targets",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); execute(a.targets,a.out_dir)
