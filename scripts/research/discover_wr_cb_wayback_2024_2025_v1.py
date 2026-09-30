#!/usr/bin/env python3
"""Bounded exact-URL Wayback index scan for 2024-2025 WR/CB source articles.

Metadata only: no archived body fetch, no current-page reacquisition, no
sportsbook, no target outcomes, no model fitting. Empty/blocked CDX responses
are access states, not proof that no snapshot exists.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _load_schedule

CDX="https://web.archive.org/cdx/search/cdx"
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0"}
SEASONS={2024,2025}

def same_url(a:str,b:str)->bool:
    pa,pb=urlparse(a),urlparse(b)
    ha=(pa.hostname or "").lower().removeprefix("www.")
    hb=(pb.hostname or "").lower().removeprefix("www.")
    return ha==hb and pa.path.rstrip("/")==pb.path.rstrip("/")

def ts(raw:str)->datetime|None:
    try:
        return datetime.strptime(str(raw),"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError,TypeError):
        return None

def classify(capture:datetime, first:datetime, last:datetime)->str:
    if capture < first:
        return "FULL_WEEK_PREGAME_INDEX_CANDIDATE"
    if capture < last:
        return "PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"
    return "POST_WEEK_INDEX_ONLY"

def query_exact(url:str,season:int)->tuple[list[dict],str]:
    params={
      "url":url,"output":"json","filter":"statuscode:200",
      "from":str(season),"to":str(season),"limit":"30",
    }
    last_status=""
    for attempt in range(2):
        try:
            r=requests.get(CDX,params=params,headers=HEADERS,timeout=16)
            if r.status_code!=200:
                last_status=f"HTTP_{r.status_code}"
            else:
                try: payload=r.json()
                except ValueError:
                    last_status="NOT_JSON"
                else:
                    if not isinstance(payload,list) or not payload:
                        return [],"NO_INDEX_MATCH_NOT_PROOF_OF_ABSENCE"
                    header=payload[0]
                    if not isinstance(header,list) or not {"timestamp","original"}.issubset(set(header)):
                        return [],"INVALID_CDX_SCHEMA"
                    out=[]
                    for row in payload[1:]:
                        if not isinstance(row,list) or len(row)!=len(header): continue
                        x=dict(zip(header,row))
                        if not same_url(url,str(x.get("original",""))): continue
                        if str(x.get("statuscode",""))!="200": continue
                        when=ts(str(x.get("timestamp","")))
                        if when is None: continue
                        out.append({
                          "timestamp":str(x["timestamp"]),
                          "digest":str(x.get("digest","")),
                          "original":str(x.get("original","")),
                        })
                    return out,("MATCHING_INDEX_RECORDS" if out else
                                "NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE")
        except requests.RequestException as exc:
            last_status="NETWORK_"+type(exc).__name__
        if attempt==0: time.sleep(1.0)
    return [],last_status or "UNKNOWN_QUERY_FAILURE"

def run(manifest_path:Path,out_dir:Path)->dict:
    manifest=pd.read_csv(manifest_path)
    manifest=manifest.loc[manifest["season"].isin(SEASONS)].copy()
    if manifest.empty: raise RuntimeError("no 2024-2025 manifest rows")
    schedule=_load_schedule(sorted(SEASONS))
    results=[]
    for rec in manifest.sort_values(["season","week"]).itertuples(index=False):
        season,week,url=int(rec.season),int(rec.week),str(rec.url)
        ks=pd.to_datetime(
          schedule.loc[schedule["season"].eq(season)&schedule["week"].eq(week),"kickoff_utc"],
          utc=True,errors="coerce"
        ).dropna().sort_values()
        if ks.empty:
            results.append({"season":season,"week":week,"source_url":url,
                            "query_status":"SCHEDULE_MISSING","captures":[]})
            continue
        first,last=ks.iloc[0].to_pydatetime(),ks.iloc[-1].to_pydatetime()
        captures,status=query_exact(url,season)
        enriched=[]
        for c in captures:
            when=ts(c["timestamp"])
            enriched.append({**c,"time_class":classify(when,first,last)})
        useful=[x for x in enriched if x["time_class"]!="POST_WEEK_INDEX_ONLY"]
        # Choose one preferred verification target per article. Latest before
        # first kickoff is best; otherwise earliest between-games snapshot.
        full=[x for x in useful if x["time_class"]=="FULL_WEEK_PREGAME_INDEX_CANDIDATE"]
        partial=[x for x in useful if x["time_class"]=="PARTIAL_WEEK_PREGAME_INDEX_CANDIDATE"]
        preferred=None
        if full:
            preferred=sorted(full,key=lambda x:x["timestamp"])[-1]
        elif partial:
            preferred=sorted(partial,key=lambda x:x["timestamp"])[0]
        results.append({
          "season":season,"week":week,"source_url":url,
          "first_game_utc":first.isoformat(),"last_game_utc":last.isoformat(),
          "query_status":status,
          "capture_count":len(enriched),
          "full_week_pregame_candidates":len(full),
          "partial_week_pregame_candidates":len(partial),
          "preferred_candidate":preferred,
          "captures":enriched,
        })
        time.sleep(0.55)
    target_count=len(results)
    with_useful=sum(bool(x.get("preferred_candidate")) for x in results)
    full_weeks=sum(bool(x.get("full_week_pregame_candidates")) for x in results)
    partial_only=sum(
      not bool(x.get("full_week_pregame_candidates"))
      and bool(x.get("partial_week_pregame_candidates")) for x in results
    )
    failures={}
    for x in results:
        st=x.get("query_status","")
        if st not in {"MATCHING_INDEX_RECORDS","NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE"}:
            failures[st]=failures.get(st,0)+1
    output={
      "contract":"WR_CB_WAYBACK_2024_2025_EXACT_URL_INDEX_V1",
      "metadata_only":True,
      "article_targets":target_count,
      "articles_with_any_usable_pregame_index_candidate":with_useful,
      "articles_with_full_week_pregame_candidate":full_weeks,
      "articles_with_partial_only_candidate":partial_only,
      "query_failure_counts":failures,
      "empty_index_is_not_absence_evidence":True,
      "archived_body_fetched":False,
      "current_article_reacquired":False,
      "sportsbook_inputs":False,
      "target_game_outcomes":False,
      "parameters_fit":0,
      "source_model_gate_cleared":False,
      "results":results,
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"wayback_2024_2025_index.json").write_text(
      json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps({k:v for k,v in output.items() if k!="results"},
                     indent=2,sort_keys=True))
    # Print compact candidate inventory only; no article body/content.
    for x in results:
        if x.get("preferred_candidate"):
            p=x["preferred_candidate"]
            print(f'CANDIDATE {x["season"]} W{x["week"]} {p["timestamp"]} '
                  f'{p["time_class"]} digest={p["digest"]}')
        elif x.get("query_status")!="MATCHING_INDEX_RECORDS":
            print(f'NO_CANDIDATE {x["season"]} W{x["week"]} status={x.get("query_status")}')
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.manifest,a.out_dir)
