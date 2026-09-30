#!/usr/bin/env python3
"""Verify exact 2022-W5 FantasyAlarm WR/CB page in Common Crawl.

Independent free archive fallback after Wayback exact replay was unavailable.
No outcomes, sportsbook data, editorial grades, fitting, provider bridge,
current-page reacquisition, or fuzzy/manual identity rescue.
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json
from datetime import timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd, requests
from warcio.archiveiterator import ArchiveIterator

from scripts.research.acquire_fantasyalarm_wr_cb_archive_v1 import parse_page
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import (
    DEF_POSITIONS, WR_POSITIONS, _load_rosters, _load_schedule,
)
from scripts.research.audit_verified_wr_cb_snapshot_rows_2024w1_v1 import exact_week_lookup
from scripts.utils.canonical_names import canonicalize_player_name_safe

ARTICLE=("https://www.fantasyalarm.com/articles/nfl/wide-receivers/"
         "2022-fantasy-football-wr-cb-match-up-report-week-5-tyreek-hill-to-burn-the-jets-in-week-5/134887")
INDEX="CC-MAIN-2022-40"
INDEX_URL=f"https://index.commoncrawl.org/{INDEX}-index"
DATA_ROOT="https://data.commoncrawl.org/"
SEASON=2022
WEEK=5
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 exact-one-commoncrawl-source"}

def name_key(value)->str:
    return str(canonicalize_player_name_safe(value)[1] or "").strip()

def _sha1_b32(data:bytes)->str:
    return base64.b32encode(hashlib.sha1(data).digest()).decode("ascii").rstrip("=")

def query_exact()->tuple[list[dict],str]:
    try:
        r=requests.get(INDEX_URL,params={"url":ARTICLE,"output":"json","filter":"status:200"},
                       headers=HEADERS,timeout=20)
        if r.status_code!=200:return [],f"HTTP_{r.status_code}"
        rows=[]
        for line in r.text.splitlines():
            if not line.strip():continue
            try:x=json.loads(line)
            except ValueError:continue
            if str(x.get("url","")).rstrip("/")!=ARTICLE.rstrip("/"):continue
            if str(x.get("status",""))!="200":continue
            rows.append(x)
        return rows,("MATCHES" if rows else "NO_EXACT_INDEX_MATCH")
    except requests.RequestException as exc:
        return [],"NETWORK_"+type(exc).__name__

def fetch_warc_record(entry:dict):
    required=("filename","offset","length","digest","timestamp")
    if any(not str(entry.get(k,"")).strip() for k in required):
        raise RuntimeError("Common Crawl index entry missing WARC locator/digest")
    offset=int(entry["offset"]); length=int(entry["length"])
    if length<=0 or length>5_000_000:raise RuntimeError("unexpected WARC record length")
    url=DATA_ROOT+str(entry["filename"])
    r=requests.get(url,headers={**HEADERS,"Range":f"bytes={offset}-{offset+length-1}"},timeout=30)
    if r.status_code not in {200,206}:raise RuntimeError(f"WARC_RANGE_HTTP_{r.status_code}")
    raw=r.content
    records=list(ArchiveIterator(io.BytesIO(raw)))
    responses=[rec for rec in records if rec.rec_type=="response"]
    if len(responses)!=1:raise RuntimeError(f"expected one WARC response got {len(responses)}")
    rec=responses[0]
    target=str(rec.rec_headers.get_header("WARC-Target-URI") or "")
    if target.rstrip("/")!=ARTICLE.rstrip("/"):raise RuntimeError("WARC target URL mismatch")
    warc_date=str(rec.rec_headers.get_header("WARC-Date") or "")
    payload=rec.content_stream().read()
    if len(payload)>3_500_000:raise RuntimeError("archived payload too large")
    return rec,target,warc_date,payload

def execute(out_dir:Path)->dict:
    result={
      "contract":"WR_CB_2022W5_COMMONCRAWL_EXACT_SOURCE_V1",
      "source_url":ARTICLE,"collection":INDEX,"status":"NOT_ATTEMPTED",
      "target_game_outcomes":False,"sportsbook_inputs":False,
      "editorial_grade_used":False,"parameters_fit":0,"provider_bridge_used":False,
      "source_model_gate_cleared":False,"raw_archive_body_saved":False,
      "strict_source_rows":[],
    }
    entries,status=query_exact()
    result["index_status"]=status
    result["index_match_count"]=len(entries)
    if not entries:
        result["status"]="NO_EXACT_COMMONCRAWL_CAPTURE"
    else:
        schedule=_load_schedule([SEASON])
        kicks=pd.to_datetime(schedule.loc[schedule.week.eq(WEEK),"kickoff_utc"],
                             utc=True,errors="coerce").dropna().sort_values()
        if kicks.empty:raise RuntimeError("Week 5 schedule missing")
        first=kicks.iloc[0]
        result["first_week_kickoff_utc"]=first.isoformat()
        candidates=[]
        for e in entries:
            stamp=pd.to_datetime(str(e.get("timestamp","")),format="%Y%m%d%H%M%S",
                                 utc=True,errors="coerce")
            candidates.append((stamp,e))
        pre=[(t,e) for t,e in candidates if not pd.isna(t) and t<first]
        result["pregame_index_match_count"]=len(pre)
        result["index_timestamps"]=[None if pd.isna(t) else t.isoformat() for t,_ in candidates]
        if not pre:
            result["status"]="COMMONCRAWL_CAPTURES_NOT_PREGAME"
        else:
            # Latest strictly pregame capture is maximally informative.
            capture,entry=sorted(pre,key=lambda x:x[0])[-1]
            result["selected_capture_utc"]=capture.isoformat()
            result["index_digest"]=str(entry.get("digest",""))
            rec,target,warc_date,payload=fetch_warc_record(entry)
            result["warc_date"]=warc_date
            result["payload_sha256"]=hashlib.sha256(payload).hexdigest()
            result["payload_sha1_b32"]=_sha1_b32(payload)
            warc_payload_digest=str(rec.rec_headers.get_header("WARC-Payload-Digest") or "")
            result["warc_payload_digest"]=warc_payload_digest
            index_digest=str(entry.get("digest",""))
            index_b32=index_digest.split(":",1)[-1].upper()
            warc_b32=warc_payload_digest.split(":",1)[-1].upper()
            # Require BOTH index and WARC payload digest to agree with bytes.
            if not index_b32 or not warc_b32:
                result["status"]="MISSING_COMMONCRAWL_DIGEST_FAIL_CLOSED"
            elif index_b32!=warc_b32 or index_b32!=result["payload_sha1_b32"]:
                result["status"]="COMMONCRAWL_PAYLOAD_DIGEST_MISMATCH"
            else:
                result["archived_payload_digest_verified"]=True
                html=payload.decode("utf-8",errors="replace")
                rows,page=parse_page(html,season=SEASON,week=WEEK,source_url=ARTICLE)
                result["source_publication_utc"]=page.get("published_at_utc","")
                result["parsed_factual_pairing_rows"]=len(rows)
                if rows.empty:
                    result["status"]="DIGEST_MATCH_NO_PARSEABLE_PAIRINGS"
                else:
                    sched=schedule.rename(columns={"team":"wr_team"})
                    merged=rows.merge(sched,on=["season","week","wr_team"],how="left",validate="many_to_one")
                    # Capture itself precedes first kickoff, but still apply exact row clock.
                    good=merged.loc[
                      (capture<pd.to_datetime(merged["kickoff_utc"],utc=True,errors="coerce"))
                      & merged["opponent"].astype(str).eq(merged["scheduled_opponent"].astype(str))
                      & merged["alignment_bucket"].isin(["LWR_VS_RCB","RWR_VS_LCB","SWR_VS_SCB"])
                    ].copy()
                    result["verified_pregame_factual_rows"]=len(good)
                    rosters=_load_rosters([SEASON])
                    wr_l=exact_week_lookup(rosters,WR_POSITIONS)
                    cb_l=exact_week_lookup(rosters,DEF_POSITIONS)
                    good["wr_key"]=good["wr_raw"].map(name_key)
                    good["cb_key"]=good["cb_raw"].map(name_key)
                    def resolve(row,lookup,team_col,key_col):
                        return lookup.get((SEASON,WEEK,str(row[team_col]),str(row[key_col])),
                                          ("","NOT_FOUND"))
                    wr=good.apply(lambda r:resolve(r,wr_l,"wr_team","wr_key"),axis=1)
                    cb=good.apply(lambda r:resolve(r,cb_l,"opponent","cb_key"),axis=1)
                    good["wr_gsis_id"]=[x[0] for x in wr]; good["wr_roster_status"]=[x[1] for x in wr]
                    good["cb_gsis_id"]=[x[0] for x in cb]; good["cb_roster_status"]=[x[1] for x in cb]
                    ready=good.loc[
                      good.wr_roster_status.eq("WEEK_EXACT")
                      & good.cb_roster_status.eq("WEEK_EXACT")
                    ].copy()
                    result["wr_week_exact_rows"]=int(good.wr_roster_status.eq("WEEK_EXACT").sum())
                    result["cb_opponent_week_exact_rows"]=int(good.cb_roster_status.eq("WEEK_EXACT").sum())
                    result["strict_exact_week_identity_rows"]=len(ready)
                    result["quarantined_rows"]=len(good)-len(ready)
                    result["strict_source_rows"]=ready[[
                      "wr_team","wr_gsis_id","opponent","cb_gsis_id","alignment_bucket"
                    ]].drop_duplicates().to_dict("records")
                    result["status"]=(
                      "COMMONCRAWL_DIGEST_MATCH_WITH_STRICT_SOURCE_ROWS"
                      if len(ready) else "COMMONCRAWL_DIGEST_MATCH_NO_STRICT_IDENTITY_ROWS"
                    )
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"commoncrawl_2022w5_source_verification.json").write_text(
      json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="strict_source_rows"},
                     indent=2,sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();r=execute(a.out_dir)
    raise SystemExit(0 if r["status"]=="COMMONCRAWL_DIGEST_MATCH_WITH_STRICT_SOURCE_ROWS" else 2)
