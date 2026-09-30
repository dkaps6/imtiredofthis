#!/usr/bin/env python3
"""Verify one protected 2025-W14 WR/CB Wayback snapshot as SOURCE ONLY.

No 2025 outcomes, sportsbook data, model fitting, editorial grade use, current
article reacquisition, provider-ID bridge, or fuzzy/manual identity rescue.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd, requests
from bs4 import BeautifulSoup

from scripts.research.acquire_fantasyalarm_wr_cb_archive_v1 import parse_page
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import (
    DEF_POSITIONS, WR_POSITIONS, _load_rosters, _load_schedule,
)
from scripts.research.audit_verified_wr_cb_snapshot_rows_2024w1_v1 import exact_week_lookup
from scripts.research.verify_wr_cb_wayback_body_2024w1_v1 import (
    archived_article_body_inventory, _synthetic_page_from_archived_body,
)

ARTICLE=("https://www.fantasyalarm.com/articles/nfl/wide-receivers/"
         "2025-fantasy-football-wr-cb-matchup-report-week-14/183858")
TS="20251206131409"
ARCHIVED_UTC=datetime.strptime(TS,"%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
REPLAY=f"https://web.archive.org/web/{TS}id_/{ARTICLE}"
CDX="https://web.archive.org/cdx/search/cdx"
HEADERS={"User-Agent":"NFLSourceProvenanceAudit/1.0 exact-one-protected-source"}

def sha1_b32(data:bytes)->str:
    return base64.b32encode(hashlib.sha1(data).digest()).decode("ascii").rstrip("=")

def cdx_exact_digest()->tuple[str,str]:
    params={"url":ARTICLE,"output":"json","fl":"timestamp,original,statuscode,digest",
            "filter":"statuscode:200","from":TS,"to":TS,"limit":"5"}
    last=""
    for attempt in range(3):
        try:
            r=requests.get(CDX,params=params,headers=HEADERS,timeout=12)
            if r.status_code!=200:
                last=f"HTTP_{r.status_code}"
            else:
                try: data=r.json()
                except ValueError:
                    last="NOT_JSON"
                else:
                    if isinstance(data,list) and len(data)>=2 and isinstance(data[0],list):
                        head=data[0]
                        for row in data[1:]:
                            if not isinstance(row,list) or len(row)!=len(head):continue
                            x=dict(zip(head,row))
                            if str(x.get("timestamp",""))!=TS:continue
                            if str(x.get("statuscode",""))!="200":continue
                            if str(x.get("original","")).rstrip("/")!=ARTICLE.rstrip("/"):continue
                            digest=str(x.get("digest","")).strip()
                            if digest:return digest,"EXACT_CDX_DIGEST"
                    last="NO_EXACT_CDX_RECORD"
        except requests.RequestException as exc:
            last="NETWORK_"+type(exc).__name__
        if attempt<2:time.sleep(1.2)
    return "",last or "CDX_UNKNOWN"

def name_key(v)->str:
    from scripts.utils.canonical_names import canonicalize_player_name_safe
    return str(canonicalize_player_name_safe(v)[1] or "").strip()

def execute(out_dir:Path)->dict:
    result={
      "contract":"WR_CB_2025W14_PROTECTED_EXACT_ARCHIVE_SOURCE_V1",
      "season":2025,"week":14,"source_url":ARTICLE,
      "archive_index_timestamp_utc":ARCHIVED_UTC.isoformat(),
      "exact_replay_url":REPLAY,
      "status":"NOT_ATTEMPTED","cdx_digest":"","cdx_status":"",
      "archived_body_digest_verified":False,"parsed_factual_pairing_rows":0,
      "verified_pregame_factual_rows":0,"strict_exact_week_identity_rows":0,
      "raw_body_saved":False,"provider_bridge_used":False,
      "editorial_grade_used":False,"sportsbook_inputs":False,
      "target_game_outcomes":False,"confirmation_outcomes_accessed":False,
      "parameters_fit":0,"source_model_gate_cleared":False,
      "protected_source_rows":[],
    }
    digest,cdx_status=cdx_exact_digest()
    result["cdx_digest"]=digest;result["cdx_status"]=cdx_status
    try:
        response=requests.get(REPLAY,headers=HEADERS,timeout=22,allow_redirects=True)
        result["http_status"]=response.status_code
        result["final_replay_url"]=response.url
        if response.status_code!=200:
            result["status"]=f"BODY_LOOKUP_HTTP_{response.status_code}"
        elif urlparse(response.url).hostname!="web.archive.org" or f"/web/{TS}" not in urlparse(response.url).path:
            result["status"]="REPLAY_REDIRECTED_AWAY_FROM_EXACT_TIMESTAMP"
        elif len(response.content)>3500000:
            result["status"]="BODY_TOO_LARGE_FAIL_CLOSED"
        else:
            raw=response.content
            result["body_sha256"]=hashlib.sha256(raw).hexdigest()
            result["body_sha1_b32"]=sha1_b32(raw)
            if not digest:
                result["status"]="BODY_PRESENT_CDX_DIGEST_UNAVAILABLE_FAIL_CLOSED"
            elif result["body_sha1_b32"]!=digest:
                result["status"]="REPLAY_BODY_DIGEST_MISMATCH_UNVERIFIED"
            else:
                result["archived_body_digest_verified"]=True
                soup=BeautifulSoup(raw,"html.parser")
                bodies,inventory=archived_article_body_inventory(soup)
                result["archived_article_body_structure"]={
                  "candidate_count":len(bodies),"candidates":inventory,
                  "raw_article_body_saved":False,
                }
                full=raw.decode(response.encoding or "utf-8",errors="replace")
                rows,page=parse_page(full,season=2025,week=14,source_url=ARTICLE)
                source_publish=page["published_at_utc"]
                frames=[]
                for body in bodies:
                    candidate,_=parse_page(
                      _synthetic_page_from_archived_body(body,source_publish),
                      season=2025,week=14,source_url=ARTICLE)
                    if not candidate.empty:frames.append(candidate)
                embedded=(pd.concat(frames,ignore_index=True).drop_duplicates(
                  ["season","week","alignment_bucket","wr_clean_key","cb_clean_key"],keep="last")
                  if frames else pd.DataFrame())
                if rows.empty and not embedded.empty:
                    rows=embedded;result["pairing_parse_source"]="ARCHIVED_JSON_LD_ARTICLEBODY"
                elif not rows.empty:
                    result["pairing_parse_source"]="ARCHIVED_OUTER_HTML"
                else:
                    result["pairing_parse_source"]="NONE"
                result["parsed_factual_pairing_rows"]=int(len(rows))
                result["source_publication_utc"]=source_publish
                if rows.empty:
                    result["status"]="HASH_MATCH_NO_EXPLICIT_PAIRINGS"
                else:
                    sched=_load_schedule([2025]).rename(columns={"team":"wr_team"})
                    merged=rows.merge(sched,on=["season","week","wr_team"],how="left",validate="many_to_one")
                    pub=pd.to_datetime(source_publish,utc=True,errors="coerce")
                    good=merged.loc[
                      pub.notna()
                      & (pub<=pd.Timestamp(ARCHIVED_UTC))
                      & (pd.Timestamp(ARCHIVED_UTC)<pd.to_datetime(merged["kickoff_utc"],utc=True,errors="coerce"))
                      & merged["opponent"].astype(str).eq(merged["scheduled_opponent"].astype(str))
                      & merged["alignment_bucket"].isin(["LWR_VS_RCB","RWR_VS_LCB","SWR_VS_SCB"])
                    ].copy()
                    result["verified_pregame_factual_rows"]=int(len(good))
                    rosters=_load_rosters([2025])
                    wr_l=exact_week_lookup(rosters,WR_POSITIONS);cb_l=exact_week_lookup(rosters,DEF_POSITIONS)
                    good["wr_key"]=good["wr_raw"].map(name_key);good["cb_key"]=good["cb_raw"].map(name_key)
                    def resolve(row,lookup,team_col,key_col):
                        return lookup.get((2025,14,str(row[team_col]),str(row[key_col])),("","NOT_FOUND"))
                    wr=good.apply(lambda r:resolve(r,wr_l,"wr_team","wr_key"),axis=1)
                    cb=good.apply(lambda r:resolve(r,cb_l,"opponent","cb_key"),axis=1)
                    good["wr_gsis_id"]=[x[0] for x in wr];good["wr_roster_status"]=[x[1] for x in wr]
                    good["cb_gsis_id"]=[x[0] for x in cb];good["cb_roster_status"]=[x[1] for x in cb]
                    ready=good.loc[good.wr_roster_status.eq("WEEK_EXACT")&good.cb_roster_status.eq("WEEK_EXACT")].copy()
                    result["strict_exact_week_identity_rows"]=int(len(ready))
                    result["wr_week_exact_rows"]=int(good.wr_roster_status.eq("WEEK_EXACT").sum())
                    result["cb_opponent_week_exact_rows"]=int(good.cb_roster_status.eq("WEEK_EXACT").sum())
                    result["quarantined_rows"]=int(len(good)-len(ready))
                    result["protected_source_rows"]=ready[[
                      "wr_team","wr_gsis_id","opponent","cb_gsis_id","alignment_bucket"
                    ]].drop_duplicates().to_dict("records")
                    result["status"]=(
                      "ARCHIVE_BODY_HASH_MATCH_WITH_STRICT_PROTECTED_SOURCE_ROWS"
                      if len(ready) else "ARCHIVE_BODY_HASH_MATCH_NO_STRICT_IDENTITY_ROWS"
                    )
    except requests.RequestException as exc:
        result["status"]="BODY_LOOKUP_NETWORK_"+type(exc).__name__
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"protected_2025w14_snapshot_verification.json").write_text(
      json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="protected_source_rows"},indent=2,sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();r=execute(a.out_dir)
    raise SystemExit(0 if r["status"]=="ARCHIVE_BODY_HASH_MATCH_WITH_STRICT_PROTECTED_SOURCE_ROWS" else 2)
