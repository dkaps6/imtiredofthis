#!/usr/bin/env python3
"""Acquire current official NFL game-day inactive sections using the hardened M78 contract.

This is an operational adapter over already-audited M78 parsing semantics. Endpoint
reachability alone never certifies a team; only complete parseable team sections do.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import audit_qb_official_inactive_availability_v3 as hard

OUT=Path("data/official_inactives_v1.csv")
STATUS=Path("data/official_inactives_v1_status.json")

def build(url:str=m78.LIVE_INACTIVES_URL)->tuple[pd.DataFrame,dict]:
    now=datetime.now(timezone.utc).isoformat(); rows=[]; section_rows=[]
    try:
        r=m78.request(url)
        soup=m78.BeautifulSoup(r.text,"html.parser")
        sections=hard._team_section_candidates(soup)
        for team,_label,ul in sections:
            bullets=hard._candidate_bullets(ul); parsed=[m78.parse_player_bullet(x) for x in bullets]; ok=[x for x in parsed if x is not None]
            complete=bool(len(bullets)>=3 and len(ok)==len(bullets) and len({m78.norm_name(name) for _,name in ok})==len(ok))
            section_rows.append({"team":team,"section_complete":int(complete),"candidate_bullets":len(bullets),"parsed_bullets":len(ok)})
            # Always emit a section ledger row so complete absence checks are explicit.
            rows.append({"team":team,"player":"","listed_position":"","section_complete":int(complete),"source_url":r.url,"source_asof_utc":now})
            for pos,name in ok:
                rows.append({"team":team,"player":name,"listed_position":pos,"section_complete":int(complete),"source_url":r.url,"source_asof_utc":now})
        frame=pd.DataFrame(rows,columns=["team","player","listed_position","section_complete","source_url","source_asof_utc"])
        complete_teams=[x["team"] for x in section_rows if x["section_complete"]==1]
        status={"generated_at_utc":now,"source":"nfl_official_inactives","url":r.url,"http_status":int(r.status_code),"endpoint_reachable":True,"complete_team_sections":len(complete_teams),"complete_teams":sorted(complete_teams),"listed_players":int(frame.player.astype(str).str.strip().ne("").sum()) if len(frame) else 0,"sections":section_rows,"payload_valid":bool(complete_teams)}
        return frame,status
    except Exception as exc:
        frame=pd.DataFrame(columns=["team","player","listed_position","section_complete","source_url","source_asof_utc"])
        status={"generated_at_utc":now,"source":"nfl_official_inactives","url":url,"endpoint_reachable":False,"complete_team_sections":0,"complete_teams":[],"listed_players":0,"sections":[],"payload_valid":False,"error":f"{type(exc).__name__}:{exc}"}
        return frame,status

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--status",type=Path,default=STATUS); a=ap.parse_args()
    df,status=build(); a.out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False); a.status.write_text(json.dumps(status,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(status,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
