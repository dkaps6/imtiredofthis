#!/usr/bin/env python3
"""Build a timestamped Ourlads depth/status sidecar without changing legacy roles.

V1 implementation branch only. This reuses the production parser but preserves
its raw availability signal and source provenance for reconciliation.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from scripts.providers import ourlads_depth as od

OUT=Path("data/roles_ourlads_status_v1.csv")
STATUS=Path("data/roles_ourlads_status_v1.json")

def build() -> tuple[pd.DataFrame,dict]:
    now=datetime.now(timezone.utc).isoformat(); rows=[]; failures=[]
    for tm in sorted(od.TEAM_URLS):
        try:
            soup=od._get_depth_soup(tm)
            rr=od.fetch_team_roles(tm,soup,include_inactive=True)
            for r in rr:
                q=dict(r); q["team"]=od.normalize_team(q.get("team")); q["source"]="ourlads_depth"; q["source_url"]=od.TEAM_URLS[tm]; q["source_asof_utc"]=now; rows.append(q)
        except Exception as exc:
            failures.append({"team":tm,"error":f"{type(exc).__name__}:{exc}"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df["player"]=df["player"].map(od.clean_ourlads_name)
        df["player_clean_key"]=df["player"].map(lambda v: od.make_keys(v)[2])
        df=df[df.team.isin(od.VALID)].copy()
        df=df.drop_duplicates(["team","player_clean_key","role"],keep="first").sort_values(["team","position_group","depth_index","role","player"])
    meta={"generated_at_utc":now,"source":"ourlads_depth","teams_expected":32,"teams_with_rows":int(df.team.nunique()) if len(df) else 0,"rows":int(len(df)),"inactive_rows":int(df.status.astype(str).str.lower().eq("inactive").sum()) if "status" in df else 0,"team_failures":failures,"complete":bool(len(df) and df.team.nunique()==32 and not failures)}
    return df.reset_index(drop=True),meta

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--status",type=Path,default=STATUS); a=ap.parse_args()
    df,meta=build(); a.out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False); a.status.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(meta,indent=2,sort_keys=True))
    if not meta["complete"]: raise RuntimeError("Ourlads depth/status source incomplete")
    return 0
if __name__=="__main__": raise SystemExit(main())
