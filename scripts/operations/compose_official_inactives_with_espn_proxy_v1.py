#!/usr/bin/env python3
"""Compose true NFL game-day inactives with ESPN injury-OUT proxy facts.

The T-75 gate may only be satisfied by true NFL official-inactive sections.
ESPN's core injury endpoint remains useful as a supplemental OUT signal but
must never impersonate an official inactive section.

This adapter preserves both semantics in one downstream compatibility ledger:
  data/official_inactives_v1.csv

No sportsbook data is read.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

DEFAULT_NFL=Path("data/nfl_official_inactives_v1.csv")
DEFAULT_NFL_STATUS=Path("data/nfl_official_inactives_v1_status.json")
DEFAULT_ESPN=Path("data/espn_injury_out_proxy_v1.csv")
DEFAULT_ESPN_STATUS=Path("data/espn_injury_out_proxy_v1_status.json")
DEFAULT_OUT=Path("data/official_inactives_v1.csv")
DEFAULT_STATUS=Path("data/official_inactives_v1_status.json")


def _read_csv(path:Path)->pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.stat().st_size<=0:
        return pd.DataFrame()
    return pd.read_csv(path,low_memory=False)


def _read_json(path:Path)->dict:
    if not path.exists() or path.stat().st_size<=0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compose(nfl:pd.DataFrame, espn:pd.DataFrame, *, nfl_status:dict, espn_status:dict)->tuple[pd.DataFrame,dict]:
    n=nfl.copy()
    e=espn.copy()

    if not n.empty:
        n.columns=[str(c).lower() for c in n.columns]
        missing={"team","player","section_complete"}-set(n.columns)
        if missing:
            raise RuntimeError(f"NFL official inactive ledger missing columns: {sorted(missing)}")
        n["team"]=n["team"].map(canon_team)
        if n["team"].astype(str).eq("").any():
            raise RuntimeError("NFL official inactive ledger contains unresolved team")
        if "source_semantics" not in n.columns:
            n["source_semantics"]="OFFICIAL_GAME_DAY_INACTIVE"
        else:
            sem=n["source_semantics"].astype("string").fillna("").str.upper().str.strip()
            if sem.eq("INJURY_STATUS_OUT").any():
                raise RuntimeError("NFL official inactive ledger contains ESPN proxy semantics")
            n.loc[sem.eq(""),"source_semantics"]="OFFICIAL_GAME_DAY_INACTIVE"
        if "provider_status" not in n.columns:
            n["provider_status"]=""
        if "source_fetch_complete" not in n.columns:
            n["source_fetch_complete"]=pd.to_numeric(n["section_complete"],errors="coerce").fillna(0).astype(int)

    if not e.empty:
        e.columns=[str(c).lower() for c in e.columns]
        missing={"team","player","section_complete","source_semantics"}-set(e.columns)
        if missing:
            raise RuntimeError(f"ESPN proxy ledger missing columns: {sorted(missing)}")
        e["team"]=e["team"].map(canon_team)
        if e["team"].astype(str).eq("").any():
            raise RuntimeError("ESPN injury proxy contains unresolved team")
        sem=e["source_semantics"].astype("string").fillna("").str.upper().str.strip()
        if not sem.eq("INJURY_STATUS_OUT").all():
            raise RuntimeError("ESPN proxy ledger contains non-proxy semantics")
        # Defense in depth: a proxy can never certify an official section.
        e["section_complete"]=0

    frames=[x for x in (n,e) if not x.empty]
    if frames:
        out=pd.concat(frames,ignore_index=True,sort=False)
    else:
        out=pd.DataFrame(columns=[
            "team","player","listed_position","section_complete",
            "source_semantics","provider_status","source_fetch_complete",
            "source_url","source_asof_utc",
        ])

    complete=set()
    if not n.empty:
        good=pd.to_numeric(n["section_complete"],errors="coerce").fillna(0).eq(1)
        complete=set(n.loc[good,"team"].astype(str))

    nfl_players=0 if n.empty else int(n["player"].astype("string").fillna("").str.strip().ne("").sum())
    espn_players=0 if e.empty else int(e["player"].astype("string").fillna("").str.strip().ne("").sum())
    status={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "source":"nfl_official_inactives_plus_espn_injury_out_proxy",
        "source_semantics":"COMPOSITE_OFFICIAL_PLUS_INJURY_PROXY",
        "official_inactive_source":True,
        "complete_team_sections":len(complete),
        "complete_teams":sorted(complete),
        "official_listed_players":nfl_players,
        "espn_proxy_listed_out_players":espn_players,
        "nfl_status":nfl_status,
        "espn_status":espn_status,
        "sportsbook_inputs_used":0,
        "payload_valid":bool(complete) or bool(espn_status.get("payload_valid")),
    }
    return out,status


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--nfl",type=Path,default=DEFAULT_NFL)
    ap.add_argument("--nfl-status",type=Path,default=DEFAULT_NFL_STATUS)
    ap.add_argument("--espn",type=Path,default=DEFAULT_ESPN)
    ap.add_argument("--espn-status",type=Path,default=DEFAULT_ESPN_STATUS)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--status",type=Path,default=DEFAULT_STATUS)
    a=ap.parse_args()

    out,status=compose(
        _read_csv(a.nfl),
        _read_csv(a.espn),
        nfl_status=_read_json(a.nfl_status),
        espn_status=_read_json(a.espn_status),
    )
    a.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.out,index=False)
    a.status.write_text(json.dumps(status,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":"OFFICIAL_INACTIVES_COMPOSITE_READY",
        "rows":int(len(out)),
        "complete_teams":status["complete_teams"],
        "official_listed_players":status["official_listed_players"],
        "espn_proxy_listed_out_players":status["espn_proxy_listed_out_players"],
        "sportsbook_inputs_used":0,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
