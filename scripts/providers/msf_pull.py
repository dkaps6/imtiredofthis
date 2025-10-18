#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MySportsFeeds provider (2025).
Secrets:
  MSF_KEY (or MSF_USER) and MSF_PASSWORD, or MSF_TOKEN
Writes:
  data/msf_team_form.csv
  data/msf_player_form.csv
Unknown metrics remain NaN; CSVs are always written to satisfy fallback sweep.
"""

import argparse
import base64
import os
import sys
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import requests

DATA_DIR = "data"

TEAM_COLS = [
    "team","def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
    "light_box_rate","heavy_box_rate"
]
PLAYER_COLS = [
    "player","team",
    "tgt_share","route_rate","rush_share",
    "yprr","ypt","ypc","ypa",
    "receptions_per_target",
    "rz_share","rz_tgt_share","rz_rush_share",
]

def _safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def _auth_header() -> Dict[str, str]:
    # support either token or basic (key/user + password)
    token = os.environ.get("MSF_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user = os.environ.get("MSF_KEY") or os.environ.get("MSF_USER") or ""
    pwd  = os.environ.get("MSF_PASSWORD", "")
    if user and pwd:
        basic = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {basic}"}
    return {}

def _get_json(url: str, params: Dict[str, Any], headers: Dict[str,str]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[msf] GET {url} failed: {e}", file=sys.stderr)
        return {}

def _write_csv(path: str, df: pd.DataFrame, cols: List[str]):
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out.columns = [c.lower() for c in out.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(path, index=False)
    return out

def fetch_team_2025(headers: Dict[str,str]) -> pd.DataFrame:
    # Example endpoint (adjust to your plan)
    url = "https://api.mysportsfeeds.com/v2.1/pull/nfl/2025-2026-regular/team_stats_totals.json"
    js = _get_json(url, {}, headers)
    rows: List[Dict[str, Any]] = []
    for item in js.get("teams", []):
        abbr = item.get("team", {}).get("abbreviation") or item.get("abbr") or item.get("name")
        stats = item.get("stats", {})
        rows.append({
            "team": str(abbr).upper() if abbr else None,
            # map any available fields into our schema; rest remain NaN
            "def_sack_rate": stats.get("defense", {}).get("sacksPerPassAttempt"),
            # leave other fields NaN if not present
        })
    return pd.DataFrame(rows)

def fetch_player_2025(headers: Dict[str,str]) -> pd.DataFrame:
    # Example endpoint (adjust to your plan)
    url = "https://api.mysportsfeeds.com/v2.1/pull/nfl/2025-2026-regular/player_stats_totals.json"
    js = _get_json(url, {}, headers)
    rows: List[Dict[str, Any]] = []
    for item in js.get("players", []):
        ply = item.get("player", {})
        team = ply.get("currentTeam", {}).get("abbreviation") or ply.get("team") or ""
        stats = item.get("stats", {})
        passing = stats.get("passing", {})
        rushing = stats.get("rushing", {})
        rows.append({
            "player": str(ply.get("firstName","") + " " + ply.get("lastName","")).strip().replace(".",""),
            "team": str(team).upper(),
            "ypa": passing.get("yardsPerAttempt"),
            "ypc": rushing.get("yardsPerAttempt"),
            # tgt_share/route_rate etc not in MSF totals — leave NaN
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    if args.season != 2025:
        print("[msf] 2025-only; skipping", file=sys.stderr)
        return 0

    _safe_mkdir(DATA_DIR)
    headers = _auth_header()
    if not headers:
        print("[msf] no credentials; writing schema-only files", file=sys.stderr)
        _write_csv(os.path.join(DATA_DIR,"msf_team_form.csv"), pd.DataFrame(), TEAM_COLS)
        _write_csv(os.path.join(DATA_DIR,"msf_player_form.csv"), pd.DataFrame(), PLAYER_COLS)
        return 0

    team_df = fetch_team_2025(headers)
    player_df = fetch_player_2025(headers)

    t_out = _write_csv(os.path.join(DATA_DIR,"msf_team_form.csv"), team_df, TEAM_COLS)
    p_out = _write_csv(os.path.join(DATA_DIR,"msf_player_form.csv"), player_df, PLAYER_COLS)
    print(f"[msf] wrote team rows={len(t_out)} → data/msf_team_form.csv")
    print(f"[msf] wrote player rows={len(p_out)} → data/msf_player_form.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
