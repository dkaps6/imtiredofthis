#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APISports provider (2025-only).
- Reads APISPORTS_KEY from env.
- Fetches team + player stats if available (endpoints can vary by plan/product).
- Always writes:
    data/apisports_team_form.csv
    data/apisports_player_form.csv
  using the exact schemas your builders expect.

If the API is unavailable, we still write the CSVs with the required columns and NaNs.
"""

import argparse
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

def _write_csv_with_schema(path: str, df: pd.DataFrame, required_cols: List[str]):
    out = (df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame())
    out.columns = [c.lower() for c in out.columns]
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[required_cols]
    out.to_csv(path, index=False)
    return out

def _get_json(url: str, params: Dict[str, Any], headers: Dict[str,str]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[apisports] GET {url} failed: {e}", file=sys.stderr)
        return {}

def _norm_team(s: Any) -> str:
    return str(s).upper().strip() if s is not None else ""


def fetch_team_stats_2025(api_key: str) -> pd.DataFrame:
    """
    Adjust the base URL to the exact APISports product you use.
    This function maps only what we can safely read; unknowns remain NaN.
    """
    # EXAMPLE endpoint. Replace with your actual product path.
    base_url = "https://v1.apisports.io/nfl/teams/statistics"
    params = {"season": 2025}
    headers = {"x-apisports-key": api_key}

    js = _get_json(base_url, params, headers)
    rows: List[Dict[str, Any]] = []
    for item in js.get("response", []):
        team = item.get("team", {}) or {}
        stats = item.get("statistics", {}) or {}
        rows.append({
            "team": _norm_team(team.get("code") or team.get("abbr") or team.get("name")),
            "def_pass_epa": stats.get("def_pass_epa"),
            "def_rush_epa": stats.get("def_rush_epa"),
            "def_sack_rate": stats.get("def_sack_rate"),
            "pace": stats.get("pace_neutral") or stats.get("pace"),
            "proe": stats.get("proe"),
            "rz_rate": stats.get("rz_rate"),
            "12p_rate": stats.get("personnel_12_rate") or stats.get("12p_rate"),
            "slot_rate": stats.get("slot_rate"),
            "ay_per_att": stats.get("ay_per_att"),
            "light_box_rate": stats.get("light_box_rate"),
            "heavy_box_rate": stats.get("heavy_box_rate"),
        })
    return pd.DataFrame(rows)


def fetch_player_stats_2025(api_key: str) -> pd.DataFrame:
    """
    If your APISports plan exposes per-player season stats, map them here.
    Otherwise we’ll simply return an empty frame and the writer will emit NaN columns.
    """
    # EXAMPLE: placeholder — replace with real endpoint(s) you have.
    # base_url = "https://v1.apisports.io/nfl/players/statistics"
    # params = {"season": 2025}
    # headers = {"x-apisports-key": api_key}
    # js = _get_json(base_url, params, headers)
    rows: List[Dict[str, Any]] = []
    # for entry in js.get("response", []):
    #     player = entry.get("player", {})
    #     team = entry.get("team", {})
    #     stats = entry.get("statistics", {}) or {}
    #     rows.append({
    #         "player": str(player.get("name") or player.get("fullname") or "").strip().replace(".", ""),
    #         "team": _norm_team(team.get("code") or team.get("abbr") or team.get("name")),
    #         "tgt_share": stats.get("target_share"),
    #         "route_rate": stats.get("route_rate"),
    #         "rush_share": stats.get("rush_share"),
    #         "yprr": stats.get("yprr"),
    #         "ypt": stats.get("yards_per_target"),
    #         "ypc": stats.get("yards_per_rush"),
    #         "ypa": stats.get("yards_per_attempt"),
    #         "receptions_per_target": stats.get("receptions_per_target"),
    #         "rz_share": stats.get("rz_share"),
    #         "rz_tgt_share": stats.get("rz_tgt_share"),
    #         "rz_rush_share": stats.get("rz_rush_share"),
    #     })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    if args.season != 2025:
        print("[apisports] 2025-only; skipping", file=sys.stderr)
        return 0

    _safe_mkdir(DATA_DIR)
    key = os.environ.get("APISPORTS_KEY") or os.environ.get("API_SPORTS_KEY") or ""
    if not key:
        print("[apisports] no APISPORTS_KEY; writing schema-only files", file=sys.stderr)
        _write_csv_with_schema(os.path.join(DATA_DIR,"apisports_team_form.csv"), pd.DataFrame(), TEAM_COLS)
        _write_csv_with_schema(os.path.join(DATA_DIR,"apisports_player_form.csv"), pd.DataFrame(), PLAYER_COLS)
        return 0

    try:
        team_df = fetch_team_stats_2025(key)
    except Exception as e:
        print(f"[apisports] team fetch failed: {e}", file=sys.stderr)
        team_df = pd.DataFrame()

    try:
        player_df = fetch_player_stats_2025(key)
    except Exception as e:
        print(f"[apisports] player fetch failed: {e}", file=sys.stderr)
        player_df = pd.DataFrame()

    team_out = _write_csv_with_schema(os.path.join(DATA_DIR,"apisports_team_form.csv"), team_df, TEAM_COLS)
    player_out = _write_csv_with_schema(os.path.join(DATA_DIR,"apisports_player_form.csv"), player_df, PLAYER_COLS)

    print(f"[apisports] wrote team rows={len(team_out)} → data/apisports_team_form.csv")
    print(f"[apisports] wrote player rows={len(player_out)} → data/apisports_player_form.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
