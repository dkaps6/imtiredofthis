#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MySportsFeeds placeholder fetcher.
- Uses MSF key (BASIC or token) if present.
- Writes a normalized CSV only for season=2025.
Graceful no-op if credentials or endpoint are unavailable in CI.
"""

import argparse
import base64
import os
import sys
from typing import Dict, Any, List

import pandas as pd
import requests


OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "msf_teams_2025.csv")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _auth_header() -> Dict[str, str]:
    token = os.environ.get("MSF_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}

    user = os.environ.get("MSF_USER", "")
    pwd = os.environ.get("MSF_PASSWORD", "")
    if user and pwd:
        basic = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {basic}"}
    return {}


def _api_get(url: str, params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[msf_pull] GET failed: {e}", file=sys.stderr)
        return {}


def _normalize_rows(payload: Dict[str, Any], season: int) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()

    # Guess shape: {"teams": [ {"abbr": "...", "stats": {...}}, ... ]}
    rows: List[Dict[str, Any]] = []
    for item in payload.get("teams", []):
        abbr = item.get("abbr") or item.get("teamAbbr") or item.get("name")
        stats = item.get("stats", {})
        rec = {
            "team": str(abbr).upper() if abbr else None,
            "season": season,
            "def_pressure_rate": stats.get("def_pressure_rate"),
            "def_sack_rate": stats.get("def_sack_rate"),
            "def_pass_epa": stats.get("def_pass_epa"),
            "def_rush_epa": stats.get("def_rush_epa"),
            "pace_neutral": stats.get("pace_neutral"),
            "light_box_rate": stats.get("light_box_rate"),
            "heavy_box_rate": stats.get("heavy_box_rate"),
            "ay_per_att": stats.get("ay_per_att"),
            "pass_rate_neutral": stats.get("pass_rate_neutral"),
        }
        rows.append(rec)

    df = pd.DataFrame(rows)
    if "season" in df.columns:
        df = df[df["season"].astype(int) == 2025].copy()
    else:
        df["season"] = 2025

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, help="Season year (use 2025)")
    args = parser.parse_args()

    if args.season != 2025:
        print("[msf_pull] This job is for 2025-only; skipping non-2025 call.")
        return

    _ensure_dir(OUT_DIR)

    headers = _auth_header()
    if not headers:
        print("[msf_pull] No MSF credentials; skipping.", file=sys.stderr)
        return

    # Update to your exact MSF endpoint/params
    url = "https://api.mysportsfeeds.com/v2.1/pull/nfl/2025-2026-regular/team_stats.json"
    params: Dict[str, Any] = {}

    payload = _api_get(url, params, headers)
    df = _normalize_rows(payload, season=2025)

    if not len(df):
        print("[msf_pull] no rows; nothing written")
        return

    df.to_csv(OUT_FILE, index=False)
    print(f"[msf_pull] wrote {OUT_FILE} rows={len(df)}")


if __name__ == "__main__":
    main()
