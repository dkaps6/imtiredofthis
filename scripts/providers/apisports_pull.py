#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APISports (or similar) placeholder fetcher.
- Reads env keys if available.
- Writes a normalized CSV only for season=2025.
If the API is unavailable in CI, this script degrades gracefully and writes nothing (job continues).
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import pandas as pd
import requests


OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "apisports_teams_2025.csv")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _api_get(url: str, params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[apisports_pull] GET failed: {e}", file=sys.stderr)
        return {}


def _normalize_rows(payload: Dict[str, Any], season: int) -> pd.DataFrame:
    """
    Adapt this to your APISports JSON shape.
    We only keep season=2025 rows and a small set of columns that are useful to later merges.
    """
    if not payload:
        return pd.DataFrame()

    # Guess a common shape: {"response": [ {team: {...}, statistics: {...}}, ... ]}
    rows: List[Dict[str, Any]] = []
    for item in payload.get("response", []):
        team = item.get("team", {})
        stat = item.get("statistics", {})
        rec = {
            "team": team.get("code") or team.get("abbr") or team.get("name"),
            "season": season,
            # optional fields
            "def_pressure_rate": stat.get("def_pressure_rate"),
            "def_sack_rate": stat.get("def_sack_rate"),
            "def_pass_epa": stat.get("def_pass_epa"),
            "def_rush_epa": stat.get("def_rush_epa"),
            "pace_neutral": stat.get("pace_neutral"),
            "light_box_rate": stat.get("light_box_rate"),
            "heavy_box_rate": stat.get("heavy_box_rate"),
            "ay_per_att": stat.get("ay_per_att"),
            "pass_rate_neutral": stat.get("pass_rate_neutral"),
        }
        rows.append(rec)

    df = pd.DataFrame(rows)
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper()

    # Clamp season
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
        print("[apisports_pull] This job is for 2025-only; skipping non-2025 call.")
        return

    _ensure_dir(OUT_DIR)

    # Adjust URL and headers to your APISports product
    api_key = os.environ.get("APISPORTS_KEY") or os.environ.get("API_SPORTS_KEY") or ""
    if not api_key:
        print("[apisports_pull] No APISports key found in env; skipping.", file=sys.stderr)
        return

    url = "https://v1.apisports.io/nfl/teams/statistics"  # NOTE: Update to your exact endpoint
    params = {
        "season": 2025,
    }
    headers = {
        "x-apisports-key": api_key
    }

    payload = _api_get(url, params, headers)
    df = _normalize_rows(payload, season=2025)

    if not len(df):
        print("[apisports_pull] no rows; nothing written")
        return

    df.to_csv(OUT_FILE, index=False)
    print(f"[apisports_pull] wrote {OUT_FILE} rows={len(df)}")


if __name__ == "__main__":
    main()
