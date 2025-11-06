#!/usr/bin/env python3
"""
Auto-scrape FantasyPoints WR-CB matchup data weekly and normalize team codes.
Outputs: data/wr_cb_matchups_WEEK{week}.csv
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from pathlib import Path
import datetime

TEAM_WEEK_MAP = Path("data/team_week_map.csv")
URL = "https://www.fantasypoints.com/nfl/reports/wr-cb-matchups#/"

# Canonical team code remaps (your model’s canon in parentheses)
CANON_TEAM_MAP = {
    "BLT": "BAL",  # (BAL)
    "CLV": "CLE",  # (CLE)
    "HST": "HOU",  # (HOU)
    # common alternates/legacy
    "WSH": "WAS",
    "JAC": "JAX",
    "LA":  "LAR",
    "LACH":"LAC",
    "STL": "LAR",
    "SD":  "LAC",
}

def norm_abbr(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().upper()
    return CANON_TEAM_MAP.get(x, x)

def get_current_week():
    """Detect current season/week from team_week_map.csv (non-bye rows)."""
    if TEAM_WEEK_MAP.exists():
        df = pd.read_csv(TEAM_WEEK_MAP)
        if "bye" in df.columns:
            df = df[df["bye"] == False]
        season = int(df["season"].max())
        week = int(df["week"].max())
        print(f"🗓️  Detected season {season}, week {week}")
        return season, week
    # Fallback to calendar week if map missing
    today = datetime.date.today()
    return today.year, today.isocalendar().week

def scrape_wr_cb():
    season, week = get_current_week()

    print(f"🔍 Scraping WR-CB matchups for Week {week} ({season}) from FantasyPoints…")
    html = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script:
        raise RuntimeError("FantasyPoints WR/CB data not found in page source")

    data = json.loads(script.string)
    page_props = data.get("props", {}).get("pageProps", {})

    # Find the list of dicts containing 'wr' key
    table_data = None
    for v in page_props.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "wr" in v[0]:
            table_data = v
            break
    if table_data is None:
        raise RuntimeError("❌ Could not locate WR/CB table structure in FantasyPoints JSON")

    df = pd.DataFrame(table_data)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Normalize/rename expected columns
    rename_map = {
        "wr": "player",
        "wr_team": "team_abbr",
        "cb_team": "opponent_abbr",
        "adv": "matchup_adv",
        "slot_%": "slot_rate",
        "lwr_%": "left_align_rate",
        "rwr_%": "right_align_rate",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Clean and normalize
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip().str.title()
    for col in ["team_abbr", "opponent_abbr"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(norm_abbr)

    df["week"] = week
    df["season"] = season

    # Optional sanity check vs your team_form canon
    try:
        tf = pd.read_csv("data/team_form.csv")
        valid = set(tf["team_abbr"].astype(str).str.upper().unique())
        present = set(df.get("team_abbr", [])).union(set(df.get("opponent_abbr", [])))
        bad = sorted(present - valid)
        if bad:
            print(f"⚠️ Unknown team codes after mapping: {bad}")
    except Exception:
        pass

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wr_cb_matchups_WEEK{week}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved WR/CB matchups to {out_path} ({len(df)} rows)")
    return out_path

if __name__ == "__main__":
    scrape_wr_cb()
