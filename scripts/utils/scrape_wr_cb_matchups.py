#!/usr/bin/env python3
"""
Scrape WR-CB matchup data from FantasyPoints and export as CSV.
This enriches WR projections with cornerback matchup grades and coverage alignment info.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from pathlib import Path

URL = "https://www.fantasypoints.com/nfl/reports/wr-cb-matchups#/"

def scrape_wr_cb():
    html = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script:
        raise RuntimeError("FantasyPoints data not found in page source")

    data = json.loads(script.string)

    # Look for the table data in props
    page_props = data.get("props", {}).get("pageProps", {})
    table_data = None
    for v in page_props.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "wr" in v[0]:
            table_data = v
            break

    if table_data is None:
        raise RuntimeError("WR/CB table structure not found in FantasyPoints JSON")

    df = pd.DataFrame(table_data)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "wr": "player",
        "wr_team": "team_abbr",
        "cb_team": "opponent_abbr",
        "adv": "matchup_adv",
        "slot_%": "slot_rate",
        "lwr_%": "left_align_rate",
        "rwr_%": "right_align_rate"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Clean names
    for col in ["player", "team_abbr", "opponent_abbr"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    out_path = Path("data/wr_cb_matchups.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ WR/CB matchup data saved to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    scrape_wr_cb()
