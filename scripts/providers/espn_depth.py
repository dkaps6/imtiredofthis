#!/usr/bin/env python3
"""
Scrape ESPN depth charts (unofficial) and write a simple normalized CSV:

data/depth_chart_espn.csv
  - team (NFL abbr)
  - player (display name)
  - position (QB/RB/WR/TE/FB/…)
  - role (QB1/RB1/WR1/WR2/SLOT/TE1 etc.) — best-effort tags
"""

from __future__ import annotations
from pathlib import Path
import re, time
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}

TEAM_ESPN = {
    # ESPN uses 2-4 letter team paths; we map from standard abbr
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin","CLE":"cle","DAL":"dal",
    "DEN":"den","DET":"det","GB":"gnb","HOU":"hou","IND":"ind","JAX":"jac","KC":"kan","LV":"lv","LAC":"lac","LAR":"lar",
    "MIA":"mia","MIN":"min","NE":"nwe","NO":"nor","NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea","SF":"sfo",
    "TB":"tam","TEN":"ten","WAS":"was",
}

def _fetch_team(team_abbr: str) -> list[dict]:
    url = f"https://www.espn.com/nfl/team/depth/_/name/{TEAM_ESPN[team_abbr]}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    for sec in soup.select("section div.Table__Title"):
        pos = sec.get_text(strip=True).upper()
        table = sec.find_parent("section").select_one("table.Table")
        if not table:
            continue
        df_list = pd.read_html(str(table))
        if not df_list:
            continue
        df = df_list[0]
        # ESPN depth tables often show starters in first column order
        # Normalize names from the table cells; columns vary by pos
        for col in df.columns:
            names = df[col].astype(str).tolist()
            for i, cell in enumerate(names):
                nm = re.sub(r"\s*\(.*?\)", "", cell).strip()
                if not nm or nm.lower() in ("nan","none"):
                    continue
                role = f"{pos}1" if i == 0 else (f"{pos}2" if i == 1 else f"{pos}{i+1}")
                if pos == "WR" and i == 2:
                    role = "SLOT"
                rows.append({"team": team_abbr, "position": pos, "player": nm, "role": role})
    return rows

def main():
    Path("data").mkdir(exist_ok=True)
    all_rows = []
    for team in TEAM_ESPN.keys():
        try:
            all_rows.extend(_fetch_team(team))
            time.sleep(0.5)
        except Exception as e:
            print(f"[espn_depth] {team}: {e}")
    out = pd.DataFrame(all_rows)
    out.to_csv("data/depth_chart_espn.csv", index=False)
    print(f"[espn_depth] wrote rows={len(out)} → data/depth_chart_espn.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
