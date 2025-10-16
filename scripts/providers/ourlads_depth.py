#!/usr/bin/env python3
"""
Scrape OurLads NFL depth charts and write:

data/depth_chart_ourlads.csv
  - team, player, position, role
"""

from __future__ import annotations
from pathlib import Path
import time, re
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}

TEAM_OL = {
    "ARI":"arizona-cardinals","ATL":"atlanta-falcons","BAL":"baltimore-ravens","BUF":"buffalo-bills",
    "CAR":"carolina-panthers","CHI":"chicago-bears","CIN":"cincinnati-bengals","CLE":"cleveland-browns",
    "DAL":"dallas-cowboys","DEN":"denver-broncos","DET":"detroit-lions","GB":"green-bay-packers",
    "HOU":"houston-texans","IND":"indianapolis-colts","JAX":"jacksonville-jaguars","KC":"kansas-city-chiefs",
    "LV":"las-vegas-raiders","LAC":"los-angeles-chargers","LAR":"los-angeles-rams",
    "MIA":"miami-dolphins","MIN":"minnesota-vikings","NE":"new-england-patriots","NO":"new-orleans-saints",
    "NYG":"new-york-giants","NYJ":"new-york-jets","PHI":"philadelphia-eagles","PIT":"pittsburgh-steelers",
    "SEA":"seattle-seahawks","SF":"san-francisco-49ers","TB":"tampa-bay-buccaneers","TEN":"tennessee-titans",
    "WAS":"washington-commanders"
}

def _fetch_team(team_abbr: str) -> list[dict]:
    slug = TEAM_OL[team_abbr]
    url = f"https://www.ourlads.com/nfldepthcharts/depthchart/{slug}"
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    for table in soup.select("table.dc-table"):
        # Position header is in the previous sibling or a th
        heading = table.find_previous("h2")
        pos = (heading.get_text(strip=True).upper() if heading else "UNK")
        df_list = pd.read_html(str(table))
        if not df_list:
            continue
        df = df_list[0]
        # First column is typically starter
        for i, col in enumerate(df.columns):
            names = df[col].astype(str).tolist()
            for j, cell in enumerate(names):
                nm = re.sub(r"\s*\(.*?\)", "", cell).strip()
                if not nm or nm.lower() in ("nan","none"):
                    continue
                role = f"{pos}1" if j == 0 else (f"{pos}2" if j == 1 else f"{pos}{j+1}")
                if pos == "WR" and j == 2:
                    role = "SLOT"
                rows.append({"team": team_abbr, "position": pos, "player": nm, "role": role})
    return rows

def main():
    Path("data").mkdir(exist_ok=True)
    all_rows = []
    for team in TEAM_OL.keys():
        try:
            all_rows.extend(_fetch_team(team))
            time.sleep(0.5)
        except Exception as e:
            print(f"[ourlads_depth] {team}: {e}")
    out = pd.DataFrame(all_rows)
    out.to_csv("data/depth_chart_ourlads.csv", index=False)
    print(f"[ourlads_depth] wrote rows={len(out)} → data/depth_chart_ourlads.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
