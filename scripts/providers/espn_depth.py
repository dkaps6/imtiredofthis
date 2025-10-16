#!/usr/bin/env python3
# scripts/providers/espn_depth.py
from __future__ import annotations
from pathlib import Path
import re, time
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}
TEAM_ESPN = {
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin","CLE":"cle","DAL":"dal",
    "DEN":"den","DET":"det","GB":"gnb","HOU":"hou","IND":"ind","JAX":"jac","KC":"kan","LV":"lv","LAC":"lac","LAR":"lar",
    "MIA":"mia","MIN":"min","NE":"nwe","NO":"nor","NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea","SF":"sfo",
    "TB":"tam","TEN":"ten","WAS":"was",
}

def _fetch_team(team: str) -> list[dict]:
    url = f"https://www.espn.com/nfl/team/depth/_/name/{TEAM_ESPN[team]}"
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = []
    for sec in soup.select("section div.Table__Title"):
        pos = sec.get_text(strip=True).upper()
        table = sec.find_parent("section").select_one("table.Table")
        if not table: continue
        dfs = pd.read_html(str(table))
        if not dfs: continue
        df = dfs[0]
        for col in df.columns:
            for i, cell in enumerate(df[col].astype(str).tolist()):
                nm = re.sub(r"\s*\(.*?\)", "", cell).strip()
                if not nm or nm.lower() in ("nan", "none"): continue
                role = f"{pos}1" if i == 0 else (f"{pos}2" if i == 1 else f"{pos}{i+1}")
                if pos == "WR" and i == 2: role = "SLOT"
                rows.append({"team": team, "position": pos, "player": nm, "role": role})
    return rows

def main():
    Path("data").mkdir(exist_ok=True)
    all_rows = []
    for t in TEAM_ESPN:
        try:
            all_rows += _fetch_team(t); time.sleep(0.5)
        except Exception as e:
            print(f"[espn_depth] {t}: {e}")
    pd.DataFrame(all_rows).to_csv("data/depth_chart_espn.csv", index=False)
    print(f"[espn_depth] wrote rows={len(all_rows)} → data/depth_chart_espn.csv"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
