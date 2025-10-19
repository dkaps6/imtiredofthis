#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ourlads depth charts → data/roles_ourlads.csv

Surgical changes:
- Normalize player names (strip suffixes, periods).
- Guarantee output columns: player, team, role.
- Defensive coding around their table structure changes.
"""

import os, sys, re, time
import warnings
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict

DATA_DIR = "data"
OUT_ROLES = os.path.join(DATA_DIR, "roles_ourlads.csv")

TEAM_URLS: Dict[str, str] = {
    "ARI": "https://www.ourlads.com/nfldepthcharts/depthchart/ARI",
    "ATL": "https://www.ourlads.com/nfldepthcharts/depthchart/ATL",
    "BAL": "https://www.ourlads.com/nfldepthcharts/depthchart/BAL",
    "BUF": "https://www.ourlads.com/nfldepthcharts/depthchart/BUF",
    "CAR": "https://www.ourlads.com/nfldepthcharts/depthchart/CAR",
    "CHI": "https://www.ourlads.com/nfldepthcharts/depthchart/CHI",
    "CIN": "https://www.ourlads.com/nfldepthcharts/depthchart/CIN",
    "CLE": "https://www.ourlads.com/nfldepthcharts/depthchart/CLE",
    "DAL": "https://www.ourlads.com/nfldepthcharts/depthchart/DAL",
    "DEN": "https://www.ourlads.com/nfldepthcharts/depthchart/DEN",
    "DET": "https://www.ourlads.com/nfldepthcharts/depthchart/DET",
    "GB":  "https://www.ourlads.com/nfldepthcharts/depthchart/GB",
    "HOU": "https://www.ourlads.com/nfldepthcharts/depthchart/HOU",
    "IND": "https://www.ourlads.com/nfldepthcharts/depthchart/IND",
    "JAX": "https://www.ourlads.com/nfldepthcharts/depthchart/JAX",
    "KC":  "https://www.ourlads.com/nfldepthcharts/depthchart/KC",
    "LAC": "https://www.ourlads.com/nfldepthcharts/depthchart/LAC",
    "LAR": "https://www.ourlads.com/nfldepthcharts/depthchart/LAR",
    "LV":  "https://www.ourlads.com/nfldepthcharts/depthchart/LV",
    "MIA": "https://www.ourlads.com/nfldepthcharts/depthchart/MIA",
    "MIN": "https://www.ourlads.com/nfldepthcharts/depthchart/MIN",
    "NE":  "https://www.ourlads.com/nfldepthcharts/depthchart/NE",
    "NO":  "https://www.ourlads.com/nfldepthcharts/depthchart/NO",
    "NYG": "https://www.ourlads.com/nfldepthcharts/depthchart/NYG",
    "NYJ": "https://www.ourlads.com/nfldepthcharts/depthchart/NYJ",
    "PHI": "https://www.ourlads.com/nfldepthcharts/depthchart/PHI",
    "PIT": "https://www.ourlads.com/nfldepthcharts/depthchart/PIT",
    "SEA": "https://www.ourlads.com/nfldepthcharts/depthchart/SEA",
    "SF":  "https://www.ourlads.com/nfldepthcharts/depthchart/SF",
    "TB":  "https://www.ourlads.com/nfldepthcharts/depthchart/TB",
    "TEN": "https://www.ourlads.com/nfldepthcharts/depthchart/TEN",
    "WAS": "https://www.ourlads.com/nfldepthcharts/depthchart/WAS",
}

SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=re.IGNORECASE)

def _norm_player(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.replace(".", "").strip()
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def fetch_team_roles(team: str) -> pd.DataFrame:
    url = TEAM_URLS[team]
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    # Ourlads has rows: <tr><td>Pos</td><td>Player1</td><td>Player2</td>...</tr>
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        pos = tds[0].get_text(" ", strip=True).upper()
        # standardized positions we care about
        if pos not in {"QB","RB","WR","TE"}:
            continue
        for i, td in enumerate(tds[1:], start=1):
            a = td.find("a")
            text = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
            player = _norm_player(text)
            if not player:
                continue
            role = f"{pos}{i}"
            rows.append({"team": team, "player": player, "role": role})

    return pd.DataFrame(rows)

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)

    all_roles = []
    for tm in sorted(TEAM_URLS.keys()):
        try:
            df = fetch_team_roles(tm)
            if not df.empty:
                all_roles.append(df)
        except Exception as e:
            print(f"[ourlads_depth] WARN: failed {tm}: {e}", file=sys.stderr)
        time.sleep(0.4)

    if not all_roles:
        pd.DataFrame(columns=["team","player","role"]).to_csv(OUT_ROLES, index=False)
        print(f"[ourlads_depth] wrote rows=0 → {OUT_ROLES}")
        return

    roles = pd.concat(all_roles, ignore_index=True).drop_duplicates()
    roles.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(roles)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
