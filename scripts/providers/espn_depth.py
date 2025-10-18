#!/usr/bin/env python3
# scripts/providers/espn_depth.py
from __future__ import annotations
from pathlib import Path
import re, time
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}
TEAM_ESPN = {
    "ARI": "ari","ATL": "atl","BAL": "bal","BUF": "buf","CAR": "car","CHI": "chi","CIN": "cin","CLE": "cle",
    "DAL": "dal","DEN": "den","DET": "det","GB": "gb","HOU": "hou","IND": "ind","JAX": "jax","KC": "kc",
    "LV": "lv","LAC": "lac","LAR": "lar","MIA": "mia","MIN": "min","NE": "ne","NO": "no","NYG": "nyg",
    "NYJ": "nyj","PHI": "phi","PIT": "pit","SEA": "sea","SF": "sf","TB": "tb","TEN": "ten","WAS": "wsh",
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
                rows.append({"team": team, "position": pos, "player": nm.replace(".",""), "role": role})
    return rows

def main():
    Path(DATA_DIR).mkdir(exist_ok=True)
    all_rows = []
    for t in TEAM_ESPN:
        try:
            all_rows += _fetch_team(t); time.sleep(0.5)
        except Exception as e:
            print(f"[espn_depth] {t}: {e}")
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{DATA_DIR}/depth_chart_espn.csv", index=False)
    # also emit roles_espn.csv and update merged roles.csv (prefer ESPN over others)
    df[["player","team","role"]].to_csv(f"{DATA_DIR}/roles_espn.csv", index=False)
    try:
        # merge preference: ESPN first, then ourlads
        espn = pd.read_csv(f"{DATA_DIR}/roles_espn.csv")
        try:
            ol = pd.read_csv(f"{DATA_DIR}/roles_ourlads.csv")
        except Exception:
            ol = pd.DataFrame(columns=espn.columns)
        merged = pd.concat([espn, ol], ignore_index=True)
        merged = merged.drop_duplicates(subset=["player","team"], keep="first")
        merged.to_csv(f"{DATA_DIR}/roles.csv", index=False)
    except Exception:
        pass
    print(f"[espn_depth] wrote rows={len(df)} → data/depth_chart_espn.csv + roles_espn.csv (and roles.csv)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
