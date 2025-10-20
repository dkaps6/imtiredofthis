#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/espn_depth.py (corrected, non-destructive outputs)
import os, time, re, sys, random, warnings
from typing import Dict, List
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
OUT_WIDE = os.path.join(DATA_DIR, "depth_chart_espn.csv")
OUT_ROLES = os.path.join(DATA_DIR, "roles_espn.csv")

TEAM_URLS: Dict[str, str] = {
    "ARI": "https://www.espn.com/nfl/team/depth/_/name/ari",
    "ATL": "https://www.espn.com/nfl/team/depth/_/name/atl",
    "BAL": "https://www.espn.com/nfl/team/depth/_/name/bal",
    "BUF": "https://www.espn.com/nfl/team/depth/_/name/buf",
    "CAR": "https://www.espn.com/nfl/team/depth/_/name/car",
    "CHI": "https://www.espn.com/nfl/team/depth/_/name/chi",
    "CIN": "https://www.espn.com/nfl/team/depth/_/name/cin",
    "CLE": "https://www.espn.com/nfl/team/depth/_/name/cle",
    "DAL": "https://www.espn.com/nfl/team/depth/_/name/dal",
    "DEN": "https://www.espn.com/nfl/team/depth/_/name/den",
    "DET": "https://www.espn.com/nfl/team/depth/_/name/det",
    "GB":  "https://www.espn.com/nfl/team/depth/_/name/gb",
    "HOU": "https://www.espn.com/nfl/team/depth/_/name/hou",
    "IND": "https://www.espn.com/nfl/team/depth/_/name/ind",
    "JAX": "https://www.espn.com/nfl/team/depth/_/name/jax",
    "KC":  "https://www.espn.com/nfl/team/depth/_/name/kc",
    "LAC": "https://www.espn.com/nfl/team/depth/_/name/lac",
    "LAR": "https://www.espn.com/nfl/team/depth/_/name/lar",
    "LV":  "https://www.espn.com/nfl/team/depth/_/name/lv",
    "MIA": "https://www.espn.com/nfl/team/depth/_/name/mia",
    "MIN": "https://www.espn.com/nfl/team/depth/_/name/min",
    "NE":  "https://www.espn.com/nfl/team/depth/_/name/ne",
    "NO":  "https://www.espn.com/nfl/team/depth/_/name/no",
    "NYG": "https://www.espn.com/nfl/team/depth/_/name/nyg",
    "NYJ": "https://www.espn.com/nfl/team/depth/_/name/nyj",
    "PHI": "https://www.espn.com/nfl/team/depth/_/name/phi",
    "PIT": "https://www.espn.com/nfl/team/depth/_/name/pit",
    "SEA": "https://www.espn.com/nfl/team/depth/_/name/sea",
    "SF":  "https://www.espn.com/nfl/team/depth/_/name/sf",
    "TB":  "https://www.espn.com/nfl/team/depth/_/name/tb",
    "TEN": "https://www.espn.com/nfl/team/depth/_/name/ten",
    "WAS": "https://www.espn.com/nfl/team/depth/_/name/wsh",
}
VALID = set(TEAM_URLS.keys())

SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=re.IGNORECASE)
LEADING_NUM_RE = re.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", re.UNICODE)

def _norm_player(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.replace(".", "").strip()
    s = LEADING_NUM_RE.sub("", s)
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _role_rank(role: str) -> int:
    m = re.search(r"(\d+)$", str(role))
    return int(m.group(1)) if m else 999

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})

def _extract_table_names(soup: BeautifulSoup) -> pd.DataFrame:
    tables = soup.select("table.Table")
    rows = []
    for tb in tables:
        ths = tb.select("thead tr th")
        if not ths:
            continue
        pos_label = ths[0].get_text(" ", strip=True).upper()
        if pos_label not in {"QB","RB","WR","TE"}:
            continue
        for tr in tb.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            for i, td in enumerate(tds[1:], start=1):
                a = td.find("a")
                text = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
                player = _norm_player(text)
                if not player or player.isdigit():
                    continue
                role = f"{pos_label}{i}"
                rows.append({"position_role": role, "player": player})
    return pd.DataFrame(rows)

def fetch_team(team: str) -> pd.DataFrame:
    url = TEAM_URLS[team]
    last_err = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 403:
                time.sleep(0.6 + random.random()*0.8)
                SESSION.headers["Referer"] = f"https://www.espn.com/nfl/team/_/name/{team.lower()}"
                continue
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            long_df = _extract_table_names(soup)
            long_df["team"] = team
            return long_df
        except Exception as e:
            last_err = e
            time.sleep(0.6 + random.random()*0.8)
    raise last_err if last_err else RuntimeError("ESPN depth fetch failed")

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)

    all_long: List[pd.DataFrame] = []
    for tm in sorted(VALID):
        try:
            df_long = fetch_team(tm)
            if not df_long.empty:
                all_long.append(df_long)
        except Exception as e:
            print(f"[espn_depth] WARN: failed {tm}: {e}", file=sys.stderr)
        time.sleep(0.4)

    if not all_long:
        # keep prior outputs if they exist; otherwise create minimal headers once
        if not os.path.exists(OUT_ROLES):
            pd.DataFrame(columns=["team","player","role"]).to_csv(OUT_ROLES, index=False)
        if not os.path.exists(OUT_WIDE):
            pd.DataFrame(columns=["team"]).to_csv(OUT_WIDE, index=False)
        print("[espn_depth] 0 rows; kept prior outputs.")
        return

    long_df = pd.concat(all_long, ignore_index=True)

    long_df["_rank"] = long_df["position_role"].str.extract(r"(\d+)$", expand=False).astype(float).fillna(999).astype(int)
    roles = (long_df.sort_values(["team","player","_rank"])
                    .drop_duplicates(["team","player"], keep="first")
                    .assign(role=lambda d: d["position_role"].str.replace(r"[^A-Z]+", "", regex=True))
                    [["team","player","role"]])
    roles.to_csv(OUT_ROLES, index=False)

    wide = long_df.copy()
    wide["pos"]  = wide["position_role"].str.replace(r"\d+$", "", regex=True)
    wide["slot"] = wide["position_role"].str.extract(r"(\d+)$", expand=False).astype(float).fillna(1).astype(int)
    pivot = (wide.pivot_table(index="team", columns=["pos","slot"], values="player", aggfunc="first")
                   .sort_index(axis=1))
    pivot.columns = [f"{p}{s}" for p,s in pivot.columns]
    pivot.reset_index().to_csv(OUT_WIDE, index=False)
    print(f"[espn_depth] wrote roles={len(roles)} → {OUT_ROLES}; wide={len(pivot)} → {OUT_WIDE}")

if __name__ == "__main__":
    main()
