#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPN depth charts → data/depth_chart_espn.csv and data/roles_espn.csv

Surgical upgrades:
- Robust name extraction from nested <a>/<span>.
- Strip jersey-number prefixes just in case.
- Deduplicate to best role per (team, player).
- Session with browser headers, retries, jitter, rotating Referer to decrease 403s.
- Keep prior good outputs when blocked (don't overwrite with empty files).
"""

import os, time, re, sys, random, warnings
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
OUT_WIDE = os.path.join(DATA_DIR, "depth_chart_espn.csv")
OUT_ROLES = os.path.join(DATA_DIR, "roles_espn.csv")

# ESPN depth pages
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
LEADING_NUM_RE = re.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", re.UNICODE)  # "#8", "8", "8 –", etc.

def _norm_player(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.replace(".", "").strip()
    s = LEADING_NUM_RE.sub("", s)            # strip jersey-number prefix defensively
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _role_rank(role: str) -> int:
    m = re.search(r"(\d+)$", str(role))
    return int(m.group(1)) if m else 999

# Single shared session w/ browser headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})

def _extract_table_names(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Walk the depth tables and extract visible player text.
    Returns columns: position_role (like 'WR1'), player
    """
    tables = soup.select("table.Table")
    rows = []
    for tb in tables:
        ths = tb.select("thead tr th")
        if not ths:
            continue

        for tr in tb.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            pos_label = tds[0].get_text(" ", strip=True).upper()  # e.g., QB/RB/WR/TE
            if pos_label not in {"QB","RB","WR","TE"}:
                continue
            for i, td in enumerate(tds[1:], start=1):
                a = td.find("a")
                text = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
                player = _norm_player(text)
                if not player or player.isdigit():
                    continue
                rows.append({"position_role": f"{pos_label}{i}", "player": player})
    return pd.DataFrame(rows)

def fetch_team(team: str) -> pd.DataFrame:
    url = TEAM_URLS[team]
    last_err = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 403:
                # backoff + rotate referer; ESPN sometimes blocks CI IPs on first hit
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
    # give up after retries
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
        time.sleep(0.4)  # polite delay

    # If ESPN blocked all teams, keep yesterday's roles instead of writing empties
    if not all_long:
        kept = False
        if os.path.exists(OUT_ROLES) or os.path.exists(OUT_WIDE):
            kept = True
            print("[espn_depth] 403/empty today — keeping previously generated roles files.")
        else:
            # first ever run: write minimal shells so downstream doesn't crash
            pd.DataFrame(columns=["team","player","position","role"]).to_csv(OUT_ROLES, index=False)
            pd.DataFrame(columns=["team"]).to_csv(OUT_WIDE, index=False)
        print(f"[espn_depth] wrote rows=0 (kept_prior={kept}) → {OUT_WIDE} / {OUT_ROLES}")
        return

    long_df = pd.concat(all_long, ignore_index=True)

    # Promote WR1 -> position=WR, role=WR1 and dedupe per (team, player) to best depth
    roles = long_df.assign(
        role=lambda d: d["position_role"].astype(str).str.upper().str.strip(),
        position=lambda d: d["position_role"].astype(str).str.replace(r"\d+$", "", regex=True)
    )[["team","player","role","position"]]

    roles["_rank"] = roles["role"].map(_role_rank)
    roles = (
        roles.sort_values(["team","player","_rank"])
             .drop_duplicates(["team","player"], keep="first")
             .drop(columns=["_rank"])
             .reset_index(drop=True)
    )

    # Build wide table (team × roles)
    wide = (
        roles.assign(val=roles["player"])
             .pivot_table(index="team", columns="role", values="val", aggfunc="first")
             .reset_index()
    )

    # Ensure consistent columns for your downstream usage
    wanted_cols = ["QB1","RB1","RB2","WR1","WR2","WR3","TE1","TE2"]
    for c in wanted_cols:
        if c not in wide.columns:
            wide[c] = pd.NA
    wide = wide[["team"] + wanted_cols]

    # Write
    wide.to_csv(OUT_WIDE, index=False)
    roles[["team","player","role"]].to_csv(OUT_ROLES, index=False)
    print(f"[espn_depth] wrote rows={len(wide)} → {OUT_WIDE} and rows={len(roles)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
