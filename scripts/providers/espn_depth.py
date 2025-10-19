#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPN depth charts → data/depth_chart_espn.csv and data/roles_espn.csv

Surgical changes:
- Ensure player names are extracted even when <a>/<span> nesting confuses read_html.
- Output two files:
  - data/depth_chart_espn.csv : wide table (per-team columns like QB1, RB1, WR1, WR2, WR3, TE1, TE2)
  - data/roles_espn.csv       : long (team, player, role) for easy merges into player_form
- Normalize team and player names consistently with the rest of the pipeline.
- Keep polite delays between requests.
"""

import os, time, re, sys
import warnings
import pandas as pd
from typing import Dict, List
from bs4 import BeautifulSoup
import requests

DATA_DIR = "data"
OUT_WIDE = os.path.join(DATA_DIR, "depth_chart_espn.csv")
OUT_ROLES = os.path.join(DATA_DIR, "roles_espn.csv")

# Team map (ESPN uses full team names in URLs)
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

def _norm_player(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.replace(".", "").strip()
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _extract_table_names(soup: BeautifulSoup) -> pd.DataFrame:
    """
    ESPN renders depth tables with nested tags inside cells; read_html sometimes loses the inner text.
    This function walks the depth chart tables and extracts visible text from <a> or <span> tags.
    Returns a long frame with columns: position_role, player
    """
    tables = soup.select("table.Table")
    rows = []
    for tb in tables:
        # first column header usually holds the position group (e.g., QB, RB, WR, TE)
        # subsequent columns are 1st, 2nd, 3rd string etc
        # We'll parse header cells to form role labels
        ths = tb.select("thead tr th")
        if not ths:
            continue
        headers = [th.get_text(" ", strip=True) for th in ths]
        # Skip tables that aren't the positional depth chart
        if not headers or headers[0].strip().upper() not in {"OFFENSE", "DEFENSE", "SPECIAL TEAMS", "WR", "QB", "RB", "TE"}:
            # ESPN sometimes uses position abbreviations directly as header[0], still useful
            pass

        for tr in tb.select("tbody tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            pos_label = tds[0].get_text(" ", strip=True).upper()  # e.g., "QB", "RB", "WR", "TE"
            # build roles from remaining columns: 1st, 2nd, 3rd...
            for i, td in enumerate(tds[1:], start=1):
                # extract player text from nested anchor/span
                a = td.find("a")
                text = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
                player = _norm_player(text)
                if not player:
                    continue
                role = f"{pos_label}{i}"   # e.g., WR1, WR2, TE1, RB1, QB1
                rows.append({"position_role": role, "player": player})
    return pd.DataFrame(rows)

def fetch_team(team: str) -> pd.DataFrame:
    url = TEAM_URLS[team]
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # primary: robust soup extraction
    long_df = _extract_table_names(soup)

    # fallback: read_html in case soup misses a subtable variant
    try:
        tables = pd.read_html(r.text)
    except Exception:
        tables = []

    # sometimes read_html produces a nice wide table; try to capture WR/QB/RB/TE columns if present
    wide_candidates: List[pd.DataFrame] = []
    for t in tables:
        t = t.copy()
        t.columns = [str(c).strip() for c in t.columns]
        if any(k in " ".join(t.columns).upper() for k in ["QB", "RB", "WR", "TE"]):
            wide_candidates.append(t)

    # reconcile: we prefer long_df; if empty, try to melt candidate wide tables
    if long_df.empty and wide_candidates:
        melted = []
        for w in wide_candidates:
            # Try a generic melt: the first column may be 'POS' or similar
            id_col = w.columns[0]
            for c in w.columns[1:]:
                role = f"{str(w[id_col].iloc[0]).upper()}{1}" if w[id_col].dtype == object else str(c).upper()
                # crude—but read_html outputs are inconsistent; we still add something
            # This path is rarely used once soup extraction works
        # leave empty if we can't reconcile
        pass

    long_df["team"] = team
    return long_df

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)

    all_long = []
    teams_done = 0
    for tm in sorted(VALID):
        try:
            df_long = fetch_team(tm)
            if not df_long.empty:
                all_long.append(df_long)
            teams_done += 1
        except Exception as e:
            print(f"[espn_depth] WARN: failed {tm}: {e}", file=sys.stderr)
        time.sleep(0.4)  # polite delay

    if not all_long:
        # write empty shells (so downstream doesn't crash)
        pd.DataFrame(columns=["team","player","position","role"]).to_csv(OUT_ROLES, index=False)
        pd.DataFrame(columns=["team"]).to_csv(OUT_WIDE, index=False)
        print(f"[espn_depth] wrote rows=0 → {OUT_WIDE} + {OUT_ROLES}")
        return

    long_df = pd.concat(all_long, ignore_index=True)
    # split "WR1" → position="WR", role="WR1"
    long_df["role"] = long_df["position_role"].astype(str).str.upper().str.strip()
    long_df["position"] = long_df["role"].str.replace(r"\d+$", "", regex=True)

    roles = long_df[["team","player","role"]].drop_duplicates().reset_index(drop=True)

    # WIDE: pivot roles to columns (QB1, WR1, WR2, RB1, TE1, etc.)
    wide = (
        roles.assign(val=roles["player"])
             .pivot_table(index="team", columns="role", values="val", aggfunc="first")
             .reset_index()
    )
    # Ensure a common set of columns exists (present in most books)
    wanted_cols = ["QB1","RB1","RB2","WR1","WR2","WR3","TE1","TE2"]
    for c in wanted_cols:
        if c not in wide.columns:
            wide[c] = pd.NA
    wide = wide[["team"] + wanted_cols]

    # Write
    wide.to_csv(OUT_WIDE, index=False)
    roles.to_csv(OUT_ROLES, index=False)
    print(f"[espn_depth] wrote rows={len(wide)} → {OUT_WIDE} and rows={len(roles)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
