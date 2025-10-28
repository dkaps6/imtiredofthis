#!/usr/bin/env python3
# build_cb_coverage_team.py
# Fetches free, up-to-date NFL defensive coverage metrics by TEAM for 2025 YTD
# Sources (free):
# - SumerSports Defensive Team page (EPA/Pass, Success %): https://sumersports.com/teams/defensive/
# - NFL.com Team Defensive Passing (Att, 20+, Sck): https://www.nfl.com/stats/team-stats/defense/passing/2025/reg/all
# - Sharp Football Analysis Coverage Schemes (Man %, Zone %): https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/
#
# Output schema (CSV): team,success_allowed,epa_allowed,explosive_allowed,target_rate_allowed,man_rate,zone_rate
#
# Notes:
# - success_allowed uses SumerSports "Success %" (overall defensive success rate). If you prefer pass-only success, compute from play-by-play.
# - explosive_allowed is computed as 20+ completions allowed / opponent attempts (NFL.com columns "20+" and "Att").
# - target_rate_allowed is a transparent proxy: attempts / (attempts + sacks).
# - Man/Zone are read from Sharp Football's free coverage-schemes table.
#
# This script relies on pandas.read_html to parse public tables. If a site changes markup,
# scraping may require adjustment. Keep the URLs and column names in the constants below.

import sys
import io
import re
import time
from typing import Dict, Tuple
import pandas as pd
import requests

SUMER_DEF_URL = "https://sumersports.com/teams/defensive/"
NFL_DEF_PASS_URL = "https://www.nfl.com/stats/team-stats/defense/passing/2025/reg/all"
SHARP_COVERAGE_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"

# Canonical team codes per user spec
TEAM_CODE = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI",
    "Bengals":"CIN","Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB",
    "Texans":"HOU","Colts":"IND","Jaguars":"JAX","Chiefs":"KC","Raiders":"LV","Chargers":"LAC",
    "Rams":"LAR","Dolphins":"MIA","Vikings":"MIN","Patriots":"NE","Saints":"NO","Giants":"NYG",
    "Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","Seahawks":"SEA","49ers":"SF","Buccaneers":"TB",
    "Titans":"TEN","Commanders":"WAS"
}

def _read_html_single_table(url: str, match: str = None) -> pd.DataFrame:
    """Read the first HTML table that matches from a URL using pandas.read_html.
    If 'match' is provided, selects tables whose string repr contains the substring (case-insensitive)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    if match:
        m = match.lower()
        for t in tables:
            if m in t.to_string().lower():
                return t
        # fallback to first table
        return tables[0]
    else:
        return tables[0]

def fetch_sumer_defense() -> pd.DataFrame:
    """Fetch SumerSports defensive team stats (EPA/Pass, Success %) for 2025."""
    df = _read_html_single_table(SUMER_DEF_URL, match="EPA/Pass")
    # Normalize columns
    # Expect columns like: ['Team', 'Season', 'EPA/Play', 'Total EPA', 'Success %', 'EPA/Pass', ...]
    # Filter to 2025 season if Season present
    if "Season" in df.columns:
        df = df[df["Season"] == 2025]
    # Keep only needed
    cols_map = {}
    # Find columns robustly
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in ("team", "team "):
            cols_map[c] = "team_name"
        elif "success" in lc:
            cols_map[c] = "success_allowed"
        elif "epa/pass" in lc or (("epa" in lc) and ("pass" in lc)):
            cols_map[c] = "epa_allowed"
    df = df.rename(columns=cols_map)
    keep = ["team_name","success_allowed","epa_allowed"]
    df = df[[c for c in keep if c in df.columns]].copy()
    # Clean formats: success like "39.29%" -> 39.29
    if "success_allowed" in df.columns:
        df["success_allowed"] = (
            df["success_allowed"].astype(str).str.replace("%","",regex=False).astype(float).round(2)
        )
    if "epa_allowed" in df.columns:
        df["epa_allowed"] = pd.to_numeric(df["epa_allowed"], errors="coerce").round(2)
    # Map to codes
    df["team"] = df["team_name"].map(TEAM_CODE)
    df = df.dropna(subset=["team"]).drop(columns=["team_name"]).reset_index(drop=True)
    return df

def fetch_nfl_def_passing() -> pd.DataFrame:
    """Fetch NFL.com defensive passing table and compute explosive and target-rate proxies."""
    # NFL.com page includes a renderable table for pandas.read_html
    df = _read_html_single_table(NFL_DEF_PASS_URL, match="Team")
    # Normalize columns (expect: Team, Att, 20+, Sck)
    # Some tables repeat team names twice; drop duplicated columns if present
    df = df.loc[:,~df.columns.duplicated()].copy()
    # Robust column selection
    colmap = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc.startswith("team"):
            colmap[c] = "team_name"
        elif lc in ("att","att.") or "att" in lc:
            colmap[c] = "att"
        elif "20+" in str(c):
            colmap[c] = "twenty_plus"
        elif "sck" in lc or "sack" in lc:
            colmap[c] = "sacks"
    df = df.rename(columns=colmap)
    needed = ["team_name","att","twenty_plus","sacks"]
    df = df[[c for c in needed if c in df.columns]].copy()
    # Coerce numeric
    for c in ["att","twenty_plus","sacks"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Compute rates
    df["explosive_allowed"] = (df["twenty_plus"] / df["att"]).round(4)
    df["target_rate_allowed"] = (df["att"] / (df["att"] + df["sacks"])).round(4)
    df["team"] = df["team_name"].map(TEAM_CODE)
    df = df.dropna(subset=["team"]).drop(columns=["team_name"]).reset_index(drop=True)
    return df[["team","explosive_allowed","target_rate_allowed"]]

def fetch_sharp_coverage() -> pd.DataFrame:
    """Fetch man/zone rates from Sharp Football coverage schemes page."""
    # Try to read any coverage table; if multiple, choose one containing 'Man' and 'Zone'
    resp = requests.get(SHARP_COVERAGE_URL, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("man" in c for c in cols) and any("zone" in c for c in cols):
            target = t
            break
    if target is None:
        # fallback: first table
        target = tables[0]
    df = target.copy()
    # Normalize columns
    colmap = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if "team" in lc:
            colmap[c] = "team_name"
        elif "man" in lc:
            colmap[c] = "man_rate"
        elif "zone" in lc:
            colmap[c] = "zone_rate"
    df = df.rename(columns=colmap)
    df = df[[c for c in ["team_name","man_rate","zone_rate"] if c in df.columns]].copy()
    # Clean percents (e.g., 40.5% -> 40.5)
    for c in ["man_rate","zone_rate"]:
        df[c] = (
            df[c].astype(str).str.replace("%","",regex=False).str.extract(r"([0-9]+\.?[0-9]*)")[0]
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    df["team"] = df["team_name"].map(TEAM_CODE)
    df = df.dropna(subset=["team"]).drop(columns=["team_name"]).reset_index(drop=True)
    return df[["team","man_rate","zone_rate"]]

def main(out_path: str = "cb_coverage_team.csv") -> None:
    sumer = fetch_sumer_defense()
    nfl = fetch_nfl_def_passing()
    sharp = fetch_sharp_coverage()

    # Merge
    df = (
        sumer.merge(nfl, on="team", how="left")
             .merge(sharp, on="team", how="left")
             .sort_values("team")
    )

    # Ensure final schema and presence of all teams
    cols = ["team","success_allowed","epa_allowed","explosive_allowed","target_rate_allowed","man_rate","zone_rate"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Round/format
    df["success_allowed"] = pd.to_numeric(df["success_allowed"], errors="coerce").round(2)
    df["epa_allowed"] = pd.to_numeric(df["epa_allowed"], errors="coerce").round(2)
    df["explosive_allowed"] = pd.to_numeric(df["explosive_allowed"], errors="coerce").round(4)
    df["target_rate_allowed"] = pd.to_numeric(df["target_rate_allowed"], errors="coerce").round(4)
    df["man_rate"] = pd.to_numeric(df["man_rate"], errors="coerce").round(1)
    df["zone_rate"] = pd.to_numeric(df["zone_rate"], errors="coerce").round(1)

    # Reindex to include every code even if a site is momentarily missing one
    all_codes = [
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC",
        "LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
    ]
    df = df.set_index("team").reindex(all_codes).reset_index()

    # Save
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {df.shape}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    out_path = "cb_coverage_team.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out_path)
