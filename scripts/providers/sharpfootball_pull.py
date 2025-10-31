#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp Football team metrics puller (2025 hardened)

What this does:
- Fetches several Sharp Football tables (def/off tendencies, coverage by position, DL/OL, pace)
- Parses HTML locally (never hands the URL to pandas.read_html to avoid 403)
- Robustly detects the right table & normalizes columns
- Writes per-source CSVs + a merged team form CSV

Env (optional):
- SHARP_HTTP_PROXY / SHARP_HTTPS_PROXY: HTTP(S) proxy URLs if you need them

Outputs (in ./data):
- sharp_def_tendencies_<season>.csv
- sharp_off_tendencies_<season>.csv
- sharp_coverage_pos_<season>.csv
- sharp_dl_<season>.csv
- sharp_ol_<season>.csv
- sharp_pace_<season>.csv
- sharp_team_form.csv  (merged)
"""

from __future__ import annotations
import os
import re
import sys
import json
import time
import math
import string
import argparse
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

URLS = {
    "def_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-tendencies/",
    "off_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-tendencies-stats/",
    "coverage_pos": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-stats-by-position/",
    "coverage_scheme": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/",
    "dl": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-line-stats/",
    "ol": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-line-stats/",
    "pace": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-team-pace-stats/",
}

# Basic name normalization map (Sharp team names -> abbreviations)
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS", "Washington Redskins": "WAS",
    # some sites use city only
    "Washington": "WAS", "New Orleans": "NO", "New England": "NE", "San Francisco": "SF",
    "Tampa Bay": "TB", "Los Angeles": "LA",
}

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    # optional proxies via env (SHARP_HTTP_PROXY / SHARP_HTTPS_PROXY)
    http_proxy = os.getenv("SHARP_HTTP_PROXY")
    https_proxy = os.getenv("SHARP_HTTPS_PROXY")
    if http_proxy or https_proxy:
        s.proxies.update({
            "http": http_proxy or "",
            "https": https_proxy or "",
        })
    return s

def _fetch_html(url: str, season: int, dump_prefix: str) -> Optional[str]:
    """
    Fetch the page HTML (with/without season query) and return content.
    Always dump last HTML for debugging to data/_sharp_dump_<name>_<season>.html
    """
    s = _session()
    # try with season first, then without
    tried = []
    for with_season in (True, False):
        full = url
        if with_season:
            sep = "&" if "?" in url else "?"
            full = f"{url}{sep}season={season}"
        tried.append(full)
        try:
            r = s.get(full, headers=HEADERS, timeout=20)
            # we don’t trust status alone; some CDNs return 200 with block content.
            html = r.text
            # dump every attempt (last wins)
            dump_path = os.path.join(DATA_DIR, f"_sharp_dump_{dump_prefix}_{season}.html")
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(html)
            return html
        except Exception:
            time.sleep(0.5)
            continue
    return None

def _read_single_table_from_html(html: str) -> Optional[pd.DataFrame]:
    """
    Read tables from *HTML string* with pandas and return the largest one.
    (Never pass a URL into read_html to avoid 403.)
    """
    if not html:
        return None
    try:
        # prefer lxml, fallback to html5lib
        dfs = pd.read_html(StringIO(html), flavor="lxml")
    except Exception:
        try:
            dfs = pd.read_html(StringIO(html), flavor="bs4")
        except Exception:
            return None
    if not dfs:
        return None
    # pick the table with the most rows×cols (usually the real one)
    scores = [(i, (df.shape[0] * max(1, df.shape[1]))) for i, df in enumerate(dfs)]
    idx = max(scores, key=lambda x: x[1])[0]
    return dfs[idx]

def _slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", "_", s)
    return s


OFF_TEND_ALIASES = {
    "pass_rate_over_expected": "pass_rate_over_expected",
    "pass_rate_over_expectation": "pass_rate_over_expected",
    "pass_rate_over_exp": "pass_rate_over_expected",
    "proe": "pass_rate_over_expected",
    "pass_rate_vs_expected": "pass_rate_over_expected",
    "neutral_pass_rate": "neutral_db_rate",
    "neutral_pass_rate_last_5": "neutral_db_rate_last_5",
}

PACE_ALIASES = {
    "neutral_pace": "neutral_pace",
    "seconds_per_play": "seconds_per_play",
    "seconds_per_play_last_5": "seconds_per_play_last5",
    "plays_per_game": "plays_per_game",
}

COVERAGE_ALIASES = {
    "man_coverage_rate": "coverage_man_rate",
    "zone_coverage_rate": "coverage_zone_rate",
    "man_coverage_pct": "coverage_man_rate",
    "zone_coverage_pct": "coverage_zone_rate",
}


def _rename_expected_cols(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {}
    if kind == "off_tend":
        alias_map = OFF_TEND_ALIASES
    elif kind == "pace":
        alias_map = PACE_ALIASES
    elif kind == "coverage_pos":
        alias_map = COVERAGE_ALIASES

    if not alias_map:
        return df

    rename_map = {}
    for col in df.columns:
        key = _slug(col)
        if key in alias_map:
            rename_map[col] = alias_map[key]

    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _normalize_team_col(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: _slug(c) for c in df.columns}
    df = df.rename(columns=cols)

    # find team column by fuzzy name
    team_col = None
    for cand in df.columns:
        if any(tok in cand for tok in ("team", "defense", "offense", "club", "squad")):
            team_col = cand
            break
    if team_col is None:
        # sometimes first col is team
        team_col = df.columns[0]

    # rename to 'team' and map to ABBR when possible
    if team_col != "team":
        df = df.rename(columns={team_col: "team"})

    # drop rows where team is missing/aggregate
    df["team"] = df["team"].astype(str).str.strip()
    df = df[~df["team"].str.contains("Rank|Ranking|Ranks|NFL", case=False, na=False)]
    df.loc[:, "team_abbr"] = df["team"].map(TEAM_ABBR).fillna(df["team"])
    return df

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("team", "team_abbr"):
            continue
        # strip % and commas
        ser = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(ser, errors="coerce")
    return df

def _rename_expected_cols(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    """"
    Map Sharp's column headers (after slugging/to_numeric) to the canonical names
    our downstream code expects in team_form.csv.

    We only rename columns that actually exist. If a column isn't there, we skip.
    """

    # We'll build a dict of {old_col: new_col} per kind.

    # For pace table we want neutral pace as `neutral_pace`
    # Sharp may call it things like "neutral pace", "neutral_pace", "neutral pace %", etc.
    pace_aliases = {
        "neutral_pace": "neutral_pace",
        "neutral_pace_": "neutral_pace",
        "neutral_pace_percent": "neutral_pace",
        "neutral_pace_pct": "neutral_pace",
        "neutral_pace__seconds_per_play": "neutral_pace",  # fallback
    }

    # For off_tend we expect pass_rate_over_expected
    off_tend_aliases = {
        "pass_rate_over_expected": "pass_rate_over_expected",
        "pass_rate_over_exp": "pass_rate_over_expected",
        "proe": "pass_rate_over_expected",
        "pass_rate_over_expected_pct": "pass_rate_over_expected",
        "pass_rate_over_exp_pct": "pass_rate_over_expected",
    }

    # For coverage_scheme we expect coverage_man_rate, coverage_zone_rate
    coverage_scheme_aliases = {
        "man_coverage_rate": "coverage_man_rate",
        "man_rate": "coverage_man_rate",
        "man_pct": "coverage_man_rate",
        "zone_coverage_rate": "coverage_zone_rate",
        "zone_rate": "coverage_zone_rate",
        "zone_pct": "coverage_zone_rate",
    }

    # Build final alias map for this kind
    alias_map = {}
    if kind == "pace":
        alias_map.update(pace_aliases)
    if kind == "off_tend":
        alias_map.update(off_tend_aliases)
    if kind == "coverage_scheme":
        alias_map.update(coverage_scheme_aliases)

    # Now apply the renames if present
    for old_col, new_col in alias_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    return df

def _save_csv(df: Optional[pd.DataFrame], out_path: str) -> int:
    if df is None or df.empty:
        return 0
    df.to_csv(out_path, index=False)
    return len(df)

def _pull_one(kind: str, url: str, season: int) -> Tuple[str, int, Optional[pd.DataFrame]]:
    html = _fetch_html(url, season, kind)
    df = _read_single_table_from_html(html or "")
    if df is None or df.empty:
        return kind, 0, None

    df = _normalize_team_col(df)
    df = _rename_expected_cols(kind, df)

    # simple, generic post-processing per kind (add more if/when we know exact col names)
    if kind in ("def_tend", "off_tend", "pace", "coverage_pos", "coverage_scheme", "dl", "ol"):
        df = _to_numeric(df)
        df = _rename_expected_cols(kind, df)

    out_name = f"sharp_{kind}_{season}.csv"
    out_path = os.path.join(DATA_DIR, out_name)
    n = _save_csv(df, out_path)
    return kind, n, df

def merge_team_form(season: int, pieces: Dict[str, pd.DataFrame]) -> int:
    """
    Merge available pieces on team_abbr. We only merge what we actually parsed.
    - Ensures no duplicate 'team_abbr' column
    - Ensures one row per team_abbr per piece before merging
    - Deduplicates columns defensively
    """
    base = None
    for k in ("def_tend", "off_tend", "pace", "coverage_pos", "coverage_scheme", "dl", "ol"):
        df = pieces.get(k)
        if df is None or df.empty:
            continue

        df = df.copy()
        # Make sure we have a single 'team_abbr' key column
        if "team_abbr" not in df.columns:
            if "team" in df.columns:
                df["team_abbr"] = df["team"]
            else:
                # nothing to join on; skip this piece
                continue

        # Drop any duplicate columns by name (including accidental dup 'team_abbr')
        df = df.loc[:, ~df.columns.duplicated()]

        # Keep only value columns (exclude keys)
        value_cols = [c for c in df.columns if c not in ("team", "team_abbr")]

        # Select once; ensure single row per team
        take = df[["team_abbr"] + value_cols].drop_duplicates(subset=["team_abbr"])

        # Merge non-destructively
        if base is None:
            base = take
        else:
            base = base.merge(take, on="team_abbr", how="outer")

    if base is None or base.empty:
        return 0

    # Write merged output
    out_path = os.path.join(DATA_DIR, "sharp_team_form.csv")
    base.to_csv(out_path, index=False)
    return len(base)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--dump-html", action="store_true", help="(kept for compatibility; dumps are always written)")
    args = ap.parse_args()

    season = args.season
    pieces: Dict[str, pd.DataFrame] = {}
    total_rows = 0

    for kind, url in URLS.items():
        k, n, df = _pull_one(kind, url, season)
        print(f"[sharp] {k:12s} rows={n}")
        if df is not None and n > 0:
            pieces[k] = df
            total_rows += n

    merged = merge_team_form(season, pieces)
    print(f"[sharp] wrote {merged} rows → {os.path.join(DATA_DIR, 'sharp_team_form.csv')}")
    needed = ["pass_rate_over_expected", "neutral_pace", "coverage_man_rate", "coverage_zone_rate"]
    print(
        "[sharp] sanity check cols:",
        {
            col: (
                col in pieces.get("off_tend", pd.DataFrame()).columns
                or col in pieces.get("pace", pd.DataFrame()).columns
                or col in pieces.get("coverage_scheme", pd.DataFrame()).columns
            )
            for col in needed
        },
    )

    # Non-zero exit if *everything* failed (lets the pipeline tell you)
    if merged == 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
