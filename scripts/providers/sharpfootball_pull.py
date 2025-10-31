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
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
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

# Canonical codes we expect downstream (32 teams)
TEAM_CODES: Tuple[str, ...] = (
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
)

# Extensive alias mapping so every Sharp table normalizes to canonical codes
_TEAM_ALIASES: Dict[str, Iterable[str]] = {
    "ARI": ("ARI", "ARZ", "ARIZONA", "ARIZONA CARDINALS", "CARDINALS"),
    "ATL": ("ATL", "ATLANTA", "ATLANTA FALCONS", "FALCONS"),
    "BAL": ("BAL", "BALTIMORE", "BALTIMORE RAVENS", "RAVENS"),
    "BUF": ("BUF", "BUFFALO", "BUFFALO BILLS", "BILLS"),
    "CAR": ("CAR", "CAROLINA", "CAROLINA PANTHERS", "PANTHERS"),
    "CHI": ("CHI", "CHICAGO", "CHICAGO BEARS", "BEARS"),
    "CIN": ("CIN", "CINCINNATI", "CINCINNATI BENGALS", "BENGALS"),
    "CLE": ("CLE", "CLEVELAND", "CLEVELAND BROWNS", "BROWNS"),
    "DAL": ("DAL", "DALLAS", "DALLAS COWBOYS", "COWBOYS"),
    "DEN": ("DEN", "DENVER", "DENVER BRONCOS", "BRONCOS"),
    "DET": ("DET", "DETROIT", "DETROIT LIONS", "LIONS"),
    "GB": ("GB", "GBP", "GREEN BAY", "GREEN BAY PACKERS", "PACKERS"),
    "HOU": ("HOU", "HOUSTON", "HOUSTON TEXANS", "TEXANS"),
    "IND": ("IND", "INDIANAPOLIS", "INDIANAPOLIS COLTS", "COLTS"),
    "JAX": ("JAX", "JAC", "JACKSONVILLE", "JACKSONVILLE JAGUARS", "JAGUARS"),
    "KC": ("KC", "KAN", "KCC", "KANSAS CITY", "KANSAS CITY CHIEFS", "CHIEFS"),
    "LAC": (
        "LAC",
        "LA CHARGERS",
        "LOS ANGELES CHARGERS",
        "SAN DIEGO CHARGERS",
        "SD",
        "SANDIEGO",
        "CHARGERS",
    ),
    "LAR": (
        "LAR",
        "LA RAMS",
        "LOS ANGELES RAMS",
        "ST LOUIS RAMS",
        "ST. LOUIS RAMS",
        "RAMS",
    ),
    "LV": ("LV", "LVR", "LAS VEGAS", "LAS VEGAS RAIDERS", "OAKLAND RAIDERS", "OAK", "RAIDERS"),
    "MIA": ("MIA", "MIAMI", "MIAMI DOLPHINS", "DOLPHINS"),
    "MIN": ("MIN", "MINNESOTA", "MINNESOTA VIKINGS", "VIKINGS"),
    "NE": ("NE", "NWE", "NEW ENGLAND", "NEW ENGLAND PATRIOTS", "PATRIOTS"),
    "NO": ("NO", "NOS", "NEW ORLEANS", "NEW ORLEANS SAINTS", "SAINTS"),
    "NYG": ("NYG", "NY GIANTS", "N.Y. GIANTS", "NEW YORK GIANTS", "GIANTS"),
    "NYJ": ("NYJ", "NY JETS", "N.Y. JETS", "NEW YORK JETS", "JETS"),
    "PHI": ("PHI", "PHILADELPHIA", "PHILADELPHIA EAGLES", "EAGLES"),
    "PIT": ("PIT", "PITTSBURGH", "PITTSBURGH STEELERS", "STEELERS"),
    "SEA": ("SEA", "SEATTLE", "SEATTLE SEAHAWKS", "SEAHAWKS"),
    "SF": ("SF", "SFO", "SAN FRANCISCO", "SAN FRANCISCO 49ERS", "49ERS"),
    "TB": ("TB", "TAM", "TAMPA", "TAMPA BAY", "TAMPA BAY BUCCANEERS", "BUCCANEERS"),
    "TEN": ("TEN", "TENNESSEE", "TENNESSEE TITANS", "TITANS"),
    "WAS": (
        "WAS",
        "WSH",
        "WFT",
        "WASHINGTON",
        "WASHINGTON COMMANDERS",
        "WASHINGTON FOOTBALL TEAM",
        "WASHINGTON REDSKINS",
        "COMMANDERS",
    ),
}


def _sanitize_team_key(name: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", str(name or "").upper())
    return cleaned


TEAM_MAP: Dict[str, str] = {}
for abbr, aliases in _TEAM_ALIASES.items():
    for alias in aliases:
        TEAM_MAP[_sanitize_team_key(alias)] = abbr


def _normalize_team_name(name: str) -> str:
    key = _sanitize_team_key(name)
    if not key:
        return ""
    if key in TEAM_MAP:
        return TEAM_MAP[key]

    # Fallback: try partial matches ordered by length (longest first)
    for alias_key, abbr in sorted(TEAM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if alias_key and alias_key in key:
            return abbr

    # Fallback: if the cleaned value itself looks like a team code
    if key in TEAM_CODES:
        return key
    tail = key[-3:]
    if tail in TEAM_CODES:
        return tail
    return ""


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
    Read the largest table from an HTML payload using BeautifulSoup + pandas.
    """

    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return None

    parsed: List[pd.DataFrame] = []
    for tbl in tables:
        tbl_html = str(tbl)
        df = None
        for flavor in ("lxml", "bs4"):
            try:
                dfs = pd.read_html(StringIO(tbl_html), flavor=flavor)
                if dfs:
                    df = dfs[0]
                    break
            except Exception:
                continue
        if df is not None and not df.empty:
            parsed.append(df)

    if not parsed:
        return None

    scores = [(i, parsed[i].shape[0] * max(1, parsed[i].shape[1])) for i in range(len(parsed))]
    idx = max(scores, key=lambda x: x[1])[0]
    return parsed[idx]

def _slug(s: str) -> str:
    s = str(s or "").strip().lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", "_", s)
    return s


def _normalize_pace_table(df: pd.DataFrame) -> pd.DataFrame:
    pace_df = df.copy()

    raw_cols = {c.strip().upper(): c for c in pace_df.columns}

    candidates = [
        "NEUTRAL DB RATE",
        "NEUTRAL DB RATE LAST 5",
        "NEUTRAL PACE",
        "SITUATION NEUTRAL PACE",
        "NEUTRAL SECS/PLAY",
    ]
    neutral_col = None
    for cand in candidates:
        if cand in raw_cols:
            neutral_col = raw_cols[cand]
            break

    if neutral_col is None:
        raise RuntimeError(
            "[sharpfootball_pull] could not find neutral pace col in pace_df. "
            f"Available cols: {list(pace_df.columns)}"
        )

    rename_map: Dict[str, str] = {}
    for possible_team in ["TEAM", "Team", "team"]:
        if possible_team in pace_df.columns:
            rename_map[possible_team] = "team"
            break

    rename_map[neutral_col] = "neutral_pace"

    last5_candidates = [
        "NEUTRAL DB RATE LAST 5",
        "NEUTRAL PACE LAST 5",
        "LAST 5 NEUTRAL",
        "NEUTRAL SECS/PLAY LAST 5",
    ]
    for last5_cand in last5_candidates:
        if last5_cand in raw_cols:
            rename_map[raw_cols[last5_cand]] = "neutral_pace_last5"
            break

    pace_df = pace_df.rename(columns=rename_map)

    for col in ["neutral_pace", "neutral_pace_last5"]:
        if col in pace_df.columns:
            pace_df[col] = (
                pace_df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            pace_df[col] = pd.to_numeric(pace_df[col], errors="coerce")

    if "neutral_pace" not in pace_df.columns:
        raise RuntimeError(
            "[sharpfootball_pull] pace_df missing neutral_pace after normalization. "
            f"Available cols: {list(pace_df.columns)}"
        )

    if "team" not in pace_df.columns:
        raise RuntimeError(
            "[sharpfootball_pull] pace_df missing team column after normalization. "
            f"Available cols: {list(pace_df.columns)}"
        )

    return pace_df


def _normalize_coverage_scheme_table(df: pd.DataFrame) -> pd.DataFrame:
    coverage_df = df.copy()

    raw_cols = {c.strip().upper(): c for c in coverage_df.columns}

    rename_map: Dict[str, str] = {}
    for possible_team in ["TEAM", "Team", "team"]:
        if possible_team in coverage_df.columns:
            rename_map[possible_team] = "team"
            break

    man_candidates = [
        "MAN %",
        "MAN%",
        "MAN RATE",
        "MAN COVERAGE %",
        "MAN COVERAGE",
        "MAN COVERAGE RATE",
    ]
    man_col = None
    for cand in man_candidates:
        if cand in raw_cols:
            man_col = raw_cols[cand]
            break
    if man_col is None:
        for col in coverage_df.columns:
            if "MAN" in str(col).upper():
                man_col = col
                break
    if man_col is None:
        raise RuntimeError(
            "[sharpfootball_pull] could not find coverage man rate column. "
            f"Available cols: {list(coverage_df.columns)}"
        )
    rename_map[man_col] = "coverage_man_rate"

    zone_candidates = [
        "ZONE %",
        "ZONE%",
        "ZONE RATE",
        "ZONE COVERAGE %",
        "ZONE COVERAGE",
        "ZONE COVERAGE RATE",
    ]
    zone_col = None
    for cand in zone_candidates:
        if cand in raw_cols:
            zone_col = raw_cols[cand]
            break
    if zone_col is None:
        for col in coverage_df.columns:
            if "ZONE" in str(col).upper():
                zone_col = col
                break
    if zone_col is None:
        raise RuntimeError(
            "[sharpfootball_pull] could not find coverage zone rate column. "
            f"Available cols: {list(coverage_df.columns)}"
        )
    rename_map[zone_col] = "coverage_zone_rate"

    coverage_df = coverage_df.rename(columns=rename_map)

    for col in ["coverage_man_rate", "coverage_zone_rate"]:
        if col in coverage_df.columns:
            coverage_df[col] = (
                coverage_df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            coverage_df[col] = pd.to_numeric(coverage_df[col], errors="coerce")

    if "team" not in coverage_df.columns:
        raise RuntimeError(
            "[sharpfootball_pull] coverage_df missing team column after normalization. "
            f"Available cols: {list(coverage_df.columns)}"
        )

    return coverage_df


COLUMN_ALIAS_PATTERNS: Dict[str, Dict[str, Iterable[str]]] = {
    "off_tend": {
        "pass_rate_over_expected": {
            "pass_rate_over_expected",
            "pass_rate_over_expectation",
            "pass_rate_over_exp",
            "proe",
            "pass_rate_vs_expected",
            "pass_rate_over_expected_pct",
            "pass_rate_over_exp_pct",
        },
        "neutral_db_rate": {"neutral_pass_rate"},
        "neutral_db_rate_last_5": {"neutral_pass_rate_last_5"},
    },
    "pace": {
        "neutral_pace": {
            "neutral_pace",
            "neutralpace",
            "neutral_pace_seconds_per_play",
            "neutral_pace_seconds_play",
            "neutral_seconds_per_play",
            "neutral_pace_secondsperplay",
        },
        "seconds_per_play": {"seconds_per_play"},
        "seconds_per_play_last5": {"seconds_per_play_last5", "seconds_per_play_last_5"},
        "plays_per_game": {"plays_per_game"},
    },
    "coverage_scheme": {
        "coverage_man_rate": {
            "man_coverage_rate",
            "man_rate",
            "man_coverage",
            "man_coverage_pct",
            "man_pct",
        },
        "coverage_zone_rate": {
            "zone_coverage_rate",
            "zone_rate",
            "zone_coverage",
            "zone_coverage_pct",
            "zone_pct",
        },
    },
    "coverage_pos": {
        "coverage_man_rate": {"man_coverage_rate", "man_coverage_pct"},
        "coverage_zone_rate": {"zone_coverage_rate", "zone_coverage_pct"},
    },
}


def _rename_expected_cols(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    alias_map = COLUMN_ALIAS_PATTERNS.get(kind, {})
    if not alias_map:
        return df

    out = df.copy()
    for target, aliases in alias_map.items():
        for col in list(out.columns):
            if col in ("team", "team_raw"):
                continue
            slugged = _slug(col)
            if slugged == target:
                continue
            if slugged in aliases or any(slugged.startswith(a) for a in aliases):
                if target in out.columns:
                    out[target] = out[target].where(out[target].notna(), out[col])
                    out.drop(columns=[col], inplace=True)
                else:
                    out = out.rename(columns={col: target})
                break
    return out

def _normalize_team_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c: _slug(c) for c in df.columns}
    df = df.rename(columns=cols)

    team_col = None
    for cand in df.columns:
        if any(tok in cand for tok in ("team", "defense", "offense", "club", "squad")):
            team_col = cand
            break
    if team_col is None:
        team_col = df.columns[0]

    if team_col != "team_raw":
        df = df.rename(columns={team_col: "team_raw"})

    df["team_raw"] = df["team_raw"].astype(str).str.strip()
    df = df[~df["team_raw"].str.contains("Rank|Ranking|Ranks|NFL", case=False, na=False)]
    df["team"] = df["team_raw"].apply(_normalize_team_name)
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df = df[df["team"] != ""]
    df = df.drop_duplicates(subset=["team"])
    return df


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("team", "team_raw"):
            continue
        ser = (
            df[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(r"\((?:[^)(]+|\([^)(]*\))*\)", "", regex=True)
            .str.strip()
        )
        extracted = ser.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        df[c] = pd.to_numeric(extracted, errors="coerce")
    return df

def _save_csv(df: Optional[pd.DataFrame], out_path: str) -> int:
    if df is None or df.empty:
        return 0
    df.to_csv(out_path, index=False)
    return len(df)

def _prepare_piece_for_merge(kind: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    out = df.copy()
    if "team" not in out.columns:
        return None

    out["team"] = out["team"].astype(str).str.upper().str.strip()
    out = out[out["team"].isin(TEAM_CODES)]
    if out.empty:
        return None

    if "team_raw" in out.columns:
        out = out.rename(columns={"team_raw": f"team_raw_{kind}"})

    out = out.loc[:, ~out.columns.duplicated()]
    out = out.drop_duplicates(subset=["team"])
    return out

def _pull_one(kind: str, url: str, season: int) -> Tuple[str, int, Optional[pd.DataFrame]]:
    html = _fetch_html(url, season, kind)
    df = _read_single_table_from_html(html or "")
    if df is None or df.empty:
        return kind, 0, None

    if kind == "pace":
        df = _normalize_pace_table(df)
    elif kind == "coverage_scheme":
        df = _normalize_coverage_scheme_table(df)

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
    prepared: Dict[str, pd.DataFrame] = {}
    for kind, df in pieces.items():
        prepped = _prepare_piece_for_merge(kind, df)
        if prepped is not None:
            prepared[kind] = prepped

    if not prepared:
        return 0

    base = pd.DataFrame({"team": sorted(TEAM_CODES)})
    for kind in ("def_tend", "off_tend", "pace", "coverage_pos", "coverage_scheme", "dl", "ol"):
        df = prepared.get(kind)
        if df is None or df.empty:
            continue
        base = base.merge(df, on="team", how="left")

    raw_cols = [c for c in base.columns if c.startswith("team_raw_")]
    if raw_cols:
        base["team_raw"] = base[raw_cols].bfill(axis=1).iloc[:, 0]
        base.drop(columns=raw_cols, inplace=True)

    base["team_abbr"] = base["team"]

    pace_df = prepared.get("pace")
    if pace_df is not None and "neutral_pace" in pace_df.columns:
        base = base.merge(pace_df[["team", "neutral_pace"]], on="team", how="left", suffixes=("", "_pace"))
        if "neutral_pace_pace" in base.columns:
            base["neutral_pace"] = base["neutral_pace"].combine_first(base["neutral_pace_pace"])
            base.drop(columns=["neutral_pace_pace"], inplace=True)

    cov_df = prepared.get("coverage_scheme")
    if cov_df is not None:
        cov_cols = [c for c in ("coverage_man_rate", "coverage_zone_rate") if c in cov_df.columns]
        if cov_cols:
            base = base.merge(cov_df[["team"] + cov_cols], on="team", how="left", suffixes=("", "_cov"))
            for c in cov_cols:
                aux = f"{c}_cov"
                if aux in base.columns:
                    base[c] = base[c].combine_first(base[aux])
                    base.drop(columns=[aux], inplace=True)

    merged = base

    # === FINAL CANONICAL NORMALIZATION BEFORE VALIDATION ===

    rename_final = {}

    # 1. neutral pace
    # Our downstream code expects a column called 'neutral_pace'.
    # Sharp gives us neutral tempo as something like 'neutralpaces1' or 'secplay'
    # (seconds per play in neutral script; lower = faster).
    # Rule:
    #   - if 'neutralpaces1' exists, that's our neutral_pace
    #   - else if 'secplay' exists, that's our neutral_pace
    if "neutral_pace" not in merged.columns:
        if "neutralpaces1" in merged.columns:
            merged["neutral_pace"] = merged["neutralpaces1"]
        elif "secplay" in merged.columns:
            merged["neutral_pace"] = merged["secplay"]

    # Coerce to numeric (strip weird strings like "29.1s")
    if "neutral_pace" in merged.columns:
        merged["neutral_pace"] = (
            merged["neutral_pace"]
            .astype(str)
            .str.replace("s", "", regex=False)
            .str.replace("sec", "", regex=False)
            .str.replace("seconds", "", regex=False)
            .str.strip()
        )
        merged["neutral_pace"] = pd.to_numeric(merged["neutral_pace"], errors="coerce")

    # 2. coverage columns
    # Downstream expects coverage_man_rate and coverage_zone_rate.
    # Our scrape produced 'coveragemanrate' and 'coveragezonerate'.
    if "coveragemanrate" in merged.columns and "coverage_man_rate" not in merged.columns:
        merged = merged.rename(columns={"coveragemanrate": "coverage_man_rate"})

    if "coveragezonerate" in merged.columns and "coverage_zone_rate" not in merged.columns:
        merged = merged.rename(columns={"coveragezonerate": "coverage_zone_rate"})

    # Normalize those coverage rates to numeric (drop '%' etc.)
    for col in ["coverage_man_rate", "coverage_zone_rate"]:
        if col in merged.columns:
            merged[col] = (
                merged[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # === FINAL CANONICAL NORMALIZATION ===

    # 1️⃣ Neutral pace normalization
    if "neutral_pace" not in merged.columns:
        if "neutral_pace_score" in merged.columns:
            merged["neutral_pace"] = merged["neutral_pace_score"]
        elif "secplay" in merged.columns:
            merged["neutral_pace"] = merged["secplay"]
        else:
            print("[sharpfootball_pull] WARNING: could not find neutral_pace_score or secplay")

    # 2️⃣ Convert to numeric
    merged["neutral_pace"] = pd.to_numeric(merged["neutral_pace"], errors="coerce")

    # 3️⃣ Coverage normalization
    if "coveragemanrate" in merged.columns and "coverage_man_rate" not in merged.columns:
        merged = merged.rename(columns={"coveragemanrate": "coverage_man_rate"})

    if "coveragezonerate" in merged.columns and "coverage_zone_rate" not in merged.columns:
        merged = merged.rename(columns={"coveragezonerate": "coverage_zone_rate"})

    for col in ["coverage_man_rate", "coverage_zone_rate"]:
        if col in merged.columns:
            merged[col] = (
                merged[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # 4️⃣ Required columns guard
    required_cols = ["neutral_pace", "coverage_man_rate", "coverage_zone_rate"]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise RuntimeError(
            f"[sharpfootball_pull] missing required columns: {missing}\nAvailable: {list(merged.columns)}"
        )

    # === END CANONICAL NORMALIZATION ===

    # === REQUIRED COLUMNS CHECK (keep or update existing logic) ===
    required_cols = [
        "neutral_pace",
        "coverage_man_rate",
        "coverage_zone_rate",
    ]

    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise RuntimeError(
            "[sharpfootball_pull] still missing required column(s) in merged team form. "
            f"Missing={missing} Available cols={sorted(list(merged.columns))}"
        )

    # === END FINAL CANONICAL NORMALIZATION ===

    base = merged

    required_cols = ["neutral_pace", "coverage_man_rate", "coverage_zone_rate"]
    for col in required_cols:
        if col not in base.columns:
            raise RuntimeError(
                f"[sharpfootball_pull] missing required column {col} in merged team form. "
                f"Available cols: {sorted(list(base.columns))}"
            )
        if base[col].isna().all():
            raise RuntimeError(f"[sharpfootball_pull] missing or empty required col {col}")

    out_path = os.path.join(DATA_DIR, "sharp_team_form.csv")
    base.to_csv(out_path, index=False)
    print(
        "[sharpfootball_pull] wrote data/sharp_team_form.csv with cols:",
        sorted(list(base.columns)),
    )
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
