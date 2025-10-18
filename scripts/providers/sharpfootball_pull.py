#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp Football Analysis (SFA) multi-source provider.

Fetches public team tables from SharpFootballAnalysis.com and writes:
  - data/sharp_def_tendencies_<season>.csv
  - data/sharp_off_tendencies_<season>.csv
  - data/sharp_coverage_pos_<season>.csv
  - data/sharp_dl_<season>.csv
  - data/sharp_ol_<season>.csv
  - data/sharp_off_metrics_<season>.csv
  - data/sharp_def_metrics_<season>.csv
  - data/sharp_pace_<season>.csv
  - data/sharp_team_form.csv      # normalized, merge-ready fallback

Only uses public static tables (parsed via pandas.read_html).
"""

from __future__ import annotations
import argparse, os, sys, re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"

URLS = {
    # Defensive tendencies: blitz %, light/heavy box % (CONFIRMED)
    "def_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-tendencies/",
    # Offensive tendencies: motion %, play-action %, AY/Att, shotgun %, no-huddle % (CONFIRMED)
    "off_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-tendencies-stats/",
    # Positional coverage YPT (CONFIRMED)
    "cov_pos":  "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-stats-by-position/",
    # Defensive line statistics: pressure %, no-blitz pressure %, YBC/rush, stuff rate % (CONFIRMED)
    "dl":       "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-line-stats/",
    # Offensive line statistics (table exists; columns can vary by season)
    "ol":       "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-line-stats/",
    # Team pace: seconds/play, plays/game (page exists; table structure can vary)
    "pace":     "https://www.sharpfootballanalysis.com/stats-nfl/nfl-team-pace-stats/",
    # Aggregate offense/defense metrics (EPA/yds etc.; tables exist but headers vary; we export raw-normalized)
    "off_metrics": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-stats/",
    "def_metrics": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-stats/",
}

# Canonical team mapping
CANON = {
    "OAK":"LV","SD":"LAC","STL":"LAR","JAC":"JAX","WSH":"WAS","LA":"LAR",
    "LAS":"LV","LOS ANGELES":"LAR","LOS ANGELES RAMS":"LAR","LOS ANGELES CHARGERS":"LAC",
    "WASHINGTON":"WAS","NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ"
}
VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

NAME2ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","LAS VEGAS RAIDERS":"LV","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO",
    "NEW YORK GIANTS":"NYG","NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT",
    "SEATTLE SEAHAWKS":"SEA","SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN",
    "WASHINGTON COMMANDERS":"WAS"
}

def canon_team(x: str) -> str:
    if x is None: return ""
    s = str(x).strip().upper()
    s = NAME2ABBR.get(s, s)
    s = CANON.get(s, s)
    return s if s in VALID else ""

def _safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def _pct_to_float(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    s = s.replace("%","")
    try:
        return float(s) / 100.0
    except Exception:
        return pd.to_numeric(v, errors="coerce") / 100.0

def _fetch_tables(url: str) -> List[pd.DataFrame]:
    # Pull HTML and let pandas parse tables
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    try:
        tables = pd.read_html(r.text)
        return tables or []
    except ValueError:
        return []

def _first_table_with_keywords(tables: List[pd.DataFrame], required_any: List[str], required_all: List[str] = None) -> pd.DataFrame:
    if required_all is None: required_all = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        cols_l = [c.lower() for c in cols]
        any_hit = any(any(k.lower() in c for c in cols) for k in required_any) if required_any else True
        all_hit = all(any(k.lower() in c for c in cols_l) for k in required_all) if required_all else True
        if any_hit and all_hit:
            return t
    return pd.DataFrame()

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _extract_team_col(df: pd.DataFrame) -> pd.DataFrame:
    if "team" in df.columns: return df
    for cand in ["teams","club","name","squad"]:
        if cand in df.columns:
            return df.rename(columns={cand:"team"})
    # sometimes first column is team
    first = df.columns[0]
    return df.rename(columns={first:"team"})

def _clean_team_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df["team"] = df["team"].astype(str).str.strip().str.upper().map(canon_team)
    return df[df["team"] != ""].drop_duplicates(subset=["team"])

def _maybe_pct(df: pd.DataFrame, cols_like: List[str]) -> pd.DataFrame:
    for c in list(df.columns):
        for like in cols_like:
            if like in c and df[c].dtype == object:
                df[c] = df[c].apply(_pct_to_float)
    return df

def parse_def_tend() -> pd.DataFrame:
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["def_tend"]),
        required_any=["Blitz", "Light", "Heavy", "Sub"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    # pick expected columns and rename to our schema
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "blitz" in cl: ren[c]="blitz_rate"
        elif "light" in cl and "box" in cl: ren[c]="light_box_rate"
        elif "heavy" in cl and "box" in cl: ren[c]="heavy_box_rate"
        elif "sub" in cl: ren[c]="sub_package_rate"
    df = df.rename(columns=ren)
    for k in ["blitz_rate","light_box_rate","heavy_box_rate","sub_package_rate"]:
        if k not in df: df[k] = np.nan
    keep = ["team","blitz_rate","light_box_rate","heavy_box_rate","sub_package_rate"]
    df = df[keep]
    df = _maybe_pct(df, ["rate"])
    return _clean_team_and_filter(df)

def parse_off_tend() -> pd.DataFrame:
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["off_tend"]),
        required_any=["Motion", "Play", "Air", "Shotgun", "Huddle"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "motion" in cl: ren[c]="motion_rate"
        elif "play_action" in cl or "play-action" in cl: ren[c]="play_action_rate"
        elif "air" in cl and "att" in cl: ren[c]="air_yards_per_att"
        elif "shotgun" in cl: ren[c]="shotgun_rate"
        elif "huddle" in cl: ren[c]="no_huddle_rate"
    df = df.rename(columns=ren)
    for k in ["motion_rate","play_action_rate","air_yards_per_att","shotgun_rate","no_huddle_rate"]:
        if k not in df: df[k] = np.nan
    keep = ["team","motion_rate","play_action_rate","air_yards_per_att","shotgun_rate","no_huddle_rate"]
    df = df[keep]
    df = _maybe_pct(df, ["rate"])
    # air_yards_per_att is numeric yards; leave as-is
    return _clean_team_and_filter(df)

def parse_cov_pos() -> pd.DataFrame:
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["cov_pos"]),
        required_any=["YPT"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "ypt" in cl and "allowed" in cl and "wr" in cl: ren[c]="wr_ypt_allowed"
        elif "ypt" in cl and "allowed" in cl and "te" in cl: ren[c]="te_ypt_allowed"
        elif "ypt" in cl and "allowed" in cl and "rb" in cl: ren[c]="rb_ypt_allowed"
        elif "outside" in cl and "ypt" in cl: ren[c]="outside_ypt_allowed"
        elif "slot" in cl and "ypt" in cl: ren[c]="slot_ypt_allowed"
        elif "ypt" in cl and "allowed" in cl and "wr" not in cl and "te" not in cl and "rb" not in cl:
            ren[c]="ypt_allowed"
    df = df.rename(columns=ren)
    for k in ["ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed","outside_ypt_allowed","slot_ypt_allowed"]:
        if k not in df: df[k] = np.nan
    keep = ["team","ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed","outside_ypt_allowed","slot_ypt_allowed"]
    df = df[keep]
    return _clean_team_and_filter(df)

def parse_dl() -> pd.DataFrame:
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["dl"]),
        required_any=["Pressure", "Yards Before Contact", "Stuff"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "no_blitz" in cl or ("no" in cl and "blitz" in cl): ren[c]="dl_no_blitz_pressure_rate"
        elif "pressure" in cl: ren[c]="dl_pressure_rate"
        elif "yards_before_contact" in cl or ("before" in cl and "contact" in cl): ren[c]="dl_ybc_per_rush"
        elif "stuff" in cl: ren[c]="dl_stuff_rate"
    df = df.rename(columns=ren)
    for k in ["dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate"]:
        if k not in df: df[k] = np.nan
    keep = ["team","dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate"]
    df = df[keep]
    df = _maybe_pct(df, ["pressure", "stuff"])
    return _clean_team_and_filter(df)

def parse_ol() -> pd.DataFrame:
    # Columns on this page vary; keep it flexible.
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["ol"]),
        required_any=["Pressure", "Sack", "Contact", "Stuff", "Hurry", "Hit"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    # Map a few common metrics if present
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "pressure" in cl and "allowed" in cl: ren[c]="ol_pressure_allowed_rate"
        elif "sack" in cl and ("rate" in cl or "allowed" in cl): ren[c]="ol_sack_rate"
        elif "yards_before_contact" in cl or ("before" in cl and "contact" in cl): ren[c]="ol_ybc_per_rush"
        elif "stuff" in cl: ren[c]="ol_stuff_rate"
    df = df.rename(columns=ren)
    # keep whatever we could map
    keep = [c for c in ["team","ol_pressure_allowed_rate","ol_sack_rate","ol_ybc_per_rush","ol_stuff_rate"] if c in df.columns]
    if not keep:
        # fall back to just team + any numeric columns
        num_cols = [c for c in df.columns if c != "team" and pd.api.types.is_numeric_dtype(df[c])]
        keep = ["team"] + num_cols
    df = df[keep]
    df = _maybe_pct(df, ["rate","pressure","sack","stuff"])
    return _clean_team_and_filter(df)

def parse_pace() -> pd.DataFrame:
    tb = _first_table_with_keywords(
        _fetch_tables(URLS["pace"]),
        required_any=["Seconds", "Plays"]
    )
    df = _normalize(tb)
    df = _extract_team_col(df)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "team" in cl: ren[c]="team"
        elif "seconds" in cl: ren[c]="seconds_per_play"
        elif "plays" in cl and "game" in cl: ren[c]="plays_per_game"
    df = df.rename(columns=ren)
    for k in ["seconds_per_play","plays_per_game"]:
        if k not in df: df[k] = np.nan
    keep = ["team","seconds_per_play","plays_per_game"]
    df = df[keep]
    return _clean_team_and_filter(df)

def parse_off_metrics() -> pd.DataFrame:
    tb = _first_table_with_keywords(_fetch_tables(URLS["off_metrics"]), required_any=["EPA", "Yards", "Drive"])
    df = _normalize(tb)
    df = _extract_team_col(df)
    return _clean_team_and_filter(df)

def parse_def_metrics() -> pd.DataFrame:
    tb = _first_table_with_keywords(_fetch_tables(URLS["def_metrics"]), required_any=["EPA", "Yards", "Drive"])
    df = _normalize(tb)
    df = _extract_team_col(df)
    return _clean_team_and_filter(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int, help="Season (e.g., 2025). Stored in filenames for provenance.")
    args = ap.parse_args()
    season = int(args.season)

    _safe_mkdir(DATA_DIR)

    # Parse each page
    def_tend = parse_def_tend()
    off_tend = parse_off_tend()
    cov_pos  = parse_cov_pos()
    dl       = parse_dl()
    ol       = parse_ol()
    pace     = parse_pace()
    off_m    = parse_off_metrics()
    def_m    = parse_def_metrics()

    # Write raw-ish normalized dumps per season
    def_tend.to_csv(os.path.join(DATA_DIR, f"sharp_def_tendencies_{season}.csv"), index=False)
    off_tend.to_csv(os.path.join(DATA_DIR, f"sharp_off_tendencies_{season}.csv"), index=False)
    cov_pos.to_csv(os.path.join(DATA_DIR,  f"sharp_coverage_pos_{season}.csv"), index=False)
    dl.to_csv(os.path.join(DATA_DIR,       f"sharp_dl_{season}.csv"), index=False)
    ol.to_csv(os.path.join(DATA_DIR,       f"sharp_ol_{season}.csv"), index=False)
    pace.to_csv(os.path.join(DATA_DIR,     f"sharp_pace_{season}.csv"), index=False)
    off_m.to_csv(os.path.join(DATA_DIR,    f"sharp_off_metrics_{season}.csv"), index=False)
    def_m.to_csv(os.path.join(DATA_DIR,    f"sharp_def_metrics_{season}.csv"), index=False)

    # Build a thin, merge-ready team_form fallback
    # Start from defensive tendencies (gives us team + box/blitz/sub)
    tf = def_tend.copy()
    if tf.empty:
        tf = pd.DataFrame(columns=["team"])

    def _safe_merge(a, b):
        if b is None or b.empty: return a
        return a.merge(b, on="team", how="left")

    tf = _safe_merge(tf, off_tend)
    tf = _safe_merge(tf, cov_pos)
    tf = _safe_merge(tf, dl)
    tf = _safe_merge(tf, pace)

    # Column order (only those that exist will be written)
    wanted = [
        "team",
        "light_box_rate","heavy_box_rate","blitz_rate","sub_package_rate",
        "motion_rate","play_action_rate","air_yards_per_att","shotgun_rate","no_huddle_rate",
        "ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed","outside_ypt_allowed","slot_ypt_allowed",
        "dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate",
        "seconds_per_play","plays_per_game"
    ]
    cols = [c for c in wanted if c in tf.columns]
    if "team" not in cols: cols = ["team"] + [c for c in cols if c != "team"]
    tf[cols].to_csv(os.path.join(DATA_DIR, "sharp_team_form.csv"), index=False)

    print(f"[sharpfootball_pull] wrote fallbacks: {len(tf)} teams → data/sharp_team_form.csv")
    return 0

if __name__ == "__main__":
    sys.exit(main())
