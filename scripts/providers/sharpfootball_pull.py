#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp Football provider (public pages) → fallback enrichers for team form.

Writes (seasoned file names):
  data/sharp_team_form.csv                    # merged superset used by make_team_form fallbacks
  data/sharp_def_tend_{season}.csv            # defensive tendencies (blitz/sub)
  data/sharp_off_tend_{season}.csv            # offensive tendencies (motion/PA/shotgun/no-huddle)
  data/sharp_coverage_pos_{season}.csv        # YPT allowed by position/align
  data/sharp_dl_{season}.csv                  # DL pressure/YBC/stuff
  data/sharp_pace_{season}.csv                # seconds/play, plays/game

Defensive hardening added:
- multi-strategy parsing (pd.read_html on URL/HTML + BeautifulSoup table iter)
- seasonal URL attempts (?season=YYYY, ?year=YYYY) on each page
- stronger UA + retries, never crash; write schema-complete CSV even on empties
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import random
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

DATA_DIR = "data"

TEAM_NAME_TO_ABBR = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL",
    "BUFFALO BILLS":"BUF","CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI",
    "CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE","DALLAS COWBOYS":"DAL",
    "DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX",
    "KANSAS CITY CHIEFS":"KC","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR",
    "LAS VEGAS RAIDERS":"LV","MIAMI DOLPHINS":"MIA","MINNESOTA VIKINGS":"MIN",
    "NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT",
    "SEATTLE SEAHAWKS":"SEA","SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB",
    "TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
}

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]

HEADERS = {
    "User-Agent": random.choice(UAS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sharpfootballanalysis.com/stats-nfl/",
    "Cache-Control": "no-cache",
}

URLS = {
    "def_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-tendencies/",
    "off_tend": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-offensive-tendencies-stats/",
    "coverage_pos": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-stats-by-position/",
    "dl": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-defensive-line-stats/",
    "pace": "https://www.sharpfootballanalysis.com/stats-nfl/nfl-team-pace-stats/",
}

def _maybe_season_url(url: str, season: int) -> list[str]:
    cands = [url]
    for q in (f"season={season}", f"year={season}", f"Season={season}"):
        sep = "&" if "?" in url else "?"
        cands.append(f"{url}{sep}{q}")
    return cands

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def _dump_html(name: str, html: str):
    try:
        _safe_mkdir(DATA_DIR)
        with open(os.path.join(DATA_DIR, f"_sharp_dump_{name}.html"), "w", encoding="utf-8") as f:
            f.write(html or "")
    except Exception:
        pass

def _read_tables_url(url: str) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    # Strategy A: let pandas fetch directly
    try:
        dfs = pd.read_html(url)
        if dfs: return dfs
    except Exception:
        pass
    # Strategy B: requests + pd.read_html on full HTML
    sess = _session()
    html = ""
    try:
        r = sess.get(url, timeout=25)
        html = r.text
        if r.status_code != 200 or "<table" not in html.lower():
            _dump_html("no_table_" + url.rstrip("/").split("/")[-1], html)
        else:
            try:
                dfs = pd.read_html(html)
                if dfs: return dfs
            except Exception:
                pass
    except Exception:
        pass
    # Strategy C: BeautifulSoup per-table extraction
    try:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for t in soup.find_all("table"):
            try:
                d = pd.read_html(str(t))
                if d: out.extend(d)
            except Exception:
                continue
        if out:
            return out
        else:
            _dump_html("soup_zero_" + url.rstrip("/").split("/")[-1], html)
    except Exception:
        pass
    return []

def _first_team_table(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    best, best_score = None, -1
    keys = set(TEAM_NAME_TO_ABBR.keys())
    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cols = [c.lower() for c in df.columns]
        score = 0
        if cols and any(k in cols[0] for k in ["team","defense","offense"]):
            score += 2
        try:
            first = df.iloc[:, 0].astype(str).str.upper()
            hit = first.isin(keys).mean()
            if hit > 0.15:
                score += 3
            else:
                norm = (first.str.replace(r"[^A-Z\s]", "", regex=True)
                              .str.replace(r"\s+", " ", regex=True).str.strip())
                if norm.isin(keys).mean() > 0.15:
                    score += 2
        except Exception:
            pass
        if score > best_score:
            best, best_score = df, score
    return best.copy() if best is not None else pd.DataFrame()

def _team_abbr_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col = None
    for c in df.columns:
        cl = c.lower()
        if "team" in cl or "defense" in cl or "offense" in cl:
            col = c; break
    if col is None:
        col = df.columns[0]
    df = df.rename(columns={col: "team_name"})
    df["team_name"] = df["team_name"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    df["team"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    df = df[df["team"].notna()].copy()
    return df

def _clean_rate_series(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.replace(",", "", regex=False).str.strip()
    pct = v.str.endswith("%")
    num = pd.to_numeric(v.str.rstrip("%"), errors="coerce")
    out = np.where(pct, num/100.0, num)
    return pd.Series(out, index=s.index, dtype="float64")

def _try_pick(df: pd.DataFrame, contains: List[str]) -> pd.Series:
    for c in df.columns:
        cl = c.lower()
        if all(key in cl for key in contains):
            return _clean_rate_series(df[c])
    return pd.Series(dtype="float64")

def parse_def_tend(url: str) -> pd.DataFrame:
    dfs = _read_tables_url(url)
    base = _team_abbr_col(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","blitz_rate","sub_package_rate","light_box_rate","heavy_box_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["blitz_rate"] = _try_pick(base, ["blitz"])
    sub = _try_pick(base, ["sub"])
    if sub.empty: 
        sub = _try_pick(base, ["nickel"])
    out["sub_package_rate"] = sub
    heavy = pd.Series(dtype="float64"); light = pd.Series(dtype="float64")
    for c in base.columns:
        cl = c.lower()
        if "box" in cl and (("8" in cl) or ("stack" in cl) or ("heavy" in cl)):
            heavy = _clean_rate_series(base[c])
        if "box" in cl and ("light" in cl):
            light = _clean_rate_series(base[c])
    out["heavy_box_rate"] = heavy
    out["light_box_rate"] = light
    return out.drop_duplicates(subset=["team"])

def parse_off_tend(url: str) -> pd.DataFrame:
    dfs = _read_tables_url(url)
    base = _team_abbr_col(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","motion_rate","play_action_rate","shotgun_rate","no_huddle_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["motion_rate"]      = _try_pick(base, ["motion"])
    out["play_action_rate"] = _try_pick(base, ["play","action"])
    out["shotgun_rate"]     = _try_pick(base, ["shotgun"])
    out["no_huddle_rate"]   = _try_pick(base, ["no","huddle"])
    return out.drop_duplicates(subset=["team"])

def parse_coverage_pos(url: str) -> pd.DataFrame:
    dfs = _read_tables_url(url)
    base = _team_abbr_col(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=[
            "team","ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed",
            "outside_ypt_allowed","slot_ypt_allowed"
        ])
    out = pd.DataFrame({"team": base["team"]})
    def pick_any(*keys): return _try_pick(base, list(keys))
    out["ypt_allowed"]         = pick_any("yards","per","target")
    out["wr_ypt_allowed"]      = pick_any("wr","yards","target")
    out["te_ypt_allowed"]      = pick_any("te","yards","target")
    out["rb_ypt_allowed"]      = pick_any("rb","yards","target")
    out["outside_ypt_allowed"] = pick_any("outside","yards","target")
    out["slot_ypt_allowed"]    = pick_any("slot","yards","target")
    return out.drop_duplicates(subset=["team"])

def parse_dl(url: str) -> pd.DataFrame:
    dfs = _read_tables_url(url)
    base = _team_abbr_col(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["dl_pressure_rate"]          = _try_pick(base, ["pressure","rate"])
    out["dl_no_blitz_pressure_rate"] = _try_pick(base, ["no","blitz","pressure"])
    ybc = None
    for c in base.columns:
        cl = c.lower()
        if "before" in cl and "contact" in cl and ("rush" in cl or "carry" in cl):
            ybc = pd.to_numeric(base[c], errors="coerce"); break
    out["dl_ybc_per_rush"] = ybc if ybc is not None else pd.Series(dtype="float64")
    out["dl_stuff_rate"]   = _try_pick(base, ["stuff"])
    return out.drop_duplicates(subset=["team"])

def parse_pace(url: str) -> pd.DataFrame:
    dfs = _read_tables_url(url)
    base = _team_abbr_col(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","seconds_per_play","plays_per_game"])
    def pick_num(keys: List[str]) -> pd.Series:
        for c in base.columns:
            cl = c.lower()
            if all(k in cl for k in keys):
                return pd.to_numeric(base[c].astype(str).str.replace(",","",regex=False), errors="coerce")
        return pd.Series(dtype="float64")
    out = pd.DataFrame({"team": base["team"]})
    out["seconds_per_play"] = pick_num(["seconds","play"])
    out["plays_per_game"]   = pick_num(["plays","game"])
    return out.drop_duplicates(subset=["team"])

def merge_non_destructive(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty: 
        return left
    r = right.copy()
    if "team" not in r.columns: 
        return left
    keep = [c for c in r.columns if c != "team"]
    out = left.merge(r, on="team", how="left", suffixes=("","_ext"))
    for c in keep:
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", nargs="?", type=int, help="Season (e.g., 2025)")
    ap.add_argument("--season", dest="season_flag", type=int, help="Season (e.g., 2025)")
    args = ap.parse_args()
    season = args.season_flag if args.season_flag is not None else args.season
    if season is None:
        ap.error("Season is required (pass 2025 or --season 2025)")
    season = int(season)

    _safe_mkdir(DATA_DIR)

    def attempt(parse_fn, base_url):
        # try base + seasonized candidates
        for u in _maybe_season_url(base_url, season):
            df = parse_fn(u)
            if df is not None and not df.empty:
                return df
        return pd.DataFrame()

    def_t = attempt(parse_def_tend, URLS["def_tend"])
    off_t = attempt(parse_off_tend, URLS["off_tend"])
    cov_p = attempt(parse_coverage_pos, URLS["coverage_pos"])
    dl    = attempt(parse_dl, URLS["dl"])
    pace  = attempt(parse_pace, URLS["pace"])

    # Write individual CSVs (seasoned)
    if not def_t.empty:  def_t.to_csv(os.path.join(DATA_DIR, f"sharp_def_tend_{season}.csv"), index=False)
    if not off_t.empty:  off_t.to_csv(os.path.join(DATA_DIR, f"sharp_off_tend_{season}.csv"), index=False)
    if not cov_p.empty:  cov_p.to_csv(os.path.join(DATA_DIR, f"sharp_coverage_pos_{season}.csv"), index=False)
    if not dl.empty:     dl.to_csv(os.path.join(DATA_DIR, f"sharp_dl_{season}.csv"), index=False)
    if not pace.empty:   pace.to_csv(os.path.join(DATA_DIR, f"sharp_pace_{season}.csv"), index=False)

    # Merge all parsed team rows
    teams = pd.Series(pd.concat([
        def_t.get("team", pd.Series(dtype=object)),
        off_t.get("team", pd.Series(dtype=object)),
        cov_p.get("team", pd.Series(dtype=object)),
        dl.get("team", pd.Series(dtype=object)),
        pace.get("team", pd.Series(dtype=object)),
    ], ignore_index=True).dropna().unique(), name="team")
    sharp = teams.to_frame()
    for part in [def_t, off_t, cov_p, dl, pace]:
        sharp = merge_non_destructive(sharp, part)

    # Ensure schema columns exist
    for c in [
        "blitz_rate","sub_package_rate",
        "light_box_rate","heavy_box_rate",
        "motion_rate","play_action_rate","shotgun_rate","no_huddle_rate",
        "ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed","outside_ypt_allowed","slot_ypt_allowed",
        "dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate",
        "seconds_per_play","plays_per_game",
    ]:
        if c not in sharp.columns: sharp[c] = np.nan

    sharp.to_csv(os.path.join(DATA_DIR, "sharp_team_form.csv"), index=False)
    print(f"[sharp] wrote {len(sharp)} rows → data/sharp_team_form.csv")
    return 0

if __name__ == "__main__":
    sys.exit(main())
