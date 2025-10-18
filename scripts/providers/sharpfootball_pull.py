#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp Football provider (public pages) → fallback enrichers for team form.

Writes:
  data/sharp_team_form.csv
  data/sharp_def_tend_{season}.csv
  data/sharp_off_tend_{season}.csv
  data/sharp_coverage_pos_{season}.csv
  data/sharp_dl_{season}.csv
  data/sharp_pace_{season}.csv
"""

from __future__ import annotations
import argparse, os, sys, time
from typing import List
import numpy as np
import pandas as pd
import requests

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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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

def _safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def _fetch_html(url: str, retries: int = 3, timeout: int = 25) -> str:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and "<table" in r.text.lower():
                return r.text
            last = f"status={r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(1.2 * (i+1))
    print(f"[sharp] fetch failed {url}: {last}", file=sys.stderr)
    return ""

def _read_tables(html: str) -> List[pd.DataFrame]:
    if not html: return []
    try:
        dfs = pd.read_html(html)
        out = []
        for df in dfs:
            if isinstance(df, pd.DataFrame) and df.shape[1] > 0:
                df.columns = [str(c).strip() for c in df.columns]
                out.append(df)
        return out
    except Exception as e:
        print(f"[sharp] read_html failed: {e}", file=sys.stderr)
        return []

def _first_team_table(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs: return pd.DataFrame()
    best = None; score_best = -1
    for df in dfs:
        cols = [c.lower() for c in df.columns]
        score = 0
        if any("team" in c or "defense" in c or "offense" in c for c in cols): score += 2
        if df.shape[1] and df.iloc[:,0].astype(str).str.contains(r"[A-Za-z]{3,}", regex=True).mean() > 0.7: score += 1
        if score > score_best: score_best, best = score, df
    return best.copy() if score_best >= 0 else pd.DataFrame()

def _team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    col = None
    for c in df.columns:
        cl = c.lower()
        if "team" in cl or "defense" in cl or "offense" in cl:
            col = c; break
    if col is None: col = df.columns[0]
    df = df.rename(columns={col: "team_name"})
    df["team_name"] = (df["team_name"].astype(str)
                       .str.upper()
                       .str.replace(r"\s+", " ", regex=True)
                       .str.strip())
    df["team"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    return df[df["team"].notna()].copy()

def _as_rate(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.replace(",", "", regex=False).str.strip()
    pct = v.str.endswith("%")
    num = pd.to_numeric(v.str.rstrip("%"), errors="coerce")
    return pd.Series(np.where(pct, num/100.0, num), index=s.index, dtype="float64")

def _pick(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keys):
            return _as_rate(df[c])
    return pd.Series(dtype="float64")

# ------------------ Parsers ------------------

def parse_def_tend() -> pd.DataFrame:
    html = _fetch_html(URLS["def_tend"]); dfs = _read_tables(html)
    base = _team_abbr(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=[
            "team","blitz_rate","sub_package_rate","light_box_rate","heavy_box_rate"
        ])
    out = pd.DataFrame({"team": base["team"]})
    out["blitz_rate"] = _pick(base, ["blitz"])

    sub = _pick(base, ["sub"])
    if sub.empty: sub = _pick(base, ["nickel"])
    out["sub_package_rate"] = sub

    # Box rates from Sharp columns like "Stacked box (8+)", "8+ in box", "Heavy box", "Light box"
    heavy = pd.Series(dtype="float64")
    for c in base.columns:
        cl = c.lower()
        if "box" in cl and (("8" in cl) or ("stack" in cl) or ("heavy" in cl)):
            heavy = _as_rate(base[c]); 
            if heavy.notna().any(): break
    out["heavy_box_rate"] = heavy
    out["light_box_rate"] = _pick(base, ["light","box"])

    return out.drop_duplicates(subset=["team"])

def parse_off_tend() -> pd.DataFrame:
    html = _fetch_html(URLS["off_tend"]); dfs = _read_tables(html)
    base = _team_abbr(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","motion_rate","play_action_rate","shotgun_rate","no_huddle_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["motion_rate"]      = _pick(base, ["motion"])
    out["play_action_rate"] = _pick(base, ["play","action"])
    out["shotgun_rate"]     = _pick(base, ["shotgun"])
    out["no_huddle_rate"]   = _pick(base, ["no","huddle"])
    return out.drop_duplicates(subset=["team"])

def parse_coverage_pos() -> pd.DataFrame:
    html = _fetch_html(URLS["coverage_pos"]); dfs = _read_tables(html)
    base = _team_abbr(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=[
            "team","ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed",
            "outside_ypt_allowed","slot_ypt_allowed"
        ])
    out = pd.DataFrame({"team": base["team"]})
    def pick_any(*keys): return _pick(base, list(keys))
    out["ypt_allowed"]         = pick_any("yards","per","target")
    out["wr_ypt_allowed"]      = pick_any("wr","yards","target")
    out["te_ypt_allowed"]      = pick_any("te","yards","target")
    out["rb_ypt_allowed"]      = pick_any("rb","yards","target")
    out["outside_ypt_allowed"] = pick_any("outside","yards","target")
    out["slot_ypt_allowed"]    = pick_any("slot","yards","target")
    return out.drop_duplicates(subset=["team"])

def parse_dl() -> pd.DataFrame:
    html = _fetch_html(URLS["dl"]); dfs = _read_tables(html)
    base = _team_abbr(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["dl_pressure_rate"]          = _pick(base, ["pressure","rate"])
    out["dl_no_blitz_pressure_rate"] = _pick(base, ["no","blitz","pressure"])
    ybc = pd.Series(dtype="float64")
    for c in base.columns:
        cl = c.lower()
        if "before" in cl and "contact" in cl and ("rush" in cl or "carry" in cl):
            ybc = pd.to_numeric(base[c], errors="coerce"); break
    out["dl_ybc_per_rush"] = ybc
    out["dl_stuff_rate"]   = _pick(base, ["stuff"])
    return out.drop_duplicates(subset=["team"])

def parse_pace() -> pd.DataFrame:
    html = _fetch_html(URLS["pace"]); dfs = _read_tables(html)
    base = _team_abbr(_first_team_table(dfs))
    if base.empty:
        return pd.DataFrame(columns=["team","seconds_per_play","plays_per_game"])
    out = pd.DataFrame({"team": base["team"]})
    def pick_num(keys: List[str]) -> pd.Series:
        for c in base.columns:
            cl = c.lower()
            if all(k in cl for k in keys):
                return pd.to_numeric(base[c].astype(str).str.replace(",","",regex=False), errors="coerce")
        return pd.Series(dtype="float64")
    out["seconds_per_play"] = pick_num(["seconds","play"])
    out["plays_per_game"]   = pick_num(["plays","game"])
    return out.drop_duplicates(subset=["team"])

def merge_non_destructive(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty: return left
    r = right.copy()
    if "team" not in r.columns: return left
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
    ap.add_argument("season", type=int, help="Season (e.g., 2025)")
    args = ap.parse_args()
    season = int(args.season)

    _safe_mkdir(DATA_DIR)

    def_t = parse_def_tend()
    off_t = parse_off_tend()
    cov_p = parse_coverage_pos()
    dl    = parse_dl()
    pace  = parse_pace()

    if not def_t.empty: def_t.to_csv(os.path.join(DATA_DIR, f"sharp_def_tend_{season}.csv"), index=False)
    if not off_t.empty: off_t.to_csv(os.path.join(DATA_DIR, f"sharp_off_tend_{season}.csv"), index=False)
    if not cov_p.empty: cov_p.to_csv(os.path.join(DATA_DIR, f"sharp_coverage_pos_{season}.csv"), index=False)
    if not dl.empty:    dl.to_csv(os.path.join(DATA_DIR, f"sharp_dl_{season}.csv"), index=False)
    if not pace.empty:  pace.to_csv(os.path.join(DATA_DIR, f"sharp_pace_{season}.csv"), index=False)

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
