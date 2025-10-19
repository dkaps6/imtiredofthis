#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharp Football provider (public pages) → fallback enrichers for team form.

Writes (seasoned file names):
  data/sharp_team_form.csv                    # merged superset used by make_team_form fallbacks
  data/sharp_def_tend_{season}.csv            # defensive tendencies (blitz/sub + light/heavy box if present)
  data/sharp_off_tend_{season}.csv            # offensive tendencies (motion/PA/shotgun/no-huddle + AY/Att)
  data/sharp_coverage_pos_{season}.csv        # YPT allowed by position/align
  data/sharp_dl_{season}.csv                  # DL pressure/YBC/stuff (if available)
  data/sharp_pace_{season}.csv                # seconds/play, plays/game (if available)

Debug (optional --dump-html):
  data/_sharp_dump_def_tend_{season}.html
  data/_sharp_dump_off_tend_{season}.html
  data/_sharp_dump_coverage_pos_{season}.html
  data/_sharp_dump_dl_{season}.html
  data/_sharp_dump_pace_{season}.html

Proxy support (optional):
  - Env: SHARP_PROXY_BASE (e.g. https://your-proxy.example/fetch?url=)
         SHARP_PROXY_AUTH_HEADER (e.g. X-Api-Key) [optional]
         SHARP_PROXY_AUTH_TOKEN  (e.g. abc123)    [optional]
  - CLI: --proxy-base "https://your-proxy.example/fetch?url="
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from typing import List, Optional
from urllib.parse import quote, urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

DATA_DIR = "data"

# === Team normalization ===
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

# === HTTP session with retries + rotated UA ===
UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]
BASE_HEADERS = {
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

def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ---------- Proxy wiring ----------
class ProxyCfg:
    def __init__(self, base: Optional[str], auth_header: Optional[str], auth_token: Optional[str]):
        self.base = (base or "").strip()
        self.auth_header = (auth_header or "").strip()
        self.auth_token = (auth_token or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.base)

    def wrap(self, url: str) -> str:
        # Accept either style:
        #   1) base already contains ?url= (e.g., https://proxy/fetch?url=)
        #   2) base is a path (e.g., https://proxy/fetch/) → append ?url=
        if "?" in self.base and self.base.endswith("url="):
            return f"{self.base}{quote(url, safe='')}"
        if self.base.endswith("?"):
            return f"{self.base}url={quote(url, safe='')}"
        # default: ensure trailing slash and append ?url=
        b = self.base if self.base.endswith("/") else self.base + "/"
        return f"{b}?url={quote(url, safe='')}"

def _load_proxy_from_env_and_args(args) -> ProxyCfg:
    base = args.proxy_base or os.getenv("SHARP_PROXY_BASE", "").strip()
    hdr  = os.getenv("SHARP_PROXY_AUTH_HEADER", "").strip()
    tok  = os.getenv("SHARP_PROXY_AUTH_TOKEN", "").strip()
    return ProxyCfg(base, hdr, tok)
# ----------------------------------

def _session(extra_headers: Optional[dict] = None) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    headers = BASE_HEADERS.copy()
    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v})
    s.headers.update(headers)
    return s

def _maybe_season_url(url: str, season: int) -> list[str]:
    # Try “as-is”, then ?season=, ?year= patterns
    cands = [url]
    for q in (f"season={season}", f"year={season}", f"Season={season}"):
        sep = "&" if "?" in url else "?"
        cands.append(f"{url}{sep}{q}")
    return cands

# === Robust table reading with verbose logging + optional HTML dumps ===
def _read_tables_url(url: str, dump_path: Optional[str] = None, proxy: Optional[ProxyCfg] = None) -> List[pd.DataFrame]:
    """Try multiple ways to get tables; never raise. Optionally dump HTML for debugging.
       If proxy is provided, this function expects the incoming url to already be
       a proxy-wrapped URL (we'll just request it).
    """

    # A) direct read_html
    try:
        dfs = pd.read_html(url)
        if dfs:
            print(f"[sharp][read_html(url)] OK tables={len(dfs)} :: {url}")
            return dfs
        else:
            print(f"[sharp][read_html(url)] 0 tables :: {url}")
    except Exception as e:
        print(f"[sharp][read_html(url)] error: {e}")

    # B) requests + read_html(html)
    html = ""
    try:
        extra = {}
        if proxy and proxy.enabled and proxy.auth_header and proxy.auth_token:
            extra[proxy.auth_header] = proxy.auth_token

        sess = _session(extra_headers=extra)
        r = sess.get(url, timeout=25)
        html = r.text or ""
        print(f"[sharp][requests] status={r.status_code} len={len(html)} :: {url}")
        if dump_path:
            try:
                with open(dump_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"[sharp][dump] wrote {dump_path}")
            except Exception as de:
                print(f"[sharp][dump] error: {de}")

        if r.status_code == 200 and "<table" in html.lower():
            try:
                dfs = pd.read_html(html)
                print(f"[sharp][read_html(html)] tables={len(dfs)} :: {url}")
                if dfs: return dfs
            except Exception as e2:
                print(f"[sharp][read_html(html)] error: {e2}")
        else:
            print("[sharp] no <table> tag detected in HTML body (blocked or JS-only table?)")
    except Exception as e:
        print(f"[sharp][requests] error: {e}")

    # C) BeautifulSoup per-table extraction
    try:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for t in soup.find_all("table"):
            try:
                d = pd.read_html(str(t))
                if d: out.extend(d)
            except Exception:
                continue
        print(f"[sharp][bs4 per-table] tables={len(out)} :: {url}")
        if out:
            return out
    except Exception as e:
        print(f"[sharp][bs4] error: {e}")
    return []

def _first_team_table(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs: return pd.DataFrame()
    best, best_score = None, -1
    keys = set(TEAM_NAME_TO_ABBR.keys())
    for df in dfs:
        if not isinstance(df, pd.DataFrame) or df.shape[1] == 0: continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        score = 0
        first = df.iloc[:, 0].astype(str).str.upper().str.replace(r"[^A-Z\s]", "", regex=True).str.strip()
        if first.isin(keys).mean() > 0.15: score += 4
        if any(tok in df.columns[0].lower() for tok in ("team","defense","offense")): score += 1
        if score > best_score:
            best, best_score = df, score
    return best.copy() if best is not None else pd.DataFrame()

def _team_abbr_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    col = df.columns[0]
    for c in df.columns:
        cl = c.lower()
        if "team" in cl or "defense" in cl or "offense" in cl:
            col = c; break
    df = df.rename(columns={col: "team_name"})
    df["team_name"] = df["team_name"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    df["team"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    return df[df["team"].notna()].copy()

def _clean_pct_or_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    pct = s.str.endswith("%")
    num = pd.to_numeric(s.str.rstrip("%"), errors="coerce")
    return pd.Series(np.where(pct, num/100.0, num), index=s.index, dtype="float64")

def _pick_rate(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keys):
            return _clean_pct_or_num(df[c])
    return pd.Series(dtype="float64")

# === Page parsers ===
def parse_def_tend(url: str, dump: Optional[str], proxy: Optional[ProxyCfg]) -> pd.DataFrame:
    base = _team_abbr_col(_first_team_table(_read_tables_url(url, dump, proxy)))
    if base.empty:
        return pd.DataFrame(columns=["team","blitz_rate","sub_package_rate","light_box_rate","heavy_box_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["blitz_rate"] = _pick_rate(base, ["blitz"])
    sub = _pick_rate(base, ["sub"])
    if sub.empty: sub = _pick_rate(base, ["nickel"])
    out["sub_package_rate"] = sub
    heavy = pd.Series(dtype="float64"); light = pd.Series(dtype="float64")
    for c in base.columns:
        cl = c.lower()
        if "box" in cl and ("8" in cl or "heavy" in cl or "stack" in cl):
            heavy = _clean_pct_or_num(base[c])
        if "box" in cl and "light" in cl:
            light = _clean_pct_or_num(base[c])
    out["heavy_box_rate"] = heavy
    out["light_box_rate"] = light
    return out.drop_duplicates(subset=["team"])

def parse_off_tend(url: str, dump: Optional[str], proxy: Optional[ProxyCfg]) -> pd.DataFrame:
    base = _team_abbr_col(_first_team_table(_read_tables_url(url, dump, proxy)))
    if base.empty:
        return pd.DataFrame(columns=["team","motion_rate","play_action_rate","shotgun_rate","no_huddle_rate","air_yards_per_att"])
    out = pd.DataFrame({"team": base["team"]})
    out["motion_rate"]      = _pick_rate(base, ["motion"])
    out["play_action_rate"] = _pick_rate(base, ["play","action"])
    out["shotgun_rate"]     = _pick_rate(base, ["shotgun"])
    out["no_huddle_rate"]   = _pick_rate(base, ["no","huddle"])
    for c in base.columns:
        if "air" in c.lower() and "att" in c.lower():
            out["air_yards_per_att"] = pd.to_numeric(base[c], errors="coerce")
            break
    if "air_yards_per_att" not in out.columns:
        out["air_yards_per_att"] = np.nan
    return out.drop_duplicates(subset=["team"])

def parse_coverage_pos(url: str, dump: Optional[str], proxy: Optional[ProxyCfg]) -> pd.DataFrame:
    base = _team_abbr_col(_first_team_table(_read_tables_url(url, dump, proxy)))
    if base.empty:
        return pd.DataFrame(columns=[
            "team","ypt_allowed","wr_ypt_allowed","te_ypt_allowed","rb_ypt_allowed",
            "outside_ypt_allowed","slot_ypt_allowed"
        ])
    out = pd.DataFrame({"team": base["team"]})
    def pick(*k): return _pick_rate(base, list(k))
    out["ypt_allowed"]         = pick("yards","per","target")
    out["wr_ypt_allowed"]      = pick("wr","yards","target")
    out["te_ypt_allowed"]      = pick("te","yards","target")
    out["rb_ypt_allowed"]      = pick("rb","yards","target")
    out["outside_ypt_allowed"] = pick("outside","yards","target")
    out["slot_ypt_allowed"]    = pick("slot","yards","target")
    return out.drop_duplicates(subset=["team"])

def parse_dl(url: str, dump: Optional[str], proxy: Optional[ProxyCfg]) -> pd.DataFrame:
    base = _team_abbr_col(_first_team_table(_read_tables_url(url, dump, proxy)))
    if base.empty:
        return pd.DataFrame(columns=["team","dl_pressure_rate","dl_no_blitz_pressure_rate","dl_ybc_per_rush","dl_stuff_rate"])
    out = pd.DataFrame({"team": base["team"]})
    out["dl_pressure_rate"]          = _pick_rate(base, ["pressure","rate"])
    out["dl_no_blitz_pressure_rate"] = _pick_rate(base, ["no","blitz","pressure"])
    ybc = pd.Series(dtype="float64")
    for c in base.columns:
        cl = c.lower()
        if "before" in cl and "contact" in cl and ("rush" in cl or "carry" in cl):
            ybc = pd.to_numeric(base[c], errors="coerce"); break
    out["dl_ybc_per_rush"] = ybc
    out["dl_stuff_rate"]   = _pick_rate(base, ["stuff"])
    return out.drop_duplicates(subset=["team"])

def parse_pace(url: str, dump: Optional[str], proxy: Optional[ProxyCfg]) -> pd.DataFrame:
    base = _team_abbr_col(_first_team_table(_read_tables_url(url, dump, proxy)))
    if base.empty:
        return pd.DataFrame(columns=["team","seconds_per_play","plays_per_game"])
    def pick_num(keys: list[str]) -> pd.Series:
        for c in base.columns:
            cl = c.lower()
            if all(k in cl for k in keys):
                return pd.to_numeric(base[c].astype(str).str.replace(",","",regex=False), errors="coerce")
        return pd.Series(dtype="float64")
    out = pd.DataFrame({"team": base["team"]})
    out["seconds_per_play"] = pick_num(["seconds","play"])
    out["plays_per_game"]   = pick_num(["plays","game"])
    return out.drop_duplicates(subset=["team"])

# === Merge helper ===
def merge_non_destructive(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty or "team" not in right.columns: return left
    r = right.copy()
    keep = [c for c in r.columns if c != "team"]
    out = left.merge(r, on="team", how="left", suffixes=("","_ext"))
    for c in keep:
        ext = f"{c}_ext"
        if ext in out.columns:
            out[c] = out[c].combine_first(out[ext])
            out.drop(columns=[ext], inplace=True)
    return out

def _log_count(name: str, df: pd.DataFrame):
    print(f"[sharp] {name:16s} rows={0 if df is None else len(df)}")

# ---------- Candidate URL expansion (proxy-first if configured) ----------
def _expand_candidates(base_url: str, season: int, proxy: ProxyCfg) -> List[str]:
    # season variants
    raw = _maybe_season_url(base_url, season)
    out: List[str] = []
    for u in raw:
        if proxy.enabled:
            out.append(proxy.wrap(u))  # proxy-first
        out.append(u)                  # then direct
    return out
# ------------------------------------------------------------------------

# === CLI ===
def main():
    ap = argparse.ArgumentParser()
    # support both --season and positional
    ap.add_argument("--season", type=int, help="Season (e.g., 2025)")
    ap.add_argument("season_pos", nargs="?", type=int, help="Season positional (e.g., 2025)")
    ap.add_argument("--dump-html", action="store_true", help="Dump fetched HTML to data/_sharp_dump_*.html")
    ap.add_argument("--proxy-base", type=str, default=None, help="Proxy base (e.g., https://proxy/fetch?url=)")
    args = ap.parse_args()

    season = args.season if args.season is not None else args.season_pos
    if season is None:
        ap.error("season is required (use --season 2025 or positional 2025)")

    proxy = _load_proxy_from_env_and_args(args)

    _safe_mkdir(DATA_DIR)

    def attempt(parse_fn, base_url, label):
        # generate proxy-first candidates
        cands = _expand_candidates(base_url, int(season), proxy)
        dump_path = os.path.join(DATA_DIR, f"_sharp_dump_{label}_{season}.html") if args.dump_html else None
        for u in cands:
            df = parse_fn(u, dump_path, proxy if u.startswith(proxy.base) and proxy.enabled else None)
            if df is not None and not df.empty:
                _log_count(label, df)
                return df
        _log_count(label, pd.DataFrame())
        return pd.DataFrame()

    def_t = attempt(parse_def_tend, URLS["def_tend"],   "def_tend")
    off_t = attempt(parse_off_tend, URLS["off_tend"],   "off_tend")
    cov_p = attempt(parse_coverage_pos, URLS["coverage_pos"], "coverage_pos")
    dl    = attempt(parse_dl,       URLS["dl"],         "dl")
    pace  = attempt(parse_pace,     URLS["pace"],       "pace")

    # Write individual CSVs (seasoned)
    if not def_t.empty: def_t.to_csv(os.path.join(DATA_DIR, f"sharp_def_tend_{season}.csv"), index=False)
    if not off_t.empty: off_t.to_csv(os.path.join(DATA_DIR, f"sharp_off_tend_{season}.csv"), index=False)
    if not cov_p.empty: cov_p.to_csv(os.path.join(DATA_DIR, f"sharp_coverage_pos_{season}.csv"), index=False)
    if not dl.empty:    dl.to_csv(os.path.join(DATA_DIR, f"sharp_dl_{season}.csv"), index=False)
    if not pace.empty:  pace.to_csv(os.path.join(DATA_DIR, f"sharp_pace_{season}.csv"), index=False)

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
        "air_yards_per_att",
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
