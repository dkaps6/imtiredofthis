# scripts/providers/build_schedule.py
from __future__ import annotations
import io
import gzip
import json
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

log = logging.getLogger("build_schedule")
log.setLevel(logging.INFO)

# ---- Helpers ---------------------------------------------------------------

def _http_get(url: str, headers: Optional[dict] = None, expect_gzip: bool = False) -> bytes:
    r = requests.get(url, headers=headers or {}, timeout=30)
    r.raise_for_status()
    data = r.content
    if expect_gzip:
        # nflverse publishes .csv.gz frequently
        try:
            data = gzip.decompress(data)
        except OSError:
            # if server already returned plain csv
            pass
    return data

def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), low_memory=False)

# ---- NFLVerse primary sources ---------------------------------------------

def _nflverse_urls(season: int) -> List[dict]:
    """Return a list of candidate NFLVerse schedule endpoints for a season."""
    s = str(season)
    return [
        # historic path
        {"url": f"https://github.com/nflverse/nflfastR-data/raw/master/schedules/sched_{s}.csv.gz", "gzip": True},
        # current releases bucket (name has changed across years)
        {"url": f"https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/schedules/sched_{s}.csv.gz", "gzip": True},
        {"url": f"https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/schedules/schedules_{s}.csv.gz", "gzip": True},
        # plain csv fallbacks
        {"url": f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/sched_{s}.csv", "gzip": False},
        {"url": f"https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/schedules/schedules_{s}.csv", "gzip": False},
    ]

def _download_nflverse(season: int) -> pd.DataFrame:
    last_err = None
    for ent in _nflverse_urls(season):
        url, gz = ent["url"], ent["gzip"]
        log.info(f"[schedule] Trying NFLVerse: {url}")
        try:
            b = _http_get(url, expect_gzip=gz)
            df = _read_csv_bytes(b)
            # normalize expected columns
            cols = {c.lower(): c for c in df.columns}
            # standardize common names used elsewhere
            rename = {}
            if "home_team" in cols: rename[cols["home_team"]] = "home_team"
            if "away_team" in cols: rename[cols["away_team"]] = "away_team"
            if "week" in cols: rename[cols["week"]] = "week"
            if "season" in cols: rename[cols["season"]] = "season"
            df = df.rename(columns=rename)
            needed = {"home_team", "away_team", "week"}
            if not needed.issubset(set(df.columns)):
                raise ValueError(f"NFLVerse schedule missing expected columns; got {df.columns.tolist()}")
            log.info("[schedule] NFLVerse fetch OK")
            return df
        except Exception as e:
            last_err = e
            log.warning(f"[schedule] NFLVerse source failed: {e}")
    if last_err:
        raise last_err
    raise RuntimeError("No NFLVerse schedule sources attempted.")

# ---- APISports fallback ----------------------------------------------------

def _download_apisports(season: int, api_key: Optional[str]) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("APISPORTS_KEY not set; cannot use APISports fallback.")
    # American Football / NFL league=1
    url = f"https://v3.american-football.api-sports.io/games?league=1&season={season}"
    log.info(f"[schedule] Trying APISports: {url}")
    headers = {"x-apisports-key": api_key}
    b = _http_get(url, headers=headers, expect_gzip=False)
    js = json.loads(b.decode("utf-8"))
    # Normalize minimal schedule
    rows = []
    for item in js.get("response", []):
        # API returns many fields; we only need week + team codes
        week = item.get("week")
        home = (((item.get("teams") or {}).get("home") or {}).get("name")) or ((item.get("teams") or {}).get("home") or {}).get("code")
        away = (((item.get("teams") or {}).get("away") or {}).get("name")) or ((item.get("teams") or {}).get("away") or {}).get("code")
        if week and home and away:
            rows.append({"season": season, "week": int(week), "home_team": home, "away_team": away})
    if not rows:
        raise RuntimeError("APISports returned no schedule rows.")
    df = pd.DataFrame(rows)
    log.info(f"[schedule] APISports fallback OK with {len(df)} games.")
    return df

# ---- Public API ------------------------------------------------------------

def build_or_get_schedule(season: int, out_path: Optional[str] = None, schedule_override: Optional[str] = None) -> str:
    """
    Returns a CSV path containing a normalized schedule with columns:
    season, week, home_team, away_team
    Resolution order:
      1) schedule_override (local csv)
      2) NFLVerse (multiple endpoints)
      3) APISports fallback (requires APISPORTS_KEY)
    """
    target = Path(out_path or f"data/schedules/schedule_{season}.csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    # 1) local override
    if schedule_override:
        p = Path(schedule_override)
        if not p.exists():
            raise FileNotFoundError(f"Provided --schedule path not found: {p}")
        df = pd.read_csv(p, low_memory=False)
        df.to_csv(target, index=False)
        log.info(f"[schedule] Using local override {p} -> {target}")
        return str(target)

    # 2) NFLVerse
    try:
        df = _download_nflverse(season)
        df.to_csv(target, index=False)
        log.info(f"[schedule] Wrote schedule to {target}")
        return str(target)
    except Exception as nfl_err:
        log.warning(f"[schedule] NFLVerse failed: {nfl_err}")

    # 3) APISports fallback
    try:
        import os
        api_key = os.environ.get("APISPORTS_KEY")
        df = _download_apisports(season, api_key)
        df.to_csv(target, index=False)
        log.info(f"[schedule] Wrote schedule (APISports) to {target}")
        return str(target)
    except Exception as apis_err:
        log.error(f"[schedule] APISports fallback failed: {apis_err}")
        # bubble up so caller can decide to fail early with a clear message
        raise

__all__ = ["build_or_get_schedule"]
