# scripts/providers/build_schedule.py
"""Historical/research schedule helper.

This module is retained because the historical backtest input builder imports
``build_or_get_schedule``. It is NOT a canonical 2026 production schedule
provider. Live production schedule authority is
``scripts/utils/build_team_week_map_v2.py`` via Full Slate.
"""
from __future__ import annotations
import io
import gzip
import logging
from typing import List
import pandas as pd
import requests

log = logging.getLogger("build_schedule")
log.setLevel(logging.INFO)

TEAM_FIXES = {
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "ARZ": "ARI", "LA": "LAR", "WSH": "WAS"
}

def _canon_team(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = x.strip().upper()
    return TEAM_FIXES.get(x, x)


def _http_get(url: str, expect_gzip: bool = False) -> bytes:
    headers = {
        "User-Agent": "imtiredofthis/1.0 (historical schedule fetch)",
        "Accept": "*/*",
    }
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    data = r.content
    if expect_gzip:
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    return data


def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), low_memory=False)


def _nflverse_master_urls() -> List[dict]:
    return [
        {"url": "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedules.csv.gz", "gzip": True},
        {"url": "https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/schedules/schedules.csv.gz", "gzip": True},
        {"url": "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedules.csv", "gzip": False},
        {"url": "https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/schedules/schedules.csv", "gzip": False},
    ]


def _download_nflverse_master(season: int) -> pd.DataFrame:
    last_err = None
    for ent in _nflverse_master_urls():
        url, gz = ent["url"], ent["gzip"]
        log.info(f"[schedule] Trying NFLVerse master: {url}")
        try:
            b = _http_get(url, expect_gzip=gz)
            df = _read_csv_bytes(b)
            cols_lower = {c.lower(): c for c in df.columns}
            rename = {}
            for want in ["season", "week", "home_team", "away_team", "game_id"]:
                if want in cols_lower:
                    rename[cols_lower[want]] = want
            if "start_time" in cols_lower:
                rename[cols_lower["start_time"]] = "kickoff_utc"
            elif "game_time" in cols_lower:
                rename[cols_lower["game_time"]] = "kickoff_utc"
            df = df.rename(columns=rename)

            if "season" not in df.columns:
                raise ValueError("NFLVerse master schedule missing 'season' column")
            df = df[df["season"].astype(int) == int(season)].copy()
            if df.empty:
                raise ValueError(f"No rows for season {season} in master schedule")

            if "kickoff_utc" in df.columns:
                df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
            else:
                df["kickoff_utc"] = pd.NaT

            for c in ("home_team", "away_team"):
                if c in df.columns:
                    df[c] = df[c].astype(str).map(_canon_team)

            for c in ("week", "home_team", "away_team"):
                if c not in df.columns:
                    raise ValueError(f"Master schedule missing required column '{c}'")

            keep = ["season", "week", "home_team", "away_team", "kickoff_utc"]
            if "game_id" in df.columns:
                keep.append("game_id")
            df = df[keep].drop_duplicates().reset_index(drop=True)
            log.info(f"[schedule] NFLVerse master OK: {len(df)} rows for {season}")
            return df
        except Exception as exc:
            last_err = exc
            log.warning(f"[schedule] NFLVerse master failed: {exc}")
    if last_err is None:
        raise RuntimeError("No nflverse schedule source was attempted")
    raise last_err


def _download_nfl_data_py(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    log.info("[schedule] Trying nfl_data_py.import_schedules()")
    df = nfl.import_schedules([season])
    rename = {}
    if "home_team" not in df.columns and "home" in df.columns:
        rename["home"] = "home_team"
    if "away_team" not in df.columns and "away" in df.columns:
        rename["away"] = "away_team"
    if "week" not in df.columns and "game_week" in df.columns:
        rename["game_week"] = "week"
    df = df.rename(columns=rename)

    if "kickoff_utc" not in df.columns:
        if "start_time" in df.columns:
            df["kickoff_utc"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        elif {"gameday", "game_time"}.issubset(df.columns):
            df["kickoff_utc"] = pd.to_datetime(
                df["gameday"] + " " + df["game_time"], errors="coerce", utc=True
            )
        else:
            df["kickoff_utc"] = pd.NaT

    df = df[df["season"].astype(int) == int(season)].copy()
    for c in ("home_team", "away_team"):
        df[c] = df[c].astype(str).map(_canon_team)
    keep = ["season", "week", "home_team", "away_team", "kickoff_utc"]
    if "game_id" in df.columns:
        keep.append("game_id")
    df = df[keep].drop_duplicates().reset_index(drop=True)
    log.info(f"[schedule] nfl_data_py OK: {len(df)} rows for {season}")
    return df


def build_or_get_schedule(season: int) -> pd.DataFrame:
    """Return historical/research schedule rows for ``season``.

    This function intentionally has no role in live Full Slate production.
    """
    try:
        return _download_nflverse_master(season)
    except Exception as exc:
        log.warning(f"[schedule] Master fetch failed, trying nfl_data_py fallback: {exc}")
    return _download_nfl_data_py(season)
