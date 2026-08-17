#!/usr/bin/env python3
"""Build/cache the authoritative NFL schedule with nflreadpy, then delegate to make_team_week_map.

Primary source: nflreadpy.load_schedules(seasons=[season])
Fallback behavior remains inside scripts/utils/make_team_week_map.py if nflreadpy
cannot supply a valid regular-season schedule.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

CACHE_DIR = Path("data/schedules")
EXPECTED_REGULAR_GAMES = 272
EXPECTED_TEAMS = 32
EXPECTED_WEEKS = 18


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _normalize_schedule(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise RuntimeError(f"nflreadpy returned an empty schedule for season={season}")

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "season" not in df.columns:
        raise RuntimeError("nflreadpy schedule is missing required column 'season'")
    season_col = pd.to_numeric(df["season"], errors="coerce")
    df = df.loc[season_col.eq(int(season))].copy()
    if df.empty:
        raise RuntimeError(f"nflreadpy schedule contains no rows for season={season}")

    if "game_type" in df.columns:
        game_type = df["game_type"].astype("string").str.upper().str.strip()
        reg = df.loc[game_type.eq("REG")].copy()
        if not reg.empty:
            df = reg

    rename = {}
    if "gamedate" in df.columns and "gameday" not in df.columns:
        rename["gamedate"] = "gameday"
    if "home" in df.columns and "home_team" not in df.columns:
        rename["home"] = "home_team"
    if "away" in df.columns and "away_team" not in df.columns:
        rename["away"] = "away_team"
    if rename:
        df.rename(columns=rename, inplace=True)

    required = ["season", "week", "game_id", "home_team", "away_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"nflreadpy schedule missing required columns: {missing}")

    if "gameday" not in df.columns:
        kickoff_source = None
        for candidate in ("kickoff", "game_date", "date"):
            if candidate in df.columns:
                kickoff_source = candidate
                break
        if kickoff_source is None:
            raise RuntimeError("nflreadpy schedule has no gameday/date column")
        df["gameday"] = df[kickoff_source]

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
    df["home_team"] = df["home_team"].fillna("").astype(str).map(canon_team)
    df["away_team"] = df["away_team"].fillna("").astype(str).map(canon_team)
    df["game_id"] = df["game_id"].astype("string").str.strip()

    df = df.dropna(subset=["season", "week", "gameday", "game_id"])
    df = df.loc[df["home_team"].ne("") & df["away_team"].ne("")].copy()
    df = df.drop_duplicates(subset=["game_id"], keep="last")
    df = df.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)

    teams = set(df["home_team"]) | set(df["away_team"])
    weeks = sorted(pd.to_numeric(df["week"], errors="coerce").dropna().astype(int).unique().tolist())

    # Hard validation: the NFL regular season is 272 games, 32 clubs, Weeks 1-18.
    # Failing here is preferable to emitting a partial schedule that corrupts every
    # opponent/week join downstream.
    if len(df) != EXPECTED_REGULAR_GAMES:
        raise RuntimeError(
            f"schedule completeness failure for {season}: games={len(df)} expected={EXPECTED_REGULAR_GAMES}"
        )
    if len(teams) != EXPECTED_TEAMS:
        raise RuntimeError(
            f"schedule completeness failure for {season}: teams={len(teams)} expected={EXPECTED_TEAMS}"
        )
    if weeks != list(range(1, EXPECTED_WEEKS + 1)):
        raise RuntimeError(
            f"schedule completeness failure for {season}: weeks={weeks} expected=1..{EXPECTED_WEEKS}"
        )

    return df[["season", "week", "gameday", "game_id", "home_team", "away_team"]]


def _fetch_nflreadpy(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    print(f"[schedule_v2] loading schedule with nflreadpy for season={season}")
    raw = nfl.load_schedules(seasons=[int(season)])
    df = _normalize_schedule(_to_pandas(raw), int(season))
    print(
        f"[schedule_v2] validated nflreadpy schedule: games={len(df)} "
        f"weeks={df['week'].nunique()} teams={len(set(df['home_team']) | set(df['away_team']))}"
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", default="data/team_week_map.csv")
    args = parser.parse_args()

    season = int(args.season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"schedule_{season}.csv"

    schedule_path: Path | None = None
    try:
        schedule = _fetch_nflreadpy(season)
        schedule.to_csv(cache, index=False)
        schedule_path = cache
        print(f"[schedule_v2] cached authoritative schedule -> {cache}")
    except Exception as exc:
        print(f"[schedule_v2] nflreadpy primary source failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[schedule_v2] delegating to legacy provider fallbacks", file=sys.stderr)

    cmd = [
        sys.executable,
        "scripts/utils/make_team_week_map.py",
        "--season",
        str(season),
        "--out",
        str(args.out),
    ]
    if schedule_path is not None:
        cmd.extend(["--schedule", str(schedule_path)])

    print("[schedule_v2] delegate:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
