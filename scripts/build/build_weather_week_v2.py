#!/usr/bin/env python3
"""Build weekly weather from the authoritative NFL schedule."""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.build import build_weather_week as legacy
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.runtime_context import resolve_season, resolve_week

OUT_PATH = Path("data/weather_week.csv")
TEAM_WEEK_MAP = Path("data/team_week_map.csv")


def load_schedule_from_team_week_map(
    season: int,
    week: int,
    path: Path = TEAM_WEEK_MAP,
) -> pd.DataFrame:
    """Load the already-validated Full Slate schedule without a second provider call.

    ``team_week_map.csv`` is materialized earlier in the canonical workflow and
    contains two team-perspective rows per game. Weather needs one game row, so
    this adapter validates the required schedule identity and collapses to the
    unique game grain expected by ``build_weather_slate``.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Authoritative team-week map missing/empty: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {
        "season",
        "week",
        "home_team_abbr",
        "away_team_abbr",
        "kickoff_utc",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Authoritative team-week map missing weather schedule columns: {sorted(missing)}"
        )

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    scoped = df.loc[
        df["season"].eq(int(season)) & df["week"].eq(int(week))
    ].copy()
    if scoped.empty:
        raise RuntimeError(
            f"Authoritative team-week map has no rows for season={season} week={week}"
        )

    scoped = scoped.rename(
        columns={"home_team_abbr": "home", "away_team_abbr": "away"}
    )
    keep = ["season", "week", "home", "away", "kickoff_utc"]
    if "game_id" in scoped.columns:
        keep.append("game_id")
    scoped = scoped[keep].copy()

    dedupe_keys = ["game_id"] if "game_id" in scoped.columns else ["home", "away", "kickoff_utc"]
    scoped = scoped.drop_duplicates(dedupe_keys, keep="first").reset_index(drop=True)
    if scoped.empty:
        raise RuntimeError(
            f"Authoritative team-week map normalized to zero games for season={season} week={week}"
        )
    return scoped


def build_weather_slate(season: int, week: int, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return exactly one row per authoritative game for season/week."""
    sched = get_nfl_schedule(int(season)) if schedule is None else schedule.copy()
    if sched is None or sched.empty:
        raise RuntimeError(f"Weather slate schedule is empty for season={season}")

    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "home", "away", "kickoff_utc"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"Weather schedule missing required columns: {sorted(missing)}")

    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    scoped = sched.loc[
        sched["season"].eq(int(season)) & sched["week"].eq(int(week))
    ].copy()
    if scoped.empty:
        raise RuntimeError(f"No authoritative games found for season={season} week={week}")

    # Normalize every schedule provider through the repository-wide team identity
    # contract before stadium lookup. This is where aliases such as LA -> LAR are
    # resolved; weather must never invent its own team code vocabulary.
    scoped["home"] = scoped["home"].map(canon_team)
    scoped["away"] = scoped["away"].map(canon_team)
    invalid_team = scoped["home"].eq("") | scoped["away"].eq("")
    if invalid_team.any():
        raise RuntimeError(
            f"Weather slate contains unresolvable team identity rows={int(invalid_team.sum())}"
        )

    dedupe_keys = ["home", "away"]
    if "game_id" in scoped.columns:
        dedupe_keys = ["game_id"]
    if scoped.duplicated(dedupe_keys, keep=False).any():
        raise RuntimeError(
            f"Authoritative weather slate contains duplicate games for season={season} week={week}"
        )

    rows: list[dict] = []
    for _, game in scoped.iterrows():
        home = str(game["home"]).upper().strip()
        away = str(game["away"]).upper().strip()
        kickoff_utc = pd.to_datetime(game["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(kickoff_utc):
            raise RuntimeError(
                f"Weather slate has invalid kickoff_utc for season={season} week={week} {away}@{home}"
            )

        local_tz = legacy._infer_tz_from_team(home)
        try:
            kickoff_local = kickoff_utc.tz_convert(ZoneInfo(local_tz))
        except Exception:
            local_tz = "UTC"
            kickoff_local = kickoff_utc.tz_convert(ZoneInfo("UTC"))

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "game_id": game.get("game_id", pd.NA),
                "home": home,
                "away": away,
                "kickoff_utc": kickoff_utc,
                "kickoff_local": kickoff_local,
                "local_tz": local_tz,
                "game_date": kickoff_local.date().isoformat(),
                "slate_date": kickoff_local.date().isoformat(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Authoritative weather slate normalized to zero rows")
    return out.sort_values(["kickoff_utc", "away", "home"]).reset_index(drop=True)


def build_weather_output(slate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, game in slate.iterrows():
        wx = legacy._weather_row_for_game(game)
        rain_flag = wx.get("rain_flag")
        wx["season"] = int(game["season"])
        wx["week"] = int(game["week"])
        wx["game_id"] = game.get("game_id", pd.NA)
        wx["temp_f"] = wx.get("temp_F_mean")
        wx["wind_mph"] = wx.get("wind_mph_mean")
        wx["precip_flag"] = None if rain_flag is None else (1 if rain_flag else 0)
        wx["notes"] = "weather_ok" if wx.get("forecast_ok") else "weather_unavailable"
        rows.append(wx)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Weather builder produced zero game rows from a non-empty authoritative slate")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--date", default=None, help="Accepted for workflow compatibility; week authority comes from runtime context")
    args = parser.parse_args()

    season = int(args.season) if args.season is not None else int(resolve_season())
    week = int(resolve_week())

    # Full Slate already validates and materializes the schedule before weather.
    # Reuse that canonical artifact so a transient second provider request cannot
    # break weather after schedule authority has already succeeded.
    if TEAM_WEEK_MAP.exists() and TEAM_WEEK_MAP.stat().st_size > 0:
        schedule = load_schedule_from_team_week_map(season, week)
        source = str(TEAM_WEEK_MAP)
    else:
        schedule = None
        source = "get_nfl_schedule"

    slate = build_weather_slate(season, week, schedule=schedule)
    print(
        f"[weather_v2] authoritative slate season={season} week={week} "
        f"games={len(slate)} source={source}"
    )

    weather = build_weather_output(slate)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUT_PATH, index=False)

    ok = int(pd.to_numeric(weather.get("forecast_ok", 0), errors="coerce").fillna(0).sum())
    unavailable = int(len(weather) - ok)
    print(f"[weather_v2] wrote {len(weather)} games -> {OUT_PATH}; forecast_ok={ok} forecast_unavailable={unavailable}")
    if unavailable and ok == 0:
        print("[weather_v2] no kickoff forecasts are currently available; schedule/stadium rows were retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
