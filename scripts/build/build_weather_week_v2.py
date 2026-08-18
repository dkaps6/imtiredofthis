#!/usr/bin/env python3
"""Build weekly weather from the authoritative NFL schedule.

This replaces the old weather slate discovery behavior that tried to infer games
from `team_week_map.csv`, odds artifacts, or `data/schedule.csv`. The production
schedule authority is now `get_nfl_schedule(season)` plus `resolve_week()`, so the
weather layer should consume exactly that same game set.

The existing weather module still owns stadium metadata and NWS forecast parsing;
this entry point only owns authoritative slate construction and output semantics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.build import build_weather_week as legacy
from scripts.build._schedule_utils import get_nfl_schedule
from scripts.runtime_context import resolve_season, resolve_week

OUT_PATH = Path("data/weather_week.csv")


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

    # One game should appear once. Duplicate game rows would duplicate weather
    # evidence and are therefore a hard data-contract failure.
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
        if not home or not away:
            raise RuntimeError("Weather slate contains blank home/away identity")

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
    """Apply the existing stadium/NWS weather logic to the authoritative slate."""
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

    slate = build_weather_slate(season, week)
    print(
        f"[weather_v2] authoritative slate season={season} week={week} "
        f"games={len(slate)} source=get_nfl_schedule"
    )

    weather = build_weather_output(slate)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUT_PATH, index=False)

    ok = int(pd.to_numeric(weather.get("forecast_ok", 0), errors="coerce").fillna(0).sum())
    unavailable = int(len(weather) - ok)
    print(
        f"[weather_v2] wrote {len(weather)} games -> {OUT_PATH}; "
        f"forecast_ok={ok} forecast_unavailable={unavailable}"
    )
    if unavailable and ok == 0:
        print(
            "[weather_v2] no kickoff forecasts are currently available; "
            "schedule/stadium rows were retained and will populate automatically when the NWS forecast horizon reaches the games"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
