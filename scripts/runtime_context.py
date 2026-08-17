#!/usr/bin/env python3
"""Shared runtime season/week resolution for the NFL pipeline.

This module is the runtime source of truth for active season, prior season,
slate date, and NFL week.  Week resolution is always based on the authoritative
team-week schedule.  Calendar/ISO week numbers are intentionally never used.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.config import PRIOR_SEASON, SEASON, SLATE_DATE

TEAM_WEEK_MAP_PATH = Path("data/team_week_map.csv")


def _as_int(value, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label}: {value!r}") from exc


def resolve_season() -> int:
    """Return the active season and reject conflicting runtime values."""
    env_season = _as_int(os.getenv("SEASON", SEASON), "SEASON")
    if env_season != int(SEASON):
        raise RuntimeError(
            f"Season context mismatch: env SEASON={env_season} but config.SEASON={SEASON}"
        )
    return env_season


def resolve_prior_season() -> int:
    """Return the configured prior season and validate chronology."""
    season = resolve_season()
    prior = _as_int(os.getenv("PRIOR_SEASON", PRIOR_SEASON), "PRIOR_SEASON")
    if prior >= season:
        raise RuntimeError(
            f"PRIOR_SEASON={prior} must be earlier than SEASON={season}"
        )
    return prior


def resolve_slate_date() -> str:
    return os.getenv("SLATE_DATE", SLATE_DATE).strip()


def _load_team_week_map(path: Path, season: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Cannot resolve NFL week: {path} missing or empty")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve NFL week from {path}: {exc}") from exc
    if frame.empty:
        raise RuntimeError(f"Cannot resolve NFL week: {path} has 0 rows")

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "season" not in frame.columns or "week" not in frame.columns:
        raise RuntimeError(
            f"Cannot resolve NFL week: {path} must contain season and week columns"
        )
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    scoped = frame.loc[frame["season"].eq(int(season)) & frame["week"].notna()].copy()
    if scoped.empty:
        raise RuntimeError(f"Cannot resolve NFL week: no rows for season {season} in {path}")
    return scoped


def _schedule_timestamp(frame: pd.DataFrame) -> pd.Series:
    """Return the best available UTC kickoff timestamp per schedule row."""
    for candidate in (
        "kickoff_utc",
        "kickoff_ts",
        "commence_time",
        "game_timestamp",
        "kickoff_local",
        "gameday",
        "game_date",
        "date",
    ):
        if candidate not in frame.columns:
            continue
        raw = frame[candidate]
        # game_timestamp is sometimes epoch seconds.
        if candidate == "game_timestamp":
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().any():
                parsed = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
                if parsed.notna().any():
                    return parsed
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")


def resolve_week(
    season: int | None = None,
    slate_date: str | None = None,
    team_week_map_path: str | Path = TEAM_WEEK_MAP_PATH,
    *,
    now: pd.Timestamp | None = None,
) -> int:
    """Resolve one authoritative NFL week from ``team_week_map.csv``.

    Resolution order:
    1. Scope to the active season.
    2. If a slate date is supplied, match games on that local calendar date.
    3. If slate date is blank, choose the nearest upcoming scheduled game week;
       if the season is over, choose the most recent completed scheduled week.

    This deliberately never uses ISO/calendar week numbers.
    """
    season = int(season if season is not None else resolve_season())
    slate_date = (slate_date if slate_date is not None else resolve_slate_date()).strip()
    path = Path(team_week_map_path)
    scoped = _load_team_week_map(path, season)
    kickoff = _schedule_timestamp(scoped)
    scoped = scoped.assign(_kickoff=kickoff)

    if slate_date:
        target = pd.to_datetime(slate_date, errors="coerce")
        if pd.isna(target):
            raise RuntimeError(f"Invalid SLATE_DATE={slate_date!r}; expected YYYY-MM-DD")
        target_date = target.date()
        dated = scoped.loc[scoped["_kickoff"].dt.date.eq(target_date)].copy()
        if dated.empty:
            # Some schedule maps expose a separate slate/date column but no parsed kickoff.
            for candidate in ("slate_date", "gameday", "game_date", "date"):
                if candidate not in scoped.columns:
                    continue
                parsed = pd.to_datetime(scoped[candidate], errors="coerce")
                mask = parsed.dt.date.eq(target_date)
                if mask.any():
                    dated = scoped.loc[mask].copy()
                    break
        if dated.empty:
            raise RuntimeError(
                f"Cannot resolve NFL week: no scheduled games for season={season}, slate_date={slate_date}"
            )
        weeks = sorted({int(v) for v in dated["week"].dropna().tolist()})
    else:
        valid_time = scoped.loc[scoped["_kickoff"].notna()].copy()
        if valid_time.empty:
            weeks = sorted({int(v) for v in scoped["week"].dropna().tolist()})
            if len(weeks) != 1:
                raise RuntimeError(
                    "Cannot infer current NFL week from a full-season schedule without kickoff timestamps; "
                    f"candidates={weeks}"
                )
        else:
            current = now if now is not None else pd.Timestamp.now(tz="UTC")
            if current.tzinfo is None:
                current = current.tz_localize("UTC")
            else:
                current = current.tz_convert("UTC")
            upcoming = valid_time.loc[valid_time["_kickoff"] >= current].sort_values("_kickoff")
            if not upcoming.empty:
                chosen_time = upcoming.iloc[0]["_kickoff"]
                # Games belonging to a week can span several days.  Choose the week of
                # the nearest upcoming game rather than matching a single timestamp.
                chosen_week = int(upcoming.iloc[0]["week"])
                weeks = [chosen_week]
            else:
                completed = valid_time.sort_values("_kickoff")
                weeks = [int(completed.iloc[-1]["week"])]

    if len(weeks) != 1:
        raise RuntimeError(
            f"Cannot resolve one authoritative NFL week for season={season}, "
            f"slate_date={slate_date or '<latest>'}; candidates={weeks}"
        )
    week = int(weeks[0])
    if week <= 0:
        raise RuntimeError(f"Resolved invalid NFL week: {week}")
    return week


def log_runtime_context() -> None:
    season = resolve_season()
    prior = resolve_prior_season()
    slate = resolve_slate_date()
    print(f"[runtime] SEASON={season}")
    print(f"[runtime] PRIOR_SEASON={prior}")
    print(f"[runtime] SLATE_DATE={slate or '<latest>'}")
    if TEAM_WEEK_MAP_PATH.exists() and TEAM_WEEK_MAP_PATH.stat().st_size > 0:
        try:
            week = resolve_week(season=season, slate_date=slate)
        except Exception as exc:
            print(f"[runtime] WEEK unresolved: {exc}")
        else:
            print(f"[runtime] WEEK={week}")
