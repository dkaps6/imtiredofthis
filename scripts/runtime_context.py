#!/usr/bin/env python3
"""Shared runtime season/week resolution for the NFL pipeline.

This module is intentionally small and dependency-light. It provides one
runtime source of truth for the active season, prior season, slate date, and
NFL week so legacy builders do not silently fall back to stale constants.
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


def resolve_week(
    season: int | None = None,
    slate_date: str | None = None,
    team_week_map_path: str | Path = TEAM_WEEK_MAP_PATH,
) -> int:
    """Resolve the authoritative NFL week from team_week_map.csv.

    Resolution order:
    1. Filter to the active season.
    2. If a slate date is supplied and the map has a date-like column, use the
       rows matching that date.
    3. Otherwise require the season-filtered table to contain exactly one
       non-null week. This is deliberately strict so a stale Week 10-style
       constant can never silently leak into a run.
    """
    season = int(season if season is not None else resolve_season())
    slate_date = (slate_date if slate_date is not None else resolve_slate_date()).strip()
    path = Path(team_week_map_path)

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

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    scoped = frame.loc[frame["season"].eq(season)].copy()
    if scoped.empty:
        raise RuntimeError(f"Cannot resolve NFL week: no rows for season {season} in {path}")

    if slate_date:
        target = pd.to_datetime(slate_date, errors="coerce")
        if pd.isna(target):
            raise RuntimeError(f"Invalid SLATE_DATE={slate_date!r}; expected YYYY-MM-DD")
        target_date = target.date()
        for candidate in (
            "date",
            "game_date",
            "gameday",
            "slate_date",
            "kickoff_ts",
            "commence_time",
        ):
            if candidate not in scoped.columns:
                continue
            parsed = pd.to_datetime(scoped[candidate], errors="coerce", utc=True)
            mask = parsed.dt.date.eq(target_date)
            if mask.any():
                scoped = scoped.loc[mask].copy()
                break

    weeks = sorted(
        {
            int(v)
            for v in pd.to_numeric(scoped["week"], errors="coerce").dropna().tolist()
        }
    )
    if len(weeks) != 1:
        raise RuntimeError(
            f"Cannot resolve one authoritative NFL week for season={season}, "
            f"slate_date={slate_date or '<empty>'}; candidates={weeks}"
        )

    week = weeks[0]
    if week <= 0:
        raise RuntimeError(f"Resolved invalid NFL week: {week}")
    return week


def log_runtime_context() -> None:
    season = resolve_season()
    prior = resolve_prior_season()
    slate = resolve_slate_date()
    print(f"[runtime] SEASON={season}")
    print(f"[runtime] PRIOR_SEASON={prior}")
    print(f"[runtime] SLATE_DATE={slate or '<empty>'}")
    if TEAM_WEEK_MAP_PATH.exists() and TEAM_WEEK_MAP_PATH.stat().st_size > 0:
        try:
            week = resolve_week(season=season, slate_date=slate)
        except Exception as exc:
            print(f"[runtime] WEEK unresolved: {exc}")
        else:
            print(f"[runtime] WEEK={week}")
