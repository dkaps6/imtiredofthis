#!/usr/bin/env python3
"""Shared runtime season/week resolution for the NFL pipeline.

This module is the runtime source of truth for active season, prior season,
slate date, and NFL week. Week resolution is always based on the authoritative
team-week schedule. Calendar/ISO week numbers are intentionally never used.

Full Slate has one additional production invariant: once its NFL week is
resolved, every downstream process in that workflow must use the same week.
GitHub Actions runs therefore persist the first Full Slate resolution through
``GITHUB_ENV`` as ``NFL_RUNTIME_WEEK``. This prevents a long-running Sunday-night
job from switching weeks merely because the UTC calendar crossed midnight.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from scripts.config import PRIOR_SEASON, SEASON, SLATE_DATE

TEAM_WEEK_MAP_PATH = Path("data/team_week_map.csv")
RUNTIME_WEEK_ENV = "NFL_RUNTIME_WEEK"
NFL_CALENDAR_TZ = "America/New_York"
FULL_SLATE_WORKFLOW_MARKER = ".github/workflows/full-slate.yml@"


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


def _schedule_calendar_date(frame: pd.DataFrame, kickoff: pd.Series) -> pd.Series:
    """Return the NFL schedule's intended local game date for each row.

    ``team_week_map.csv`` is commonly built from nflverse ``gameday``, which is a
    calendar date rather than an exact kickoff instant. Parsing that date as UTC
    midnight and comparing it to wall-clock UTC caused Week 1 to roll into Week 2
    at 00:00 UTC on Sunday night. Preserve date-semantic columns as dates first;
    only fall back to a real kickoff timestamp converted to the NFL calendar zone.
    """
    for candidate in ("slate_date", "gameday", "game_date", "date"):
        if candidate not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[candidate], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed.dt.date
    if kickoff.notna().any():
        return kickoff.dt.tz_convert(NFL_CALENDAR_TZ).dt.date
    return pd.Series([pd.NaT] * len(frame), index=frame.index, dtype=object)


def _weeks_for_date(scoped: pd.DataFrame, target_date) -> list[int]:
    dated = scoped.loc[scoped["_calendar_date"].eq(target_date)].copy()
    if dated.empty:
        return []
    return sorted({int(v) for v in dated["week"].dropna().tolist()})


def _full_slate_should_freeze(path: Path) -> bool:
    if os.getenv("GITHUB_ACTIONS", "").strip().lower() != "true":
        return False
    if not os.getenv("GITHUB_ENV", "").strip():
        return False
    if FULL_SLATE_WORKFLOW_MARKER not in os.getenv("GITHUB_WORKFLOW_REF", ""):
        return False
    return path == TEAM_WEEK_MAP_PATH


def _persist_full_slate_week(week: int, path: Path) -> None:
    """Freeze the first canonical Full Slate week for all later workflow steps."""
    if os.getenv(RUNTIME_WEEK_ENV, "").strip() or not _full_slate_should_freeze(path):
        return
    env_path = Path(os.environ["GITHUB_ENV"])
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{RUNTIME_WEEK_ENV}={int(week)}\n")
    # Keep repeated resolve_week() calls inside this same Python process stable too.
    os.environ[RUNTIME_WEEK_ENV] = str(int(week))
    print(f"[runtime] frozen {RUNTIME_WEEK_ENV}={int(week)} for Full Slate workflow")


def resolve_week(
    season: int | None = None,
    slate_date: str | None = None,
    team_week_map_path: str | Path = TEAM_WEEK_MAP_PATH,
    *,
    now: pd.Timestamp | None = None,
) -> int:
    """Resolve one authoritative NFL week from ``team_week_map.csv``.

    Resolution order:
    1. Scope to the active season and authoritative schedule.
    2. Honor a previously frozen ``NFL_RUNTIME_WEEK`` after validating it against
       that schedule (and an explicit slate date, when supplied).
    3. If a slate date is supplied, match the schedule's local game date.
    4. If slate date is blank, first prefer games on the current NFL local date;
       otherwise choose the nearest upcoming scheduled game week, or the most
       recent completed week if the season is over.
    5. In the canonical Full Slate GitHub workflow, persist the resolved week to
       ``GITHUB_ENV`` so every later process in that run sees the same authority.

    This deliberately never uses ISO/calendar week numbers.
    """
    season = int(season if season is not None else resolve_season())
    slate_date = (slate_date if slate_date is not None else resolve_slate_date()).strip()
    path = Path(team_week_map_path)
    scoped = _load_team_week_map(path, season)
    kickoff = _schedule_timestamp(scoped)
    scoped = scoped.assign(
        _kickoff=kickoff,
        _calendar_date=_schedule_calendar_date(scoped, kickoff),
    )

    frozen_raw = os.getenv(RUNTIME_WEEK_ENV, "").strip()
    if frozen_raw:
        frozen = _as_int(frozen_raw, RUNTIME_WEEK_ENV)
        available = sorted({int(v) for v in scoped["week"].dropna().tolist()})
        if frozen <= 0 or frozen not in available:
            raise RuntimeError(
                f"Frozen {RUNTIME_WEEK_ENV}={frozen} is invalid for season={season}; "
                f"schedule_weeks={available}"
            )
        if slate_date:
            target = pd.to_datetime(slate_date, errors="coerce")
            if pd.isna(target):
                raise RuntimeError(f"Invalid SLATE_DATE={slate_date!r}; expected YYYY-MM-DD")
            dated_weeks = _weeks_for_date(scoped, target.date())
            if frozen not in dated_weeks:
                raise RuntimeError(
                    f"Frozen {RUNTIME_WEEK_ENV}={frozen} conflicts with "
                    f"SLATE_DATE={slate_date}; date_weeks={dated_weeks}"
                )
        return frozen

    if slate_date:
        target = pd.to_datetime(slate_date, errors="coerce")
        if pd.isna(target):
            raise RuntimeError(f"Invalid SLATE_DATE={slate_date!r}; expected YYYY-MM-DD")
        target_date = target.date()
        weeks = _weeks_for_date(scoped, target_date)
        if not weeks:
            raise RuntimeError(
                f"Cannot resolve NFL week: no scheduled games for season={season}, slate_date={slate_date}"
            )
    else:
        current = now if now is not None else pd.Timestamp.now(tz="UTC")
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        else:
            current = current.tz_convert("UTC")

        # NFL game dates are league-local calendar concepts. A Sunday-night run at
        # 20:00 ET is already Monday in UTC, but it is still Sunday's NFL slate.
        nfl_today = current.tz_convert(NFL_CALENDAR_TZ).date()
        weeks = _weeks_for_date(scoped, nfl_today)

        if not weeks:
            valid_time = scoped.loc[scoped["_kickoff"].notna()].copy()
            if valid_time.empty:
                weeks = sorted({int(v) for v in scoped["week"].dropna().tolist()})
                if len(weeks) != 1:
                    raise RuntimeError(
                        "Cannot infer current NFL week from a full-season schedule without kickoff timestamps; "
                        f"candidates={weeks}"
                    )
            else:
                upcoming = valid_time.loc[valid_time["_kickoff"] >= current].sort_values("_kickoff")
                if not upcoming.empty:
                    # Games belonging to a week can span several days. Choose the week
                    # of the nearest upcoming scheduled game.
                    weeks = [int(upcoming.iloc[0]["week"])]
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
    _persist_full_slate_week(week, path)
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