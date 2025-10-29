"""Shared helpers for loading season schedules from nflverse releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

DEFAULT_HEADERS = {
    "User-Agent": "FullSlate/Weather (+https://github.com/imtiredofthis)",
    "Accept": "text/csv,application/json",
}

_CACHE_DIR = Path("data/_cache")
_BUNDLE_PATH = Path("external/nflverse_bundle/outputs/schedules.csv")
_SOURCES: Iterable[str] = (
    "https://github.com/nflverse/nflverse-data/releases/download/sched/sched_{season}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/sched_{season}.csv",
    "https://raw.githubusercontent.com/nflverse/nflverse-data/master/schedules/sched_{season}.csv",
)


@dataclass
class _ScheduleColumns:
    season: str
    week: str
    home: str
    away: str
    kickoff: Optional[str]
    gameday: Optional[str]
    gametime: Optional[str]
    stadium: Optional[str]
    location: Optional[str]
    city: Optional[str]
    state: Optional[str]


def _detect_columns(df: pd.DataFrame) -> _ScheduleColumns:
    cols = {c.lower(): c for c in df.columns}

    def _find(names: Iterable[str]) -> Optional[str]:
        for name in names:
            if name in cols:
                return cols[name]
        return None

    season = _find(["season", "schedule_season", "year"])
    week = _find(["week", "schedule_week"])
    home = _find(["home_team", "schedule_home_team", "home"])
    away = _find(["away_team", "schedule_away_team", "away"])
    kickoff = _find(
        [
            "start_time_utc",
            "start_time",
            "kickoff",
            "commence_time",
            "game_time",
            "game_datetime",
            "gamedatetime",
        ]
    )
    gameday = _find(["gameday", "game_date", "gamedate"])
    gametime = _find(["gametime", "game_time", "kickoff_time", "start_time_local"])
    stadium = _find(["stadium", "site", "venue"])
    location = _find(["location", "city_state", "citystate"])
    city = _find(["city", "home_city", "site_city"])
    state = _find(["state", "home_state", "site_state"])

    if not (season and week and home and away):
        missing = [name for name, value in {
            "season": season,
            "week": week,
            "home": home,
            "away": away,
        }.items() if value is None]
        raise RuntimeError(
            f"Schedule CSV missing required columns: {', '.join(missing)}"
        )

    return _ScheduleColumns(
        season=season,
        week=week,
        home=home,
        away=away,
        kickoff=kickoff,
        gameday=gameday,
        gametime=gametime,
        stadium=stadium,
        location=location,
        city=city,
        state=state,
    )


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or len(df.columns) == 0:
        return None
    return df


def _normalize_team(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def _parse_location(
    df: pd.DataFrame,
    columns: _ScheduleColumns,
) -> pd.DataFrame:
    location_series = (
        df[columns.location]
        if columns.location and columns.location in df.columns
        else pd.Series(["" for _ in range(len(df))])
    )
    city_series = (
        df[columns.city]
        if columns.city and columns.city in df.columns
        else pd.Series(["" for _ in range(len(df))])
    )
    state_series = (
        df[columns.state]
        if columns.state and columns.state in df.columns
        else pd.Series(["" for _ in range(len(df))])
    )

    location_series = location_series.fillna("").astype(str)
    city_series = city_series.fillna("").astype(str)
    state_series = state_series.fillna("").astype(str)

    def split_location(value: str) -> tuple[str, str]:
        if not value:
            return "", ""
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        tokens = value.split()
        if len(tokens) >= 2:
            return " ".join(tokens[:-1]), tokens[-1]
        return value, ""

    derived_city = []
    derived_state = []
    for loc, city, state in zip(location_series, city_series, state_series, strict=False):
        c = city.strip()
        s = state.strip()
        if not c or not s:
            derived_c, derived_s = split_location(loc)
            c = c or derived_c
            s = s or derived_s
        derived_city.append(c)
        derived_state.append(s)

    return pd.DataFrame({
        "location": location_series,
        "city": pd.Series(derived_city).str.strip(),
        "state": pd.Series(derived_state).str.strip().str.upper(),
    })


def _coerce_kickoff(df: pd.DataFrame, columns: _ScheduleColumns) -> pd.Series:
    if columns.kickoff and columns.kickoff in df.columns:
        kickoff = pd.to_datetime(df[columns.kickoff], utc=True, errors="coerce")
        if kickoff.notna().any():
            return kickoff

    if columns.gameday and columns.gameday in df.columns:
        gameday = pd.to_datetime(df[columns.gameday], utc=True, errors="coerce")
    else:
        gameday = pd.Series([pd.NaT] * len(df))

    if columns.gametime and columns.gametime in df.columns:
        combo = (
            df[columns.gameday].astype(str).str.strip()
            if columns.gameday and columns.gameday in df.columns
            else gameday.dt.strftime("%Y-%m-%d")
        )
        times = df[columns.gametime].astype(str).str.strip()
        kickoff = pd.to_datetime(combo + " " + times, utc=True, errors="coerce")
        if kickoff.notna().any():
            return kickoff

    if columns.gameday and columns.gameday in df.columns:
        return pd.to_datetime(df[columns.gameday], utc=True, errors="coerce")

    raise RuntimeError("Unable to determine kickoff timestamps from schedule CSV")


def _load_from_sources(season: int, session: Optional[requests.Session]) -> pd.DataFrame:
    cache_path = _CACHE_DIR / f"sched_{season}.csv"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        cached = _read_csv(cache_path)
        if cached is not None:
            return cached

    sess = session or requests.Session()
    close_session = session is None
    sess.headers.update(DEFAULT_HEADERS)

    last_error: Optional[Exception] = None
    try:
        for template in _SOURCES:
            url = template.format(season=season)
            try:
                response = sess.get(url, timeout=45)
                response.raise_for_status()
                content = response.content
                if not content.strip():
                    raise RuntimeError("empty response body")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(content)
                df = pd.read_csv(cache_path)
                if df.empty:
                    raise RuntimeError("downloaded schedule CSV is empty")
                return df
            except Exception as exc:  # noqa: PERF203 - gather final error
                last_error = exc
                cache_path.unlink(missing_ok=True)
                continue
    finally:
        if close_session:
            sess.close()

    if last_error:
        raise RuntimeError(
            f"Unable to download nflverse schedule for season {season}: {last_error}"
        )
    raise RuntimeError(f"Unable to download nflverse schedule for season {season}")


def get_nfl_schedule(
    season: int,
    *,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Return schedule rows for ``season`` normalized to home/away/week/kickoff."""

    if season is None:
        raise RuntimeError("Season is required for nfl schedule lookup")

    bundle_df = None
    if _BUNDLE_PATH.exists() and _BUNDLE_PATH.stat().st_size > 0:
        bundle_df = _read_csv(_BUNDLE_PATH)

    if bundle_df is None:
        raw_df = _load_from_sources(season, session)
    else:
        raw_df = bundle_df

    if raw_df is None or raw_df.empty:
        raise RuntimeError("Schedule source returned no rows")

    columns = _detect_columns(raw_df)

    df = raw_df.copy()
    df = df.rename(columns={columns.season: "season_marker"})
    season_series = pd.to_numeric(df["season_marker"], errors="coerce").astype("Int64")
    df = df[season_series == season]
    if df.empty:
        raise RuntimeError(f"No schedule rows found for season {season}")

    location_df = _parse_location(df, columns)
    kickoff_series = _coerce_kickoff(df, columns)

    week_series = pd.to_numeric(df[columns.week], errors="coerce").astype("Int64")

    result = pd.DataFrame(
        {
            "season": season,
            "week": week_series,
            "home": _normalize_team(df[columns.home]),
            "away": _normalize_team(df[columns.away]),
            "kickoff_utc": kickoff_series,
            "stadium": (
                df[columns.stadium] if columns.stadium and columns.stadium in df.columns else location_df["location"]
            ).fillna("").astype(str).str.strip(),
        }
    )

    result = pd.concat([result, location_df], axis=1)
    result = result.dropna(subset=["week", "kickoff_utc"])
    result["week"] = result["week"].astype(int)
    result["kickoff_utc"] = pd.to_datetime(result["kickoff_utc"], utc=True, errors="coerce")
    result = result[result["kickoff_utc"].notna()].reset_index(drop=True)

    if result.empty:
        raise RuntimeError(f"Schedule normalization produced no rows for season {season}")

    return result
