#
# Build per-game weather for this week's slate and write data/weather_week.csv.
# - Uses stadium_locations.STADIUM_LOCATION to map home team -> stadium/city/state/outdoor/lat/lon.
# - Uses National Weather Service (NWS) hourly forecasts via the points API.
# - Summarizes temp/wind/precip around kickoff.
# - Fails fast (RuntimeError) if we cannot generate at least 1 row.

from __future__ import annotations  # defer evaluation of type hints

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import os
import re
from datetime import datetime, date
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from scripts._opponent_map import canon_team
from scripts.utils.stadium_locations import STADIUM_LOCATION

STADIUM_LOCATION.setdefault(
    "WAS",
    {
        "city": "Landover",
        "state": "MD",
        "stadium": "Commanders Field",
        "indoor": False,
        "lat": 38.9076,
        "lon": -76.8645,
    },
)

OUT_PATH = Path("data") / "weather_week.csv"

NWS_HEADERS = {
    "User-Agent": "imtiredofthis-weather (+https://github.com/imtiredofthis)",
    "Accept": "application/geo+json",
}

SESSION = requests.Session()
logger = logging.getLogger(__name__)


def _canon_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(canon_team)


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    """Coerce mixed datetime/epoch inputs to timezone-aware UTC timestamps."""

    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    numeric = pd.to_numeric(series, errors="coerce")
    dt_numeric = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    dt_series = pd.to_datetime(series, utc=True, errors="coerce")

    return dt_numeric.combine_first(dt_series)


def _parse_kickoff_et_value(raw, season=None, week=None, gameday=None) -> pd.Timestamp:
    """Parse a kickoff string expressed in Eastern Time and return a tz-aware Timestamp."""

    text = "" if raw is None else str(raw).strip()
    if text.upper() in {"", "NAN", "NAT"}:
        text = ""

    if text:
        cleaned = re.sub(r"\s*ET$", "", text, flags=re.IGNORECASE)
        ts = pd.to_datetime(cleaned, errors="coerce")
        if pd.notna(ts):
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
            return ts

    gameday_ts = pd.to_datetime(gameday, errors="coerce", utc=True)
    if pd.isna(gameday_ts):
        gameday_ts = pd.to_datetime(gameday, errors="coerce")
    ref = None
    if pd.notna(gameday_ts):
        if gameday_ts.tzinfo is None:
            gameday_ts = gameday_ts.tz_localize("UTC")
        ref = gameday_ts.tz_convert("America/New_York")

    if ref is None:
        try:
            season_val = int(season)
        except Exception:
            season_val = datetime.utcnow().year
        try:
            week_val = int(week)
        except Exception:
            week_val = 1
        if week_val < 1:
            week_val = 1
        base = pd.Timestamp(f"{season_val}-09-01", tz="America/New_York")
        base_sunday = base + pd.offsets.Week(weekday=6)
        ref = base_sunday + pd.Timedelta(weeks=week_val - 1)

    delta_days = (ref.weekday() - 6) % 7
    sunday = ref - pd.Timedelta(days=delta_days)
    return sunday.replace(hour=13, minute=0, second=0, microsecond=0)


def _normalize_team_week_map(team_week: pd.DataFrame) -> pd.DataFrame:
    """Normalize team_week_map rows that provide home_team/away_team/kickoff_et."""

    if team_week is None or team_week.empty:
        return pd.DataFrame()

    cols = {c.lower(): c for c in team_week.columns}
    home_col = cols.get("home_team")
    away_col = cols.get("away_team")
    if not (home_col and away_col):
        return pd.DataFrame()

    season_col = cols.get("season")
    week_col = cols.get("week")
    gameday_col = cols.get("gameday")
    local_tz_col = next(
        (cols[key] for key in ("local_tz", "tz", "timezone") if key in cols),
        None,
    )
    kickoff_local_col = next(
        (cols[key] for key in ("kickoff_local",) if key in cols),
        None,
    )
    kickoff_et_col = cols.get("kickoff_et")
    kickoff_utc_col = next(
        (cols[key] for key in ("kickoff_utc", "commence_time") if key in cols),
        None,
    )

    out = pd.DataFrame(
        {
            "home": _canon_series(team_week[home_col].astype(str)),
            "away": _canon_series(team_week[away_col].astype(str)),
        }
    )

    out["home"] = out["home"].astype(str).str.upper().str.strip()
    out["away"] = out["away"].astype(str).str.upper().str.strip()
    out = out.replace({"": pd.NA, "NAN": pd.NA}).dropna(subset=["home", "away"]).copy()

    local_tz_values = []
    kickoff_local_values = []
    kickoff_utc_values = []
    kickoff_et_strings = []

    for idx in out.index:
        home_team = out.at[idx, "home"]
        tz_name = ""
        if local_tz_col and local_tz_col in team_week.columns:
            tz_name = str(team_week.at[idx, local_tz_col] or "").strip()
        if not tz_name:
            tz_name = _infer_tz_from_team(home_team)
        try:
            tzinfo = ZoneInfo(tz_name)
        except Exception:
            fallback_tz = _infer_tz_from_team(home_team)
            tzinfo = ZoneInfo(fallback_tz)
            tz_name = fallback_tz

        season_val = team_week.at[idx, season_col] if season_col else None
        week_val = team_week.at[idx, week_col] if week_col else None
        gameday_val = team_week.at[idx, gameday_col] if gameday_col else None

        kickoff_local_ts = pd.NaT
        kickoff_et_ts = pd.NaT

        if kickoff_local_col and kickoff_local_col in team_week.columns:
            raw_local = team_week.at[idx, kickoff_local_col]
            ts_local = pd.to_datetime(raw_local, errors="coerce")
            if pd.notna(ts_local):
                if ts_local.tzinfo is None:
                    try:
                        ts_local = ts_local.tz_localize(tzinfo)
                    except Exception:
                        ts_local = ts_local.tz_localize("UTC").tz_convert(tzinfo)
                else:
                    try:
                        ts_local = ts_local.tz_convert(tzinfo)
                    except Exception:
                        ts_local = ts_local.tz_localize(tzinfo)
                kickoff_local_ts = ts_local
                kickoff_et_ts = ts_local.tz_convert("America/New_York")

        if pd.isna(kickoff_local_ts) and kickoff_et_col and kickoff_et_col in team_week.columns:
            raw_et = team_week.at[idx, kickoff_et_col]
            kickoff_et_ts = _parse_kickoff_et_value(raw_et, season_val, week_val, gameday_val)
            if kickoff_et_ts is not None:
                kickoff_local_ts = kickoff_et_ts.tz_convert(tzinfo)

        if pd.isna(kickoff_local_ts) and kickoff_utc_col and kickoff_utc_col in team_week.columns:
            raw_utc = team_week.at[idx, kickoff_utc_col]
            ts_utc = pd.to_datetime(raw_utc, utc=True, errors="coerce")
            if pd.notna(ts_utc):
                kickoff_local_ts = ts_utc.tz_convert(tzinfo)
                kickoff_et_ts = ts_utc.tz_convert("America/New_York")

        if pd.isna(kickoff_local_ts):
            kickoff_et_ts = _parse_kickoff_et_value("", season_val, week_val, gameday_val)
            kickoff_local_ts = kickoff_et_ts.tz_convert(tzinfo)

        if pd.isna(kickoff_et_ts):
            kickoff_et_strings.append("")
        else:
            kickoff_et_strings.append(
                kickoff_et_ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S ET")
            )

        if pd.isna(kickoff_local_ts):
            kickoff_local_values.append(pd.NaT)
            kickoff_utc_values.append(pd.NaT)
        else:
            kickoff_local_values.append(kickoff_local_ts)
            kickoff_utc_values.append(kickoff_local_ts.tz_convert("UTC"))

        local_tz_values.append(tz_name)

    out["local_tz"] = local_tz_values
    out["kickoff_local"] = kickoff_local_values
    out["kickoff_utc"] = pd.to_datetime(kickoff_utc_values, utc=True, errors="coerce")
    out["kickoff_et"] = kickoff_et_strings

    if season_col and season_col in team_week.columns:
        try:
            out["season"] = pd.to_numeric(team_week[season_col], errors="coerce").astype("Int64")
        except Exception:
            out["season"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["season"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if week_col and week_col in team_week.columns:
        try:
            out["week"] = pd.to_numeric(team_week[week_col], errors="coerce").astype("Int64")
        except Exception:
            out["week"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["week"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    return out.dropna(subset=["home", "away"]).reset_index(drop=True)


def build_slate_fallback() -> pd.DataFrame:
    """Build a minimal slate when the usual slate CSV is missing/empty."""

    fallback_path = Path("data") / "opponent_map_from_props.csv"
    if not fallback_path.exists():
        return pd.DataFrame()

    try:
        props_df = pd.read_csv(fallback_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()

    if props_df.empty:
        return pd.DataFrame()

    cols = {col.lower(): col for col in props_df.columns}
    team_col = next(
        (
            cols[key]
            for key in cols
            if key in {"team", "team_abbr", "team_code", "team_name"}
        ),
        None,
    )
    opp_col = next(
        (
            cols[key]
            for key in cols
            if key in {"opponent", "opp", "opponent_abbr", "opp_team", "opp_name"}
        ),
        None,
    )

    if not team_col or not opp_col:
        return pd.DataFrame()

    matchups = {}
    for _, row in props_df.iterrows():
        if pd.isna(row.get(team_col)) or pd.isna(row.get(opp_col)):
            continue

        team_val = str(row[team_col]).strip().upper()
        opp_val = str(row[opp_col]).strip().upper()

        if not team_val or not opp_val or team_val == "NAN" or opp_val == "NAN":
            continue

        pair_key = tuple(sorted([team_val, opp_val]))
        if pair_key not in matchups:
            matchups[pair_key] = {"home": team_val, "away": opp_val}

    if not matchups:
        return pd.DataFrame()

    slate_date_raw = os.getenv("SLATE_DATE", "").strip()
    date_obj = None
    if slate_date_raw:
        try:
            date_obj = datetime.fromisoformat(slate_date_raw).date()
        except ValueError:
            date_obj = None
    if date_obj is None:
        date_obj = datetime.now(tz=ZoneInfo("UTC")).date()

    rows = []
    for _, teams in matchups.items():
        home_team = teams["home"]
        away_team = teams["away"]

        local_tz = _infer_tz_from_team(home_team)
        try:
            tzinfo = ZoneInfo(local_tz)
        except Exception:
            tzinfo = ZoneInfo("UTC")
            local_tz = "UTC"

        kickoff_local = datetime(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            hour=13,
            minute=0,
            tzinfo=tzinfo,
        )

        rows.append(
            {
                "home": home_team,
                "away": away_team,
                "kickoff_raw": kickoff_local.isoformat(),
                "local_tz": local_tz,
                "kickoff_utc": kickoff_local.astimezone(ZoneInfo("UTC")),
                "kickoff_local": kickoff_local,
                "game_date": date_obj.isoformat(),
                "slate_date": date_obj.isoformat(),
                "stadium": "",
                "dome_flag": False,
                "fallback_source": "opponent_map_from_props",
            }
        )

    fallback_df = pd.DataFrame(rows).drop_duplicates(subset=["home", "away"])
    fallback_df.attrs["fallback_used"] = True
    fallback_df.attrs["source"] = "fallback"
    return fallback_df


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("[weather] failed reading %s: %s", path, exc)
        return pd.DataFrame()


def _load_team_week_source(season: int | None) -> pd.DataFrame:
    df = _read_csv_optional(Path("data/team_week_map.csv"))
    if df.empty:
        return pd.DataFrame()
    normalized = _normalize_team_week_map(df)
    if normalized.empty:
        return normalized
    if season is not None and "season" in normalized.columns:
        mask = normalized["season"].fillna(season).astype("Int64") == int(season)
        normalized = normalized.loc[mask].copy()
    normalized.attrs["source"] = "team_week_map"
    return normalized


def _load_odds_game_source() -> pd.DataFrame:
    df = _read_csv_optional(Path("data/odds_game.csv"))
    if df.empty:
        df = _read_csv_optional(Path("outputs/odds_game.csv"))
    if df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    home_col = next((cols[key] for key in ("home_team", "home") if key in cols), None)
    away_col = next((cols[key] for key in ("away_team", "away") if key in cols), None)
    kickoff_col = next(
        (cols[key] for key in ("commence_time", "kickoff_ts", "kickoff_utc") if key in cols),
        None,
    )
    if not (home_col and away_col):
        return pd.DataFrame()
    working = pd.DataFrame(
        {
            "home": _canon_series(df[home_col]),
            "away": _canon_series(df[away_col]),
        }
    )
    if kickoff_col:
        working["kickoff_utc"] = pd.to_datetime(df[kickoff_col], utc=True, errors="coerce")
    else:
        working["kickoff_utc"] = pd.NaT
    if "event_id" in cols:
        working["event_id"] = df[cols["event_id"]]
    working.attrs["source"] = "odds_game"
    return working


def _load_schedule_source(season: int | None) -> pd.DataFrame:
    df = _read_csv_optional(Path("data/schedule.csv"))
    if df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    home_col = next((cols[key] for key in ("home", "home_team") if key in cols), None)
    away_col = next((cols[key] for key in ("away", "away_team") if key in cols), None)
    kickoff_col = next(
        (cols[key] for key in ("kickoff_utc", "kickoff", "start_time", "commence_time") if key in cols),
        None,
    )
    date_col = next((cols[key] for key in ("gameday", "game_date", "date") if key in cols), None)
    if not (home_col and away_col):
        return pd.DataFrame()
    working = pd.DataFrame(
        {
            "home": _canon_series(df[home_col]),
            "away": _canon_series(df[away_col]),
        }
    )
    if kickoff_col:
        working["kickoff_utc"] = pd.to_datetime(df[kickoff_col], utc=True, errors="coerce")
    else:
        working["kickoff_utc"] = pd.NaT
    if date_col:
        working["kickoff_date"] = pd.to_datetime(df[date_col], errors="coerce")
    if season is not None and "season" in cols:
        season_series = pd.to_numeric(df[cols["season"]], errors="coerce").astype("Int64")
        working = working.loc[season_series == int(season)].copy()
    working.attrs["source"] = "schedule"
    return working


def _nearest_game_date(df: pd.DataFrame, slate_date: str | None) -> Optional[date]:
    if df is None or df.empty:
        return None

    slate: Optional[date] = None
    if slate_date:
        try:
            cleaned = slate_date.strip()
        except AttributeError:
            cleaned = ""
        if cleaned:
            try:
                slate = datetime.strptime(cleaned, "%Y-%m-%d").date()
            except ValueError:
                slate = None
    if slate:
        return slate
    if "kickoff_utc" in df.columns and df["kickoff_utc"].notna().any():
        kickoff = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce").dropna()
        if not kickoff.empty:
            now = pd.Timestamp.utcnow()
            upcoming = kickoff[kickoff >= now]
            target = upcoming.min() if not upcoming.empty else kickoff.max()
            if pd.notna(target):
                return target.date()
    if "kickoff_local" in df.columns and df["kickoff_local"].notna().any():
        local = pd.to_datetime(df["kickoff_local"], errors="coerce").dropna()
        if not local.empty:
            now_local = pd.Timestamp.utcnow().tz_localize("UTC")
            upcoming = local[local >= now_local]
            target = upcoming.min() if not upcoming.empty else local.max()
            if pd.notna(target):
                target = target.tz_convert("UTC") if target.tzinfo else target
                return target.date()
    if "kickoff_date" in df.columns and df["kickoff_date"].notna().any():
        dates = pd.to_datetime(df["kickoff_date"], errors="coerce").dropna()
        if not dates.empty:
            today = pd.Timestamp.utcnow().date()
            upcoming_dates = dates[dates.dt.date >= today]
            target = upcoming_dates.min() if not upcoming_dates.empty else dates.max()
            if pd.notna(target):
                return target.date()
    return None


def _filter_games_by_date(df: pd.DataFrame, target_date: Optional[date]) -> pd.DataFrame:
    if target_date is None or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    if "kickoff_utc" in df.columns:
        utc_dates = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce").dt.date
        mask = mask | (utc_dates == target_date)
    if "kickoff_local" in df.columns:
        local_dates = pd.to_datetime(df["kickoff_local"], errors="coerce").dt.date
        mask = mask | (local_dates == target_date)
    if "kickoff_date" in df.columns:
        base_dates = pd.to_datetime(df["kickoff_date"], errors="coerce").dt.date
        mask = mask | (base_dates == target_date)
    filtered = df.loc[mask.fillna(False)].copy()
    return filtered if not filtered.empty else df


def _ensure_local_kickoff(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    kickoff_local: list = []
    kickoff_utc: list = []
    tz_values: list = []
    for _, row in df.iterrows():
        home = canon_team(row.get("home"))
        tz_name = row.get("local_tz") or _infer_tz_from_team(home)
        tzinfo = ZoneInfo(tz_name)
        local_raw = pd.to_datetime(row.get("kickoff_local"), errors="coerce")
        utc_raw = pd.to_datetime(row.get("kickoff_utc"), utc=True, errors="coerce")
        date_raw = row.get("kickoff_date")
        if pd.notna(local_raw):
            if local_raw.tzinfo is None:
                local_raw = local_raw.tz_localize(tzinfo)
            else:
                local_raw = local_raw.tz_convert(tzinfo)
        elif pd.notna(utc_raw):
            local_raw = utc_raw.tz_convert(tzinfo)
        elif date_raw is not None and str(date_raw).strip():
            date_ts = pd.to_datetime(date_raw, errors="coerce")
            if pd.notna(date_ts):
                if date_ts.tzinfo is None:
                    date_ts = date_ts.tz_localize(tzinfo)
                else:
                    date_ts = date_ts.tz_convert(tzinfo)
                local_raw = date_ts.replace(hour=12, minute=0, second=0, microsecond=0)
        if pd.isna(utc_raw) and pd.notna(local_raw):
            utc_raw = local_raw.astimezone(ZoneInfo("UTC"))
        kickoff_local.append(local_raw)
        kickoff_utc.append(utc_raw)
        tz_values.append(tz_name)
    working = df.copy()
    working["kickoff_local"] = kickoff_local
    working["kickoff_utc"] = kickoff_utc
    working["local_tz"] = tz_values
    working["kickoff_date"] = [
        (val.date() if isinstance(val, datetime) else (utc.date() if pd.notna(utc) else pd.NaT))
        for val, utc in zip(kickoff_local, kickoff_utc)
    ]
    return working


def _load_slate_from_repo(season: int | None, slate_date: str | None) -> pd.DataFrame:
    loaders = [
        lambda: _load_team_week_source(season),
        _load_odds_game_source,
        lambda: _load_schedule_source(season),
    ]

    for loader in loaders:
        df = loader()
        if df.empty:
            continue
        df = _ensure_local_kickoff(df)
        target_date = _nearest_game_date(df, slate_date)
        df = _filter_games_by_date(df, target_date)
        df = _ensure_local_kickoff(df)
        if df.empty:
            continue
        return df

    fallback_df = build_slate_fallback()
    if not fallback_df.empty:
        return _ensure_local_kickoff(fallback_df)

    raise RuntimeError(
        "Could not load a slate for weather forecasts. (team_week_map/odds_game/schedule missing)"
    )

def _infer_tz_from_team(team: str) -> str:
    """
    Best-effort guess of local timezone for a team's stadium.
    """

    tz_map = {
        "SEA": "America/Los_Angeles",
        "SF": "America/Los_Angeles",
        "LAR": "America/Los_Angeles",
        "LAC": "America/Los_Angeles",
        "ARI": "America/Phoenix",
        "DEN": "America/Denver",
        "KC": "America/Chicago",
        "CHI": "America/Chicago",
        "GB": "America/Chicago",
        "MIN": "America/Chicago",
        "DAL": "America/Chicago",
        "HOU": "America/Chicago",
        "NO": "America/Chicago",
        "TEN": "America/Chicago",
        "TB": "America/New_York",
        "ATL": "America/New_York",
        "CAR": "America/New_York",
        "MIA": "America/New_York",
        "BUF": "America/New_York",
        "NE": "America/New_York",
        "NYJ": "America/New_York",
        "NYG": "America/New_York",
        "PHI": "America/New_York",
        "WAS": "America/New_York",
        "WSH": "America/New_York",
        "BAL": "America/New_York",
        "PIT": "America/New_York",
        "CLE": "America/New_York",
        "CIN": "America/New_York",
        "DET": "America/Detroit",
        "IND": "America/Indiana/Indianapolis",
        "JAX": "America/New_York",
        "LV": "America/Los_Angeles",
    }
    return tz_map.get(team, "America/New_York")


def _parse_wind_speed(raw: str):
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None

    return sum(numbers) / len(numbers)


def _get_hourly_forecast(lat: float, lon: float) -> pd.DataFrame:
    """Fetch hourly forecast periods from the NWS API."""

    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    points_resp = SESSION.get(points_url, headers=NWS_HEADERS, timeout=15)
    points_resp.raise_for_status()
    points_json = points_resp.json()

    properties = points_json.get("properties", {})
    forecast_url = properties.get("forecastHourly")
    if not forecast_url:
        raise RuntimeError(f"[weather] No forecastHourly URL for {lat},{lon}")

    forecast_resp = SESSION.get(forecast_url, headers=NWS_HEADERS, timeout=15)
    forecast_resp.raise_for_status()
    forecast_json = forecast_resp.json()

    periods = forecast_json.get("properties", {}).get("periods", [])
    rows = []
    for period in periods:
        start_time = period.get("startTime")
        if not start_time:
            continue

        try:
            ts_utc = pd.to_datetime(start_time, utc=True)
        except Exception:
            continue

        precip_raw = None
        pop_obj = period.get("probabilityOfPrecipitation")
        if isinstance(pop_obj, dict):
            precip_raw = pop_obj.get("value")
        elif pop_obj is not None:
            precip_raw = pop_obj

        pop_val = None
        if precip_raw is not None:
            try:
                pop_val = float(precip_raw) / 100.0
            except (TypeError, ValueError):
                pop_val = None

        rows.append(
            {
                "ts_utc": ts_utc,
                "temp_F": float(period.get("temperature"))
                if period.get("temperature") is not None
                else None,
                "wind_mph": _parse_wind_speed(period.get("windSpeed")),
                "weather_main": period.get("shortForecast"),
                "weather_desc": period.get("detailedForecast")
                or period.get("shortForecast"),
                "pop_rain": pop_val,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def _summarize_window(
    forecast_df: pd.DataFrame, kickoff_local_dt: datetime, local_tz: str
) -> dict:
    """
    Slice forecast around kickoff_local_dt ... kickoff_local_dt+4h (stadium local time),
    summarize temp / wind / precip.
    """

    if forecast_df.empty:
        return {}

    f = forecast_df.copy()
    f["ts_local"] = f["ts_utc"].dt.tz_convert(ZoneInfo(local_tz))

    start = kickoff_local_dt
    end = kickoff_local_dt + pd.Timedelta(hours=4)

    window = f[(f["ts_local"] >= start) & (f["ts_local"] <= end)]
    if window.empty:
        window = f[f["ts_local"].dt.date == start.date()]

    if window.empty:
        return {}

    def _mode(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    return {
        "temp_F_mean": window["temp_F"].mean(),
        "temp_F_min": window["temp_F"].min(),
        "temp_F_max": window["temp_F"].max(),
        "wind_mph_mean": window["wind_mph"].mean(),
        "precip_prob_max": window["pop_rain"].max(),
        "conditions_main": _mode(window["weather_main"]),
        "conditions_desc": _mode(window["weather_desc"]),
    }


def _kickoff_iso(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return str(value)


def _base_weather_row(
    home_team: str, away_team: str, meta: dict | None
) -> dict:
    stadium = meta.get("stadium", "") if meta else ""
    city = meta.get("city", "") if meta else ""
    state = meta.get("state", "") if meta else ""
    indoor = None if not meta else (not meta.get("outdoor", True))

    return {
        "home": home_team,
        "home_team": home_team,
        "away": away_team,
        "away_team": away_team,
        "stadium": stadium,
        "city": city,
        "state": state,
        "kickoff_local": "",
        "kickoff_utc": "",
        "slate_date": "",
        "indoor": indoor,
        "forecast_ok": 0,
        "temp_F_mean": None,
        "temp_F_min": None,
        "temp_F_max": None,
        "wind_mph_mean": None,
        "precip_prob_max": None,
        "conditions_main": None,
        "conditions_desc": None,
        "cold_flag": None,
        "wind_flag": None,
        "rain_flag": None,
        "blurb": "",
    }


def _weather_row_for_game(row: pd.Series) -> dict:
    """Build weather summary for a single game row."""

    home_team = str(row.get("home", "")).upper().strip()
    away_team = str(row.get("away", "")).upper().strip()
    kickoff_value = row.get("kickoff_local")
    slate_date_value = row.get("slate_date") or row.get("game_date")
    if not slate_date_value:
        slate_date_value = os.getenv("SLATE_DATE", "").strip()

    meta = STADIUM_LOCATION.get(home_team)
    base_row = _base_weather_row(home_team, away_team, meta)

    if not meta:
        logger.warning("[weather] No stadium mapping for home team %s", home_team)
        return base_row

    lat = meta.get("lat")
    lon = meta.get("lon")
    if lat is None or lon is None:
        logger.warning(
            "[weather] Missing coordinates for %s (%s) — skipping forecast",
            home_team,
            meta.get("stadium", "unknown stadium"),
        )
        return base_row

    kickoff_local_dt = kickoff_value
    if isinstance(kickoff_local_dt, pd.Timestamp):
        kickoff_local_dt = kickoff_local_dt.to_pydatetime()
    elif not isinstance(kickoff_local_dt, datetime):
        kickoff_local_ts = pd.to_datetime(kickoff_local_dt, errors="coerce")
        if pd.isna(kickoff_local_ts):
            logger.warning("[weather] Unusable kickoff_local for %s", home_team)
            return base_row
        tz_guess = row.get("local_tz") or _infer_tz_from_team(home_team)
        try:
            tzinfo = ZoneInfo(tz_guess)
        except Exception:
            tzinfo = ZoneInfo("UTC")
        if kickoff_local_ts.tzinfo is None:
            kickoff_local_ts = kickoff_local_ts.tz_localize(tzinfo)
        else:
            kickoff_local_ts = kickoff_local_ts.tz_convert(tzinfo)
        kickoff_local_dt = kickoff_local_ts.to_pydatetime()

    if kickoff_local_dt:
        base_row["kickoff_local"] = kickoff_local_dt.isoformat()
        base_row["kickoff_utc"] = kickoff_local_dt.astimezone(ZoneInfo("UTC")).isoformat()
        if not slate_date_value:
            slate_date_value = kickoff_local_dt.date().isoformat()

    base_row["slate_date"] = str(slate_date_value or "")

    if kickoff_local_dt is None or kickoff_local_dt.tzinfo is None:
        logger.warning("[weather] Kickoff datetime missing tz for %s", home_team)
        return base_row

    tz_name = row.get("local_tz")
    if not tz_name:
        tzinfo = kickoff_local_dt.tzinfo
        tz_name = getattr(tzinfo, "key", None) or tzinfo.tzname(kickoff_local_dt) or "UTC"

    try:
        fc_df = _get_hourly_forecast(lat, lon)
    except Exception as exc:
        logger.warning(
            "[weather] Forecast fetch failed for %s (%s,%s): %s",
            home_team,
            lat,
            lon,
            exc,
        )
        return base_row

    summary = _summarize_window(
        fc_df,
        kickoff_local_dt=kickoff_local_dt,
        local_tz=tz_name,
    )

    if not summary:
        logger.warning("[weather] No forecast window data for %s", home_team)
        return base_row

    out = {**base_row, **summary}
    out["forecast_ok"] = 1

    temp_mean = out.get("temp_F_mean")
    wind_mean = out.get("wind_mph_mean")
    precip_max = out.get("precip_prob_max")

    out["cold_flag"] = 1 if (pd.notna(temp_mean) and temp_mean < 40) else 0
    out["wind_flag"] = 1 if (pd.notna(wind_mean) and wind_mean >= 15) else 0
    out["rain_flag"] = 1 if (pd.notna(precip_max) and precip_max >= 0.4) else 0

    indoor = out.get("indoor")
    if indoor:
        out["blurb"] = "Indoors/retractable - neutral conditions"
    else:
        bits = []
        if pd.notna(temp_mean):
            bits.append(f"{round(temp_mean)}F avg")
        if pd.notna(wind_mean):
            bits.append(f"wind ~{round(wind_mean, 1)} mph")
        if out["rain_flag"]:
            bits.append("rain risk")
        if out.get("conditions_desc"):
            bits.append(str(out["conditions_desc"]))
        out["blurb"] = ", ".join(bits)

    return out


def main(season: str | None = None, slate_date: str | None = None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    env_season = (season or os.getenv("SEASON", "").strip()) or None
    season_value: int | None = None
    if env_season:
        try:
            season_value = int(env_season)
        except ValueError:
            season_value = None

    env_slate = slate_date if slate_date is not None else os.getenv("SLATE_DATE", "")
    env_slate = (env_slate or "").strip() or None

    try:
        slate_df = _load_slate_from_repo(season_value, env_slate)
    except RuntimeError as exc:
        logger.warning("[weather] %s", exc)
        slate_df = pd.DataFrame()

    if slate_df.empty:
        logger.warning(
            "[weather] Slate is empty even after fallback; writing empty weather file"
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print(f"[weather] wrote 0 games -> {OUT_PATH} (fallback_used=False)")
        return

    source_label = slate_df.attrs.get("source", "unknown")
    logger.info("[weather] slate source: %s", source_label)

    kickoff_local_raw = slate_df.get("kickoff_local")
    usable_kickoff = 0
    if kickoff_local_raw is not None:
        kickoff_local_series = pd.to_datetime(kickoff_local_raw, errors="coerce")
        if isinstance(kickoff_local_series, pd.Series):
            usable_kickoff = int(kickoff_local_series.notna().sum())
        else:
            usable_kickoff = int(pd.notna(kickoff_local_series))
    logger.info("[weather] kickoff_local rows: %s/%s", usable_kickoff, len(slate_df))

    fallback_used = bool(slate_df.attrs.get("fallback_used", False))

    rows = []
    for _, game in slate_df.iterrows():
        wxrow = _weather_row_for_game(game)
        rain_flag = wxrow.get("rain_flag")
        wxrow["temp_f"] = wxrow.get("temp_F_mean")
        wxrow["wind_mph"] = wxrow.get("wind_mph_mean")
        wxrow["precip_flag"] = None if rain_flag is None else (1 if rain_flag else 0)
        wxrow["notes"] = "weather_ok" if wxrow.get("forecast_ok") else "weather_unavailable"

        rows.append(wxrow)

    weather_df = pd.DataFrame(rows).drop_duplicates()

    if weather_df.empty:
        logger.warning("[weather] No weather rows generated; writing empty file")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print(
            f"[weather] wrote 0 games -> {OUT_PATH} (fallback_used={fallback_used})"
        )
        return

    if weather_df["forecast_ok"].fillna(0).sum() == 0:
        logger.warning(
            "[weather] forecast_ok=0 for all games; writing results for transparency"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUT_PATH, index=False)
    print(
        f"[weather] wrote {len(weather_df)} games -> {OUT_PATH} (fallback_used={fallback_used})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    main(season=args.season, slate_date=args.date)
