#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STADIUM_LOCATION: Dict[str, Dict[str, object]] = {
    "BUF": {
        "stadium": "Highmark Stadium",
        "city": "Orchard Park",
        "state": "NY",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "MIA": {
        "stadium": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "state": "FL",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "NE": {
        "stadium": "Gillette Stadium",
        "city": "Foxborough",
        "state": "MA",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "NYJ": {
        "stadium": "MetLife Stadium",
        "city": "East Rutherford",
        "state": "NJ",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "CIN": {
        "stadium": "Paycor Stadium",
        "city": "Cincinnati",
        "state": "OH",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "CLE": {
        "stadium": "Cleveland Browns Stadium",
        "city": "Cleveland",
        "state": "OH",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "PIT": {
        "stadium": "Acrisure Stadium",
        "city": "Pittsburgh",
        "state": "PA",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "BAL": {
        "stadium": "M&T Bank Stadium",
        "city": "Baltimore",
        "state": "MD",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "HOU": {
        "stadium": "NRG Stadium",
        "city": "Houston",
        "state": "TX",
        "outdoor": False,
        "tz": "America/Chicago",
    },
    "IND": {
        "stadium": "Lucas Oil Stadium",
        "city": "Indianapolis",
        "state": "IN",
        "outdoor": False,
        "tz": "America/Indiana/Indianapolis",
    },
    "JAX": {
        "stadium": "EverBank Stadium",
        "city": "Jacksonville",
        "state": "FL",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "TEN": {
        "stadium": "Nissan Stadium",
        "city": "Nashville",
        "state": "TN",
        "outdoor": True,
        "tz": "America/Chicago",
    },
    "KC": {
        "stadium": "GEHA Field at Arrowhead Stadium",
        "city": "Kansas City",
        "state": "MO",
        "outdoor": True,
        "tz": "America/Chicago",
    },
    "LAC": {
        "stadium": "SoFi Stadium",
        "city": "Inglewood",
        "state": "CA",
        "outdoor": False,
        "tz": "America/Los_Angeles",
    },
    "LV": {
        "stadium": "Allegiant Stadium",
        "city": "Las Vegas",
        "state": "NV",
        "outdoor": False,
        "tz": "America/Los_Angeles",
    },
    "DEN": {
        "stadium": "Empower Field at Mile High",
        "city": "Denver",
        "state": "CO",
        "outdoor": True,
        "tz": "America/Denver",
    },
    "DAL": {
        "stadium": "AT&T Stadium",
        "city": "Arlington",
        "state": "TX",
        "outdoor": False,
        "tz": "America/Chicago",
    },
    "PHI": {
        "stadium": "Lincoln Financial Field",
        "city": "Philadelphia",
        "state": "PA",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "WSH": {
        "stadium": "FedEx Field",
        "city": "Landover",
        "state": "MD",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "NYG": {
        "stadium": "MetLife Stadium",
        "city": "East Rutherford",
        "state": "NJ",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "GB": {
        "stadium": "Lambeau Field",
        "city": "Green Bay",
        "state": "WI",
        "outdoor": True,
        "tz": "America/Chicago",
    },
    "DET": {
        "stadium": "Ford Field",
        "city": "Detroit",
        "state": "MI",
        "outdoor": False,
        "tz": "America/Detroit",
    },
    "CHI": {
        "stadium": "Soldier Field",
        "city": "Chicago",
        "state": "IL",
        "outdoor": True,
        "tz": "America/Chicago",
    },
    "MIN": {
        "stadium": "U.S. Bank Stadium",
        "city": "Minneapolis",
        "state": "MN",
        "outdoor": False,
        "tz": "America/Chicago",
    },
    "ATL": {
        "stadium": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "state": "GA",
        "outdoor": False,
        "tz": "America/New_York",
    },
    "NO": {
        "stadium": "Caesars Superdome",
        "city": "New Orleans",
        "state": "LA",
        "outdoor": False,
        "tz": "America/Chicago",
    },
    "TB": {
        "stadium": "Raymond James Stadium",
        "city": "Tampa",
        "state": "FL",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "CAR": {
        "stadium": "Bank of America Stadium",
        "city": "Charlotte",
        "state": "NC",
        "outdoor": True,
        "tz": "America/New_York",
    },
    "SF": {
        "stadium": "Levi's Stadium",
        "city": "Santa Clara",
        "state": "CA",
        "outdoor": True,
        "tz": "America/Los_Angeles",
    },
    "SEA": {
        "stadium": "Lumen Field",
        "city": "Seattle",
        "state": "WA",
        "outdoor": True,
        "tz": "America/Los_Angeles",
    },
    "LAR": {
        "stadium": "SoFi Stadium",
        "city": "Inglewood",
        "state": "CA",
        "outdoor": False,
        "tz": "America/Los_Angeles",
    },
    "ARI": {
        "stadium": "State Farm Stadium",
        "city": "Glendale",
        "state": "AZ",
        "outdoor": False,
        "tz": "America/Phoenix",
    },
}

FALLBACK_STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "BUF": (42.7738, -78.7869),
    "MIA": (25.9579, -80.2389),
    "NE": (42.0909, -71.2643),
    "NYJ": (40.8135, -74.0745),
    "CIN": (39.0954, -84.5161),
    "CLE": (41.5061, -81.6995),
    "PIT": (40.4468, -80.0158),
    "BAL": (39.278, -76.6227),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.324, -81.6373),
    "TEN": (36.1665, -86.7713),
    "KC": (39.049, -94.4839),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.183),
    "DEN": (39.7439, -105.0201),
    "DAL": (32.7473, -97.0945),
    "PHI": (39.9008, -75.1675),
    "WSH": (38.9077, -76.8645),
    "NYG": (40.8135, -74.0745),
    "GB": (44.5013, -88.0622),
    "DET": (42.3398, -83.0456),
    "CHI": (41.8623, -87.6167),
    "MIN": (44.974, -93.258),
    "ATL": (33.7554, -84.4008),
    "NO": (29.9509, -90.0814),
    "TB": (27.9759, -82.5033),
    "CAR": (35.2258, -80.8528),
    "SF": (37.403, -121.969),
    "SEA": (47.5952, -122.3316),
    "LAR": (33.9535, -118.3392),
    "ARI": (33.5277, -112.2626),
}

CITY_NORMALIZATION: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("Miami Gardens", "FL"): ("Miami", "FL"),
    ("Orchard Park", "NY"): ("Buffalo", "NY"),
    ("Foxborough", "MA"): ("Boston", "MA"),
    ("East Rutherford", "NJ"): ("Newark", "NJ"),
    ("Inglewood", "CA"): ("Los Angeles", "CA"),
    ("Santa Clara", "CA"): ("San Jose", "CA"),
    ("Glendale", "AZ"): ("Phoenix", "AZ"),
    ("Landover", "MD"): ("Hyattsville", "MD"),
    ("Arlington", "TX"): ("Dallas", "TX"),
}


class NotFoundError(RuntimeError):
    """Raised when an upstream request returns HTTP 404."""


__doc__ = """Build a weekly NFL weather dataset using nflverse schedules and NWS data.

The script performs the following steps:

1. Fetch the full-season schedule from nflverse releases (season-filtered).
2. Identify the upcoming week based on kickoff timestamps.
3. Attach static stadium metadata (roof/outdoor + timezone) from an in-repo map.
4. Request NWS forecasts (points → forecastHourly) with per-game retry logic
   capped at 45 seconds. Forecasts prefer the city/state location metadata and
   fall back to latitude/longitude when available.
5. Require at least 16 games to return non-null weather readings; raise a
   RuntimeError if the threshold is not met or if any API step fails.

Output columns (CSV):
team,opponent,week,stadium,location,city,state,roof,forecast_summary,temp_f,wind_mph,precip_prob,forecast_datetime_utc
"""

import re
import time
from datetime import timezone
from zoneinfo import ZoneInfo

from urllib.parse import quote

import pandas as pd
import requests
from nfl_data_py import import_schedules


DEFAULT_UA = "FullSlate/Weather (+https://github.com/imtiredofthis)"
NWS_HEADERS = {"User-Agent": DEFAULT_UA, "Accept": "application/geo+json"}

MIN_SUCCESSFUL_FORECASTS = 16
MAX_GAME_WAIT_SECONDS = 45.0
OUTPUT_PATH = Path("data") / "weather_week.csv"
WEATHER_COLUMNS = [
    "team",
    "opponent",
    "week",
    "stadium",
    "location",
    "city",
    "state",
    "roof",
    "forecast_summary",
    "temp_f",
    "wind_mph",
    "precip_prob",
    "forecast_datetime_utc",
]


def get_nfl_schedule(season: int) -> pd.DataFrame:
    df = import_schedules([season])
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG"])]
    rename_map = {
        "home_team": "team_home",
        "away_team": "team_away",
        "game_date": "gameday",
        "venue": "stadium",
        "site_city": "location",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    required = ["week", "team_home", "team_away"]
    missing = [c for c in required if c not in df.columns]
    if missing or df.empty:
        raise RuntimeError(
            f"Schedule missing columns {missing} or empty for season {season}"
        )
    keep = [
        "week",
        "team_home",
        "team_away",
    ] + [c for c in ["game_id", "gameday", "stadium", "location"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def infer_default_season(now: Optional[pd.Timestamp] = None) -> int:
    """Return a reasonable season for schedule lookups."""

    now = now or pd.Timestamp.utcnow()
    return now.year if now.month >= 3 else now.year - 1


def _normalize_schedule_for_weather(
    schedule: pd.DataFrame, season: int
) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        raise RuntimeError("Schedule DataFrame is empty; cannot build weather slate")

    df = schedule.copy()
    if "team_home" not in df.columns or "team_away" not in df.columns:
        raise RuntimeError("Schedule missing team_home/team_away columns")

    df["season"] = season
    df["home"] = df["team_home"].astype(str).str.upper().str.strip()
    df["away"] = df["team_away"].astype(str).str.upper().str.strip()

    kickoff = pd.Series(pd.NaT, index=df.index)
    if "gameday" in df.columns:
        kickoff = pd.to_datetime(df["gameday"], utc=True, errors="coerce")

    if kickoff.isna().all():
        raise RuntimeError("Schedule missing usable gameday values for kickoff inference")

    df["kickoff_utc"] = kickoff

    for column in ("stadium", "location", "city", "state"):
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["state"] = df["state"].str.upper()
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df[df["week"].notna()]
    df = df[df["kickoff_utc"].notna()]

    if df.empty:
        raise RuntimeError("Normalized schedule produced no usable rows")

    result = df[
        [
            "season",
            "week",
            "home",
            "away",
            "kickoff_utc",
            "stadium",
            "location",
            "city",
            "state",
        ]
    ].reset_index(drop=True)

    return result


def request_json_with_retry(
    session: requests.Session,
    url: str,
    *,
    headers: Dict[str, str],
    params: Optional[Dict[str, object]] = None,
    deadline: float,
    label: str,
) -> Dict[str, object]:
    """GET `url` returning JSON with retries until `deadline` seconds."""

    attempt = 0
    last_error: Optional[Exception] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout = min(15.0, max(1.0, remaining))
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:  # pragma: no cover - network
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                raise NotFoundError(
                    f"NWS {label} request returned 404 for {url}"
                ) from exc
            last_error = exc
            attempt += 1
        except requests.RequestException as exc:  # pragma: no cover - network
            last_error = exc
            attempt += 1
        else:
            continue

        sleep_for = min(2**attempt, remaining)
        if sleep_for <= 0:
            break
        time.sleep(sleep_for)
    error_detail = last_error or "deadline exceeded"
    raise RuntimeError(f"NWS {label} request failed after retries: {error_detail}")


def _forecast_from_periods(
    periods: Iterable[Dict[str, object]],
    kickoff_utc: pd.Timestamp,
) -> Dict[str, Optional[object]]:
    rows = []
    for period in periods:
        start_time = pd.to_datetime(period.get("startTime"), utc=True, errors="coerce")
        if pd.isna(start_time):
            continue
        temp_f = period.get("temperature")
        wind = period.get("windSpeed")
        precip = period.get("probabilityOfPrecipitation", {}).get("value")
        summary = period.get("shortForecast") or period.get("name") or ""
        wind_values = re.findall(r"[-+]?[0-9]*\.?[0-9]+", str(wind or ""))
        wind_mph = float(wind_values[-1]) if wind_values else None
        rows.append(
            {
                "time": start_time,
                "temp_f": temp_f,
                "wind_mph": wind_mph,
                "precip_prob": precip,
                "summary": summary,
            }
        )

    if not rows:
        raise RuntimeError("NWS forecast response returned no usable periods")

    df_periods = pd.DataFrame(rows)
    idx = (df_periods["time"] - kickoff_utc).abs().idxmin()
    period_row = df_periods.loc[idx]

    return {
        "forecast_summary": period_row.get("summary", ""),
        "temp_f": period_row.get("temp_f"),
        "wind_mph": period_row.get("wind_mph"),
        "precip_prob": period_row.get("precip_prob"),
    }


def fetch_nws_forecast(
    lat: float,
    lon: float,
    kickoff_utc: pd.Timestamp,
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Optional[object]]:
    """Return nearest forecast period (summary/temp/wind/precip) for kickoff."""

    sess = session or requests.Session()
    deadline = time.monotonic() + MAX_GAME_WAIT_SECONDS

    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    points = request_json_with_retry(
        sess,
        points_url,
        headers=NWS_HEADERS,
        deadline=deadline,
        label="points",
    )

    props = points.get("properties", {})
    cwa = props.get("cwa")
    grid_x = props.get("gridX")
    grid_y = props.get("gridY")
    forecast_url = props.get("forecastHourly") or props.get("forecast")
    if not (cwa and grid_x is not None and grid_y is not None and forecast_url):
        raise RuntimeError("NWS points response missing grid/forecast metadata")

    grid_url = f"https://api.weather.gov/gridpoints/{cwa}/{grid_x},{grid_y}"
    request_json_with_retry(
        sess,
        grid_url,
        headers=NWS_HEADERS,
        deadline=deadline,
        label="gridpoints",
    )

    forecast = request_json_with_retry(
        sess,
        forecast_url,
        headers=NWS_HEADERS,
        deadline=deadline,
        label="forecast",
    )

    periods = forecast.get("properties", {}).get("periods", [])
    return _forecast_from_periods(periods, kickoff_utc)


def get_forecast_for_city(
    city: str,
    state: str,
    *,
    session: Optional[requests.Session] = None,
    deadline: Optional[float] = None,
) -> Iterable[Dict[str, object]]:
    """Return NWS forecast periods for a given ``city`` and ``state``.

    The function queries the ``points`` endpoint followed by ``forecastHourly``
    and raises ``RuntimeError`` if either request fails or if the periods array
    is empty.
    """

    city_value = str(city or "").strip()
    state_value = str(state or "").strip()
    if not city_value or not state_value:
        raise RuntimeError("City and state are required for NWS lookup")

    sess = session or requests.Session()
    close_session = session is None

    try:
        expiry = deadline if deadline is not None else time.monotonic() + MAX_GAME_WAIT_SECONDS
        encoded_city = quote(city_value)
        encoded_state = quote(state_value)
        points_url = f"https://api.weather.gov/points/{encoded_city},{encoded_state}"
        points = request_json_with_retry(
            sess,
            points_url,
            headers=NWS_HEADERS,
            deadline=expiry,
            label="points",
        )

        props = points.get("properties", {})
        forecast_url = props.get("forecastHourly")
        if not forecast_url:
            raise RuntimeError("NWS points response missing forecastHourly URL")

        forecast = request_json_with_retry(
            sess,
            forecast_url,
            headers=NWS_HEADERS,
            deadline=expiry,
            label="forecast",
        )

        periods = forecast.get("properties", {}).get("periods") or []
        if not periods:
            raise RuntimeError("NWS forecast response returned no periods")
        return periods
    finally:
        if close_session:
            sess.close()


def roof_normalize(roof: str) -> str:
    value = str(roof).lower()
    if "retract" in value:
        return "Retractable"
    if any(token in value for token in ["dome", "translucent", "fixed"]):
        return "Dome/Fixed"
    return "Open"


def _write_weather_output(df: pd.DataFrame | None, out_path: Path) -> None:
    """Write ``df`` to ``out_path`` ensuring headers exist."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        empty_df = pd.DataFrame(columns=WEATHER_COLUMNS)
        empty_df.to_csv(out_path, index=False)
        print(f"[builder WARNING] wrote headers only (0 rows) -> {out_path}")
        return

    out_df = df.copy()
    missing = [col for col in WEATHER_COLUMNS if col not in out_df.columns]
    for col in missing:
        out_df[col] = pd.NA

    ordered = WEATHER_COLUMNS + [
        col for col in out_df.columns if col not in WEATHER_COLUMNS
    ]
    out_df = out_df[ordered]
    out_df.to_csv(out_path, index=False)
    print(f"[builder] wrote {len(out_df)} rows -> {out_path}")


def upcoming_week_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        raise RuntimeError("No games available to determine an upcoming week")

    now = pd.Timestamp.utcnow()
    window = games[games["kickoff_utc"] >= now - pd.Timedelta(days=1)]
    if window.empty:
        window = games[games["kickoff_utc"] >= now]
    if window.empty:
        raise RuntimeError("Unable to locate upcoming kickoff window from schedule")

    window = window.sort_values("kickoff_utc")
    week = int(window.iloc[0]["week"])
    slate = games[games["week"] == week].copy()
    if slate.empty:
        raise RuntimeError(f"No games found for upcoming week {week}")
    return slate


def main(out_csv: str | Path = OUTPUT_PATH) -> None:
    season = infer_default_season()

    schedule = get_nfl_schedule(season)
    games = _normalize_schedule_for_weather(schedule, season)
    week_games = upcoming_week_games(games)

    with requests.Session() as session:
        out_rows = []
        successful = 0

        for row in week_games.itertuples(index=False):
            try:
                metadata = STADIUM_LOCATION[row.home]
            except KeyError as exc:  # pragma: no cover - configuration
                raise RuntimeError(
                    f"Missing stadium metadata for team: {row.home}"
                ) from exc

            schedule_stadium = (
                "" if pd.isna(row.stadium) else str(row.stadium).strip()
            )
            schedule_location = (
                "" if pd.isna(row.location) else str(row.location).strip()
            )
            schedule_city = "" if pd.isna(row.city) else str(row.city).strip()
            schedule_state = (
                "" if pd.isna(row.state) else str(row.state).strip().upper()
            )

            meta_stadium = str(metadata.get("stadium", "")).strip()
            meta_city = str(metadata.get("city", "")).strip()
            meta_state = str(metadata.get("state", "")).strip().upper()
            meta_location = f"{meta_city}, {meta_state}".strip(", ")

            stadium_name = schedule_stadium or meta_stadium
            city_value = schedule_city or meta_city
            state_value = schedule_state or meta_state
            location_value = schedule_location or meta_location

            roof_value = "Open" if metadata.get("outdoor") else "Dome/Fixed"
            roof = roof_normalize(roof_value)

            tz_name = str(metadata.get("tz", "")).strip() or None
            tzinfo = None
            if tz_name:
                try:
                    tzinfo = ZoneInfo(tz_name)
                except Exception:  # pragma: no cover - invalid tz config
                    tzinfo = None

            kickoff_ts = pd.Timestamp(row.kickoff_utc)
            if kickoff_ts.tzinfo is None:
                if tzinfo is not None:
                    kickoff_local = kickoff_ts.tz_localize(tzinfo)
                    kickoff_utc = kickoff_local.tz_convert(timezone.utc)
                else:
                    kickoff_utc = kickoff_ts.tz_localize(timezone.utc)
            else:
                kickoff_utc = kickoff_ts.tz_convert(timezone.utc)

            weather_summary = ""
            temp_f = wind_mph = precip_prob = None

            if roof != "Dome/Fixed":
                forecast = None
                city_error_msg: Optional[str] = None
                latlon_error_msg: Optional[str] = None

                if city_value and state_value:
                    normalized_city, normalized_state = CITY_NORMALIZATION.get(
                        (city_value, state_value),
                        (city_value, state_value),
                    )
                    try:
                        periods = get_forecast_for_city(
                            normalized_city,
                            normalized_state,
                            session=session,
                            deadline=time.monotonic() + MAX_GAME_WAIT_SECONDS,
                        )
                        forecast = _forecast_from_periods(periods, kickoff_utc)
                    except RuntimeError as exc:
                        city_error_msg = str(exc)

                if forecast is None:
                    lat = metadata.get("latitude")
                    lon = metadata.get("longitude")
                    if lat is None or lon is None:
                        coords = FALLBACK_STADIUM_COORDS.get(row.home)
                        if coords:
                            lat, lon = coords

                    if lat is not None and lon is not None:
                        try:
                            forecast = fetch_nws_forecast(
                                float(lat),
                                float(lon),
                                kickoff_utc,
                                session=session,
                            )
                        except RuntimeError as exc:
                            latlon_error_msg = str(exc)
                    else:
                        latlon_error_msg = (
                            latlon_error_msg or "missing latitude/longitude fallback"
                        )

                if forecast is None:
                    details = "; ".join(
                        msg for msg in [city_error_msg, latlon_error_msg] if msg
                    )
                    detail_suffix = f" ({details})" if details else ""
                    print(
                        "[builder WARNING] NWS forecast unavailable for "
                        f"{row.home} at {stadium_name}{detail_suffix}"
                    )
                else:
                    weather_summary = str(forecast.get("forecast_summary", ""))
                    temp_f = forecast.get("temp_f")
                    wind_mph = forecast.get("wind_mph")
                    precip_prob = forecast.get("precip_prob")
                    if (
                        temp_f is not None
                        or wind_mph is not None
                        or precip_prob is not None
                    ):
                        successful += 1
            else:
                weather_summary = ""

            out_rows.append(
                {
                    "team": row.home,
                    "opponent": row.away,
                    "week": int(row.week),
                    "stadium": stadium_name,
                    "location": location_value,
                    "city": city_value,
                    "state": state_value,
                    "roof": roof,
                    "forecast_summary": weather_summary,
                    "temp_f": temp_f,
                    "wind_mph": wind_mph,
                    "precip_prob": precip_prob,
                    "forecast_datetime_utc": kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

    if successful < MIN_SUCCESSFUL_FORECASTS:
        print(
            "[builder WARNING] weather_week.csv low sample size "
            f"({successful} successful forecasts < {MIN_SUCCESSFUL_FORECASTS}), writing partial output anyway"
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "team",
            "opponent",
            "week",
            "stadium",
            "location",
            "city",
            "state",
            "roof",
            "forecast_summary",
            "temp_f",
            "wind_mph",
            "precip_prob",
            "forecast_datetime_utc",
        ],
    )

    if out_df.empty:
        print(
            "[builder WARNING] weather_week.csv produced 0 rows; writing header-only output"
        )

    total_games_played = len(week_games) if "week_games" in locals() else 0
    min_dynamic = max(2000, total_games_played * 150)
    if len(out_df) < min_dynamic:
        print(
            f"[builder WARNING] weather_week.csv low sample size ({len(out_df)} rows < {min_dynamic}), writing partial output anyway"
        )

    out_path = Path(out_csv)
    _write_weather_output(out_df, out_path)


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_PATH
    main(destination)
