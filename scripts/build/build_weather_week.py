#!/usr/bin/env python3
"""Build a weekly NFL weather dataset using nflverse schedules and NWS data.

The script performs the following steps:

1. Fetch the full-season schedule from nflverse releases (season-filtered).
2. Identify the upcoming week based on kickoff timestamps.
3. Merge the home team schedule with local stadium metadata (roof + lat/lon).
4. Request NWS forecasts (points → forecastHourly) with per-game retry logic
   capped at 45 seconds. Forecasts prefer the city/state location metadata and
   fall back to latitude/longitude.
5. Require at least 16 games to return non-null weather readings; raise a
   RuntimeError if the threshold is not met or if any API step fails.

Output columns (CSV):
team,opponent,week,stadium,location,city,state,roof,forecast_summary,temp_f,wind_mph,precip_prob,forecast_datetime_utc
"""

from __future__ import annotations

import re
import sys
import time
from datetime import timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from urllib.parse import quote

import pandas as pd
import requests

from scripts.build._schedule_utils import get_nfl_schedule


DEFAULT_UA = "FullSlate/Weather (+https://github.com/imtiredofthis)"
NWS_HEADERS = {"User-Agent": DEFAULT_UA, "Accept": "application/geo+json"}

MIN_SUCCESSFUL_FORECASTS = 16
MAX_GAME_WAIT_SECONDS = 45.0
OUTPUT_PATH = Path("data") / "weather_week.csv"


def infer_default_season(now: Optional[pd.Timestamp] = None) -> int:
    """Return a reasonable season for schedule lookups."""

    now = now or pd.Timestamp.utcnow()
    return now.year if now.month >= 3 else now.year - 1


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
        except requests.RequestException as exc:  # pragma: no cover - network
            last_error = exc
            attempt += 1
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


def load_stadium_metadata(path: Path) -> pd.DataFrame:
    """Load stadium metadata with columns team/stadium/roof/latitude/longitude."""

    if not path.exists():
        raise RuntimeError(f"Stadium metadata file missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Stadium metadata file {path} is empty")

    required = {"team", "stadium", "roof"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Stadium metadata missing columns: {', '.join(sorted(missing))}"
        )

    lat_col = None
    lon_col = None
    for candidate in df.columns:
        lowered = candidate.lower()
        if lowered in {"lat", "latitude"}:
            lat_col = candidate
        elif lowered in {"lon", "long", "longitude"}:
            lon_col = candidate
    if not lat_col or not lon_col:
        raise RuntimeError("Stadium metadata requires latitude/longitude columns")

    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    df = df.dropna(subset=["team", "stadium", "roof", "latitude", "longitude"])
    df["team"] = df["team"].astype(str).str.upper()
    return df[["team", "stadium", "roof", "latitude", "longitude"]]


def roof_normalize(roof: str) -> str:
    value = str(roof).lower()
    if "retract" in value:
        return "Retractable"
    if any(token in value for token in ["dome", "translucent", "fixed"]):
        return "Dome/Fixed"
    return "Open"


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

    with requests.Session() as session:
        games = get_nfl_schedule(season, session=session)
        week_games = upcoming_week_games(games)

        stadiums = load_stadium_metadata(Path("data") / "stadiums.csv")
        merged = week_games.merge(
            stadiums,
            left_on="home",
            right_on="team",
            how="left",
            validate="many_to_one",
        )

        if merged["stadium"].isna().any():
            missing = merged[merged["stadium"].isna()]["home"].unique()
            raise RuntimeError(
                f"Missing stadium metadata for teams: {', '.join(sorted(missing))}"
            )

        for column in ("location", "city", "state"):
            if column not in merged.columns:
                merged[column] = ""

        out_rows = []
        successful = 0

        for row in merged.itertuples(index=False):
            roof = roof_normalize(row.roof)
            kickoff_ts = pd.Timestamp(row.kickoff_utc)
            if kickoff_ts.tzinfo is None:
                kickoff_utc = kickoff_ts.tz_localize(timezone.utc)
            else:
                kickoff_utc = kickoff_ts.tz_convert(timezone.utc)
            weather_summary = ""
            temp_f = wind_mph = precip_prob = None

            raw_location = getattr(row, "location", "")
            raw_city = getattr(row, "city", "")
            raw_state = getattr(row, "state", "")
            location_value = "" if pd.isna(raw_location) else str(raw_location).strip()
            city_value = "" if pd.isna(raw_city) else str(raw_city).strip()
            state_value = "" if pd.isna(raw_state) else str(raw_state).strip()

            if roof != "Dome/Fixed":
                forecast = None
                city_error: Optional[Exception] = None

                if city_value and state_value:
                    try:
                        periods = get_forecast_for_city(
                            city_value,
                            state_value,
                            session=session,
                            deadline=time.monotonic() + MAX_GAME_WAIT_SECONDS,
                        )
                        forecast = _forecast_from_periods(periods, kickoff_utc)
                    except RuntimeError as exc:
                        city_error = exc

                if forecast is None:
                    try:
                        forecast = fetch_nws_forecast(
                            float(row.latitude),
                            float(row.longitude),
                            kickoff_utc,
                            session=session,
                        )
                    except RuntimeError as exc:
                        context = (
                            f"; city/state lookup failed: {city_error}"
                            if city_error
                            else ""
                        )
                        raise RuntimeError(
                            f"Failed to fetch NWS forecast for {row.home} at {row.stadium}{context}: {exc}"
                        ) from exc

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
                    "stadium": row.stadium,
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
        raise RuntimeError(
            f"Only {successful} game forecasts returned weather data (<{MIN_SUCCESSFUL_FORECASTS} required)"
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
        raise RuntimeError("Weather builder produced empty DataFrame.")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_PATH
    main(destination)
