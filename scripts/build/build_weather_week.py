#!/usr/bin/env python3
"""Build a weekly NFL weather dataset using API-Sports and NWS data.

The script performs the following steps:

1. Fetch the full-season schedule from API-Sports using the configured API key.
2. Identify the upcoming week based on kickoff timestamps.
3. Merge the home team schedule with local stadium metadata (roof + lat/lon).
4. Request NWS forecasts (points → gridpoints → forecast/forecastHourly) with
   per-game retry logic capped at 45 seconds.
5. Require at least 16 games to return non-null weather readings; raise a
   RuntimeError if the threshold is not met or if any API step fails.

Output columns (CSV):
team,opponent,week,stadium,roof,forecast_summary,temp_f,wind_mph,precip_prob,forecast_datetime_utc
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


TEAM_CODE: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


DEFAULT_UA = "FullSlate/Weather (+https://github.com/imtiredofthis)"
APISPORTS_URL = "https://v1.american-football.api-sports.io/games"
NWS_HEADERS = {"User-Agent": DEFAULT_UA, "Accept": "application/geo+json"}
APISPORTS_HEADERS = {"User-Agent": DEFAULT_UA, "Accept": "application/json"}

MIN_SUCCESSFUL_FORECASTS = 16
MAX_GAME_WAIT_SECONDS = 45.0


def parse_week(value: object) -> Optional[int]:
    """Extract an integer week identifier from API-Sports payloads."""

    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group())
    except (TypeError, ValueError):
        return None


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


def fetch_apisports_schedule(
    season: int,
    *,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    api_key = os.getenv("APISPORTS_KEY") or os.getenv("API_SPORTS_KEY")
    if not api_key:
        raise RuntimeError("APISPORTS_KEY is required to fetch the NFL schedule")

    sess = session or requests.Session()
    page = 1
    rows = []
    headers = {**APISPORTS_HEADERS, "x-apisports-key": api_key}

    while True:
        try:
            response = sess.get(
                APISPORTS_URL,
                params={"league": "NFL", "season": season, "page": page},
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise RuntimeError(
                f"API-Sports schedule request failed on page {page}: {exc}"
            ) from exc
        for game in payload.get("response", []):
            away_name = game.get("teams", {}).get("away", {}).get("name")
            home_name = game.get("teams", {}).get("home", {}).get("name")
            away = TEAM_CODE.get(away_name)
            home = TEAM_CODE.get(home_name)
            week = parse_week(game.get("week") or game.get("round"))
            kickoff = pd.to_datetime(game.get("date"), utc=True, errors="coerce")
            if away and home and week and pd.notna(kickoff):
                rows.append(
                    {
                        "home": home,
                        "away": away,
                        "week": week,
                        "kickoff_utc": kickoff,
                    }
                )
        paging = payload.get("paging", {})
        if not paging or paging.get("current") >= paging.get("total", 0):
            break
        page += 1
        time.sleep(0.2)

    if not rows:
        raise RuntimeError(f"API-Sports schedule returned no games for season {season}")

    return pd.DataFrame(rows)


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


def main(out_csv: str = "weather.csv") -> None:
    season = infer_default_season()

    with requests.Session() as session:
        games = fetch_apisports_schedule(season, session=session)
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

            if roof != "Dome/Fixed":
                try:
                    forecast = fetch_nws_forecast(
                        float(row.latitude),
                        float(row.longitude),
                        kickoff_utc,
                        session=session,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Failed to fetch NWS forecast for {row.home} at {row.stadium}: {exc}"
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
            "roof",
            "forecast_summary",
            "temp_f",
            "wind_mph",
            "precip_prob",
            "forecast_datetime_utc",
        ],
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else "weather.csv"
    main(destination)
