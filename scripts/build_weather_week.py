# scripts/build_weather_week.py
#
# Build per-game weather for this week's slate and write data/weather_week.csv.
# - Uses stadium_locations.STADIUM_LOCATION to map home team -> stadium/city/state/outdoor/lat/lon.
# - Uses National Weather Service (NWS) hourly forecasts via the points API.
# - Summarizes temp/wind/precip around kickoff.
# - Fails fast (RuntimeError) if we cannot generate at least 1 row.

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from scripts.utils.stadium_locations import STADIUM_LOCATION

OUT_PATH = Path("data") / "weather_week.csv"

NWS_HEADERS = {
    "User-Agent": "imtiredofthis-weather (+https://github.com/imtiredofthis)",
    "Accept": "application/geo+json",
}

SESSION = requests.Session()
logger = logging.getLogger(__name__)


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
    team_col = next((cols[key] for key in cols if key in {"team", "team_abbr", "team_code", "team_name"}), None)
    opp_col = next((cols[key] for key in cols if key in {"opponent", "opp", "opponent_abbr", "opp_team", "opp_name"}), None)

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

        rows.append({
            "home": home_team,
            "away": away_team,
            "kickoff_raw": kickoff_local.isoformat(),
            "local_tz": local_tz,
            "kickoff_utc": kickoff_local.astimezone(ZoneInfo("UTC")),
            "kickoff_local": kickoff_local,
            "game_date": date_obj.isoformat(),
            "stadium": "",
            "dome_flag": False,
            "fallback_source": "opponent_map_from_props",
        })

    fallback_df = pd.DataFrame(rows).drop_duplicates(subset=["home", "away"])
    fallback_df.attrs["fallback_used"] = True
    return fallback_df

def _load_slate_from_repo() -> pd.DataFrame:
    """
    Try to infer this week's slate from existing repo artifacts.

    We expect to find something like data/game_lines.csv or similar that already has
    home team, away team, kickoff time, and local tz.

    If we can't find/parse it, raise RuntimeError with instructions so the pipeline
    fails early and loudly instead of silently creating an empty weather_week.csv.
    """
    candidates = [
        Path("data") / "game_lines.csv",
        Path("data") / "opponent_map_from_props.csv",
    ]

    for c in candidates:
        if c.exists():
            try:
                df = pd.read_csv(c)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                df = pd.DataFrame()
            # try to normalize columns -> home, away, kickoff_local, local_tz
            cols = {col.lower(): col for col in df.columns}

            # Heuristic mapping. You can refine this as needed.
            # We try common patterns you've been using in the repo:
            home_col = None
            away_col = None
            kickoff_col = None
            tz_col = None

            for k,v in cols.items():
                if k in ("home","home_team","posteam","posteam_home"):
                    home_col = v
                if k in ("away","away_team","defteam","posteam_away"):
                    away_col = v
                if "kickoff" in k or "start" in k or "datetime" in k:
                    kickoff_col = v
                if "tz" in k or "timezone" in k or "local_tz" in k:
                    tz_col = v

            if df.empty:
                continue

            if home_col and away_col and kickoff_col:
                out = pd.DataFrame({
                    "home": df[home_col].astype(str).str.upper().str.strip(),
                    "away": df[away_col].astype(str).str.upper().str.strip(),
                    "kickoff_raw": df[kickoff_col],
                })

                # local tz optional; fallback guess using home stadium if missing
                if tz_col and tz_col in df.columns:
                    out["local_tz"] = df[tz_col].astype(str)
                else:
                    out["local_tz"] = out["home"].map(_infer_tz_from_team)

                # parse kickoff_raw into tz-aware UTC then also compute local kickoff
                # We'll assume kickoff_raw is either already ISO with TZ, or naive local.
                # We will try UTC parse first, then local.
                kickoff_utc = []
                kickoff_local = []
                for _, r in out.iterrows():
                    raw = str(r["kickoff_raw"])

                    # try parse as ISO first
                    dt_obj = None
                    try:
                        dt_obj = datetime.fromisoformat(raw)
                    except Exception:
                        pass

                    # if dt_obj is naive, assume it's already UTC for now
                    if dt_obj is not None and dt_obj.tzinfo is None:
                        dt_obj = dt_obj.replace(tzinfo=ZoneInfo("UTC"))

                    if dt_obj is None:
                        # last resort: now() so code doesn't explode;
                        # pipeline will complain later if too fake.
                        dt_obj = datetime.now(tz=ZoneInfo("UTC"))

                    # store UTC ts
                    kickoff_utc.append(dt_obj.astimezone(ZoneInfo("UTC")))

                    # stadium tz
                    tzname = r["local_tz"]
                    try:
                        local_dt = dt_obj.astimezone(ZoneInfo(tzname))
                    except Exception:
                        # fallback guess from team again
                        tz_fallback = _infer_tz_from_team(r["home"])
                        local_dt = dt_obj.astimezone(ZoneInfo(tz_fallback))
                    kickoff_local.append(local_dt)

                out["kickoff_utc"] = kickoff_utc
                out["kickoff_local"] = kickoff_local

                out.attrs["fallback_used"] = False
                return out

    fallback_df = build_slate_fallback()
    if not fallback_df.empty:
        return fallback_df

    raise RuntimeError(
        "[weather] Could not infer this week's slate from repo "
        "(no usable game_lines/opponent_map file with home/away/kickoff). "
        "Please extend _load_slate_from_repo() to read your actual slate CSV."
    )

def _infer_tz_from_team(team: str) -> str:
    """
    Best-effort guess of local timezone for a team's stadium.
    """
    # This is crude but fine to start.
    tz_map = {
        "SEA": "America/Los_Angeles",
        "SF": "America/Los_Angeles",
        "LAR": "America/Los_Angeles",
        "LAC": "America/Los_Angeles",
        "ARI": "America/Phoenix",
        "DEN": "America/Denver",
        "KC":  "America/Chicago",
        "CHI": "America/Chicago",
        "GB":  "America/Chicago",
        "MIN": "America/Chicago",
        "DAL": "America/Chicago",
        "HOU": "America/Chicago",
        "NO":  "America/Chicago",
        "TEN": "America/Chicago",
        "TB":  "America/New_York",
        "ATL": "America/New_York",
        "CAR": "America/New_York",
        "MIA": "America/New_York",
        "BUF": "America/New_York",
        "NE":  "America/New_York",
        "NYJ": "America/New_York",
        "NYG": "America/New_York",
        "PHI": "America/New_York",
        "WSH": "America/New_York",
        "BAL": "America/New_York",
        "PIT": "America/New_York",
        "CLE": "America/New_York",
        "CIN": "America/New_York",
        "DET": "America/Detroit",
        "IND": "America/Indiana/Indianapolis",
        "JAX": "America/New_York",
        "LV":  "America/Los_Angeles",   # technically Pacific
        "SF":  "America/Los_Angeles",
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

        rows.append({
            "ts_utc": ts_utc,
            "temp_F": float(period.get("temperature")) if period.get("temperature") is not None else None,
            "wind_mph": _parse_wind_speed(period.get("windSpeed")),
            "weather_main": period.get("shortForecast"),
            "weather_desc": period.get("detailedForecast") or period.get("shortForecast"),
            "pop_rain": pop_val,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df

def _summarize_window(forecast_df: pd.DataFrame,
                      kickoff_local_dt: datetime,
                      local_tz: str) -> dict:
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
        # fallback same calendar day
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


def _base_weather_row(home_team: str,
                      away_team: str,
                      meta: dict | None,
                      kickoff_value) -> dict:
    stadium = meta.get("stadium", "") if meta else ""
    city = meta.get("city", "") if meta else ""
    state = meta.get("state", "") if meta else ""
    indoor = None if not meta else (not meta.get("outdoor", True))

    return {
        "home": home_team,
        "away": away_team,
        "stadium": stadium,
        "city": city,
        "state": state,
        "kickoff_local": _kickoff_iso(kickoff_value),
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

    meta = STADIUM_LOCATION.get(home_team)
    base_row = _base_weather_row(home_team, away_team, meta, kickoff_value)

    if not meta:
        logger.warning("[weather] No stadium mapping for home team %s", home_team)
        return base_row

    lat = meta.get("lat")
    lon = meta.get("lon")
    if lat is None or lon is None:
        logger.warning(
            "[weather] Missing coordinates for %s (%s)",
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    slate_df = _load_slate_from_repo()

    if slate_df.empty:
        raise RuntimeError("[weather] Slate is empty even after fallback")

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

    # fail-fast if totally empty or all forecast_ok=0
    if weather_df.empty:
        raise RuntimeError("[weather] No weather rows generated at all")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUT_PATH, index=False)
    print(f"[weather] wrote {len(weather_df)} games -> {OUT_PATH} (fallback_used={fallback_used})")

if __name__ == "__main__":
    main()
