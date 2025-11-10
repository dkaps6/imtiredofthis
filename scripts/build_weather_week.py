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
WEEK = 10

NWS_HEADERS = {
    "User-Agent": "imtiredofthis-weather (+https://github.com/imtiredofthis)",
    "Accept": "application/geo+json",
}

SESSION = requests.Session()
logger = logging.getLogger(__name__)


CITY_FALLBACK_COORDS = {
    ("MIAMI GARDENS", "FL"): {
        "city": "Miami",
        "state": "FL",
        "lat": 25.7617,
        "lon": -80.1918,
    },
    ("ORCHARD PARK", "NY"): {
        "city": "Buffalo",
        "state": "NY",
        "lat": 42.8864,
        "lon": -78.8784,
    },
    ("FOXBOUROUGH", "MA"): {
        "city": "Boston",
        "state": "MA",
        "lat": 42.3601,
        "lon": -71.0589,
    },
    ("FOXBOROUGH", "MA"): {
        "city": "Boston",
        "state": "MA",
        "lat": 42.3601,
        "lon": -71.0589,
    },
}


def request_json_with_retry(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 15,
    max_attempts: int = 3,
) -> dict:
    """Best-effort GET wrapper that bubbles errors for fallback handling."""

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = SESSION.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status in {500, 502, 503, 504} and attempt + 1 < max_attempts:
                continue
            last_exc = exc
            break
        except requests.RequestException as exc:  # network issues, timeouts, etc.
            last_exc = exc
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[weather] request_json_with_retry exhausted for {url}")


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
    """Load the upcoming slate from repo artifacts (game_lines preferred)."""

    game_lines = Path("data/game_lines.csv")
    opponent_map = Path("data/opponent_map_from_props.csv")

    if game_lines.exists():
        try:
            gl = pd.read_csv(game_lines)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            gl = pd.DataFrame()
        if not gl.empty:
            cols = {c.lower(): c for c in gl.columns}
            home_col = next((cols[key] for key in ("home", "home_team", "home_abbr") if key in cols), None)
            away_col = next((cols[key] for key in ("away", "away_team", "away_abbr") if key in cols), None)
            if home_col and away_col:
                out = pd.DataFrame(
                    {
                        "home": gl[home_col].astype(str).str.upper().str.strip(),
                        "away": gl[away_col].astype(str).str.upper().str.strip(),
                    }
                )
                tz_col = next((cols[key] for key in ("local_tz", "tz", "timezone") if key in cols), None)
                if tz_col:
                    out["local_tz"] = gl[tz_col].astype(str).str.strip()
                else:
                    out["local_tz"] = out["home"].map(_infer_tz_from_team)

                if "kickoff_local" in cols:
                    kickoff_local = pd.to_datetime(gl[cols["kickoff_local"]], errors="coerce")
                else:
                    kickoff_local = pd.Series(pd.NaT, index=gl.index, dtype="datetime64[ns]")

                if "kickoff_utc" in cols:
                    kickoff_utc = pd.to_datetime(gl[cols["kickoff_utc"]], utc=True, errors="coerce")
                elif "commence_time" in cols:
                    kickoff_utc = pd.to_datetime(gl[cols["commence_time"]], utc=True, errors="coerce")
                else:
                    kickoff_utc = pd.Series([pd.NaT] * len(gl), index=gl.index)
                    kickoff_utc = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")

                localized = []
                for idx in range(len(out)):
                    tz_name = out.at[idx, "local_tz"]
                    home_team = out.at[idx, "home"]
                    tz_name = tz_name if isinstance(tz_name, str) and tz_name else _infer_tz_from_team(home_team)
                    try:
                        tzinfo = ZoneInfo(tz_name)
                    except Exception:
                        tzinfo = ZoneInfo("UTC")

                    local_val = kickoff_local.iloc[idx]
                    utc_val = kickoff_utc.iloc[idx] if idx < len(kickoff_utc) else pd.NaT

                    if pd.isna(local_val) and pd.notna(utc_val):
                        try:
                            local_val = utc_val.tz_convert(tzinfo)
                        except Exception:
                            local_val = utc_val
                    elif isinstance(local_val, pd.Timestamp) and local_val.tzinfo is None:
                        try:
                            local_val = local_val.tz_localize(tzinfo)
                        except Exception:
                            local_val = local_val.tz_localize("UTC")
                    localized.append(local_val)

                out["kickoff_local"] = pd.Series(localized, index=out.index)

                if "kickoff_utc" in cols or "commence_time" in cols:
                    out["kickoff_utc"] = kickoff_utc
                else:
                    utc_from_local = []
                    for val in out["kickoff_local"]:
                        if pd.isna(val):
                            utc_from_local.append(pd.NaT)
                            continue
                        try:
                            utc_from_local.append(val.astimezone(ZoneInfo("UTC")))
                        except Exception:
                            utc_from_local.append(pd.NaT)
                    out["kickoff_utc"] = pd.Series(utc_from_local, index=out.index)

                out = out.dropna(subset=["home", "away"]).drop_duplicates(subset=["home", "away"])
                out.attrs["fallback_used"] = False
                return out[["home", "away", "local_tz", "kickoff_local", "kickoff_utc"]]

    if opponent_map.exists():
        try:
            om = pd.read_csv(opponent_map)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            om = pd.DataFrame()
        if not om.empty and {"team", "opponent", "game_timestamp"}.issubset(om.columns):
            out = pd.DataFrame(
                {
                    "home": om["team"].astype(str).str.upper().str.strip(),
                    "away": om["opponent"].astype(str).str.upper().str.strip(),
                }
            )
            out["kickoff_utc"] = pd.to_datetime(om["game_timestamp"], utc=True, errors="coerce")
            out["local_tz"] = out["home"].map(_infer_tz_from_team)
            local_vals = []
            for idx in range(len(out)):
                utc_ts = out.at[idx, "kickoff_utc"]
                tz_name = out.at[idx, "local_tz"]
                tz_name = tz_name if isinstance(tz_name, str) and tz_name else _infer_tz_from_team(out.at[idx, "home"])
                try:
                    tzinfo = ZoneInfo(tz_name)
                except Exception:
                    tzinfo = ZoneInfo("UTC")
                if pd.isna(utc_ts):
                    local_vals.append(pd.NaT)
                else:
                    try:
                        local_vals.append(utc_ts.tz_convert(tzinfo))
                    except Exception:
                        local_vals.append(utc_ts)
            out["kickoff_local"] = pd.Series(local_vals, index=out.index)
            out.attrs["fallback_used"] = False
            return out[["home", "away", "local_tz", "kickoff_local", "kickoff_utc"]]

    fallback_df = build_slate_fallback()
    if not fallback_df.empty:
        return fallback_df

    raise RuntimeError(
        "[weather] Could not infer this week's slate from repo (no usable game_lines/opponent_map file with home/away/kickoff). "
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
    points_json = request_json_with_retry(
        points_url,
        headers=NWS_HEADERS,
        timeout=15,
    )

    properties = points_json.get("properties", {})
    forecast_url = properties.get("forecastHourly")
    if not forecast_url:
        raise RuntimeError(f"[weather] No forecastHourly URL for {lat},{lon}")

    forecast_json = request_json_with_retry(
        forecast_url,
        headers=NWS_HEADERS,
        timeout=15,
    )

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


def _fallback_meta(meta: dict | None) -> dict | None:
    if not meta:
        return None
    city = str(meta.get("city", "")).strip().upper()
    state = str(meta.get("state", "")).strip().upper()
    fallback = CITY_FALLBACK_COORDS.get((city, state))
    if not fallback:
        return None
    merged = dict(meta)
    merged.update(fallback)
    return merged


def _mark_indoor_neutral(base_row: dict) -> dict:
    row = dict(base_row)
    row["indoor"] = True
    row["forecast_ok"] = 0
    if not row.get("blurb"):
        row["blurb"] = "Forecast unavailable; treating as indoor/neutral"
    return row


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
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        fallback = _fallback_meta(meta)
        if status == 404 and fallback:
            try:
                fc_df = _get_hourly_forecast(fallback.get("lat"), fallback.get("lon"))
                meta = fallback
            except Exception as fallback_err:
                logger.warning(
                    "[weather] WARNING: unable to fetch forecast for %s at %s (fallback %s,%s failed: %s), marking as indoor/neutral.",
                    home_team,
                    meta.get("stadium", "unknown stadium"),
                    fallback.get("city"),
                    fallback.get("state"),
                    fallback_err,
                )
                return _mark_indoor_neutral(base_row)
        else:
            logger.warning(
                "[weather] WARNING: unable to fetch forecast for %s at %s, marking as indoor/neutral.",
                home_team,
                meta.get("stadium", "unknown stadium"),
            )
            return _mark_indoor_neutral(base_row)
    except Exception as exc:
        logger.warning(
            "[weather] WARNING: unable to fetch forecast for %s at %s: %s, marking as indoor/neutral.",
            home_team,
            meta.get("stadium", "unknown stadium"),
            exc,
        )
        return _mark_indoor_neutral(base_row)

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

    try:
        slate_df = _load_slate_from_repo()
    except RuntimeError as err:
        logger.warning("%s", err)
        slate_df = pd.DataFrame()

    if slate_df.empty:
        logger.warning("[weather] Slate is empty even after fallback; writing empty weather file")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print(f"[weather] wrote 0 games -> {OUT_PATH} (fallback_used=False)")
        return

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
        logger.warning("[weather] No weather rows generated at all; writing empty file")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        weather_df.to_csv(OUT_PATH, index=False)
        print(f"[weather] wrote 0 games -> {OUT_PATH} (fallback_used={fallback_used})")
        return

    team_rows = []
    for _, row in weather_df.iterrows():
        for team_col, opp_col in (("home", "away"), ("away", "home")):
            team_code = str(row.get(team_col, "")).upper().strip()
            if not team_code:
                continue
            opp_code = str(row.get(opp_col, "")).upper().strip()
            condition = "DOME" if row.get("indoor") else row.get("conditions_main") or "UNKNOWN"
            entry = {
                "team_abbr": team_code,
                "opponent_abbr": opp_code or pd.NA,
                "week": WEEK,
                "stadium": row.get("stadium"),
                "city": row.get("city"),
                "state": row.get("state"),
                "condition": condition,
                "condition_desc": row.get("conditions_desc"),
                "temp_f": row.get("temp_F_mean"),
                "wind_mph": row.get("wind_mph_mean"),
                "precip_prob": row.get("precip_prob_max"),
                "forecast_ok": int(bool(row.get("forecast_ok"))) if condition != "DOME" else 0,
                "notes": row.get("notes"),
            }
            team_rows.append(entry)

    team_weather = pd.DataFrame(team_rows).drop_duplicates(subset=["team_abbr", "week"])
    if team_weather.empty:
        logger.warning("[weather] No team-level weather rows generated; writing empty file")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        team_weather.to_csv(OUT_PATH, index=False)
        print(f"[weather] wrote 0 games -> {OUT_PATH} (fallback_used={fallback_used})")
        return

    team_weather["team_abbr"] = team_weather["team_abbr"].astype(str).str.upper().str.strip()
    team_weather["opponent_abbr"] = team_weather["opponent_abbr"].astype(str).str.upper().str.strip().replace({"": pd.NA})

    matched_mask = team_weather["forecast_ok"].fillna(0).astype(int)
    matched_count = int(matched_mask.sum())
    total_rows = len(team_weather)
    dome_count = int((team_weather["condition"].astype(str).str.upper() == "DOME").sum())

    logger.info(
        "[WEATHER] %d/%d matched successfully, %d dome teams excluded",
        matched_count,
        total_rows,
        dome_count,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_weather.to_csv(OUT_PATH, index=False)
    print(
        f"[weather] wrote {len(team_weather)} team rows -> {OUT_PATH} "
        f"(fallback_used={fallback_used})"
    )

if __name__ == "__main__":
    main()
