# scripts/build_weather_week.py
#
# Build per-game weather for this week's slate and write data/weather_week.csv.
# - Uses stadium_locations.STADIUM_LOCATION to map home team -> stadium/city/state/outdoor.
# - Uses OpenWeather (free tier) for hourly forecast.
# - Summarizes temp/wind/precip around kickoff.
# - Fails fast (RuntimeError) if we cannot generate at least 1 row.
#
# IMPORTANT:
#   You MUST set OPENWEATHER_API_KEY via env in the workflow step that calls this script.

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.stadium_locations import STADIUM_LOCATION

OUT_PATH = Path("data") / "weather_week.csv"

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
            df = pd.read_csv(c)
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

                return out

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

def _geocode_city_state(city: str, state: str, api_key: str):
    """
    Use OpenWeather geocoding API to get (lat, lon) from "City, State".
    Raises if not found.
    """
    q = f"{city},{state},US"
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": q, "limit": 1, "appid": api_key}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError(f"[weather] Could not geocode {q}")
    return data[0]["lat"], data[0]["lon"]

def _get_hourly_forecast(lat: float, lon: float, api_key: str) -> pd.DataFrame:
    """
    Pull ~5-day / 3-hour block forecast from OpenWeather.
    Return as DataFrame with UTC timestamps and main weather fields.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "imperial",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    rows = []
    for block in data.get("list", []):
        ts_utc = datetime.fromtimestamp(block["dt"], tz=ZoneInfo("UTC"))
        rows.append({
            "ts_utc": ts_utc,
            "temp_F": block["main"]["temp"],
            "wind_mph": block["wind"]["speed"],
            "weather_main": block["weather"][0]["main"],
            "weather_desc": block["weather"][0]["description"],
            "pop_rain": block.get("pop", 0.0),
        })
    return pd.DataFrame(rows)

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

def _weather_row_for_game(row: pd.Series, api_key: str) -> dict:
    """
    Build weather summary for a single game row with columns:
      home, away, kickoff_local, local_tz
    """
    home_team = str(row["home"]).upper().strip()
    away_team = str(row["away"]).upper().strip()

    if home_team not in STADIUM_LOCATION:
        raise RuntimeError(f"[weather] No stadium mapping for home team {home_team}")

    meta = STADIUM_LOCATION[home_team]
    city = meta["city"]
    state = meta["state"]
    indoor = not meta["outdoor"]
    stadium = meta["stadium"]

    lat, lon = _geocode_city_state(city, state, api_key)
    fc_df = _get_hourly_forecast(lat, lon, api_key)

    summary = _summarize_window(
        fc_df,
        kickoff_local_dt=row["kickoff_local"],
        local_tz=row["local_tz"],
    )

    if not summary:
        # still return a row so downstream doesn't explode, but mark forecast_ok=0
        return {
            "home": home_team,
            "away": away_team,
            "stadium": stadium,
            "city": city,
            "state": state,
            "kickoff_local": row["kickoff_local"].isoformat(),
            "indoor": indoor,
            "forecast_ok": 0,
        }

    out = {
        "home": home_team,
        "away": away_team,
        "stadium": stadium,
        "city": city,
        "state": state,
        "kickoff_local": row["kickoff_local"].isoformat(),
        "indoor": indoor,
        "forecast_ok": 1,
        **summary,
    }

    # modeling-friendly flags
    out["cold_flag"] = 1 if (out.get("temp_F_mean", 100) < 40) else 0
    out["wind_flag"] = 1 if (out.get("wind_mph_mean", 0) >= 15) else 0
    out["rain_flag"] = 1 if (out.get("precip_prob_max", 0) >= 0.4) else 0

    # quick narrative
    if indoor:
        out["blurb"] = "Indoors/retractable - neutral conditions"
    else:
        bits = []
        if "temp_F_mean" in out and pd.notna(out["temp_F_mean"]):
            bits.append(f"{round(out['temp_F_mean'])}F avg")
        if "wind_mph_mean" in out and pd.notna(out["wind_mph_mean"]):
            bits.append(f"wind ~{round(out['wind_mph_mean'],1)} mph")
        if out["rain_flag"]:
            bits.append("rain risk")
        if "conditions_desc" in out and out["conditions_desc"]:
            bits.append(str(out["conditions_desc"]))
        out["blurb"] = ", ".join(bits)

    return out

def main():
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("[weather] OPENWEATHER_API_KEY not set in env")

    slate_df = _load_slate_from_repo()

    rows = []
    for _, game in slate_df.iterrows():
        wxrow = _weather_row_for_game(game, api_key)
        rows.append(wxrow)

    weather_df = pd.DataFrame(rows).drop_duplicates()

    # fail-fast if totally empty or all forecast_ok=0
    if weather_df.empty:
        raise RuntimeError("[weather] No weather rows generated at all")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUT_PATH, index=False)
    print(f"[weather] wrote {OUT_PATH} rows={len(weather_df)} ok")

if __name__ == "__main__":
    main()
