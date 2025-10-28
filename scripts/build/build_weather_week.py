#!/usr/bin/env python3
# build_weather_week.py
# Auto-detect the current NFL week, map home team -> stadium/city/roof,
# and fetch game-time weather (NWS first, Open-Meteo fallback).
#
# Output CSV schema:
# team,opponent,week,stadium,roof,forecast_summary,temp_f,wind_mph,precip_prob,forecast_datetime_utc
#
# Free sources used at runtime (citations in your docs/readme):
# - Pro-Football-Reference 2025 Games (weekly schedule): https://www.pro-football-reference.com/years/2025/games.htm
# - Wikipedia: List of current NFL stadiums (stadium, location, roof): https://en.wikipedia.org/wiki/List_of_current_NFL_stadiums
# - NWS API: https://api.weather.gov (points -> gridpoints -> forecast/hourly)
# - Open-Meteo (fallback): https://open-meteo.com/  + https://geocoding-api.open-meteo.com/v1/search
#
import sys, io, re, math, json, time, datetime
from datetime import datetime as dt, timezone, timedelta
from typing import Dict, Tuple, List, Optional
import pandas as pd
import requests

HDRS = {"User-Agent": "Mozilla/5.0 (+github.com/your-org/your-repo)"}

# Canonical team codes per user
TEAM_CODE = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF","Carolina Panthers":"CAR",
    "Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL","Denver Broncos":"DEN",
    "Detroit Lions":"DET","Green Bay Packers":"GB","Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
    "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG","New York Jets":"NYJ",
    "Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA","San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":"TEN","Washington Commanders":"WAS"
}

ABBR = {v:v for v in TEAM_CODE.values()}

def read_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HDRS, timeout=45)
    r.raise_for_status()
    try:
        return pd.read_html(io.StringIO(r.text))
    except ValueError:
        return []

def get_current_week_games() -> pd.DataFrame:
    """Scrape PFR 2025 schedule and select the upcoming 'current week' window (Thu..Mon around today)."""
    url = "https://www.pro-football-reference.com/years/2025/games.htm"
    tables = read_tables(url)
    # Find table(s) containing columns Week, Date, Time, Winner/tie, Loser/tie
    frames = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if ("week" in " ".join(cols)) and any("date" in c for c in cols) and any("winner" in c for c in cols):
            frames.append(t)
    if not frames:
        raise RuntimeError("Could not parse PFR weekly schedule.")
    df = pd.concat(frames, ignore_index=True)
    # Normalize columns
    cmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "week" in lc: cmap[c] = "week"
        elif "date" in lc: cmap[c] = "date"
        elif "time" in lc: cmap[c] = "time_et"
        elif "winner" in lc: cmap[c] = "winner"
        elif "loser" in lc: cmap[c] = "loser"
    df = df.rename(columns=cmap)
    keep = [c for c in ["week","date","time_et","winner","loser"] if c in df.columns]
    df = df[keep].dropna(subset=["week","date","winner","loser"])
    # Parse date/time (ET) -> UTC kickoff
    # PFR dates are YYYY-MM-DD; Time is ET like "1:00PM"
    def parse_kick(row):
        d = str(row["date"])
        t = str(row["time_et"]) if not pd.isna(row.get("time_et", "")) else "1:00 PM"
        t = t.replace("ET", "").strip()
        try:
            dt_et = dt.fromisoformat(d)  # naive date
        except Exception:
            dt_et = pd.to_datetime(d).to_pydatetime().replace(tzinfo=None)
        # parse time
        try:
            tm = pd.to_datetime(t).time()
        except Exception:
            tm = pd.to_datetime("1:00 PM").time()
        # Build aware ET (America/New_York). We avoid pytz; approximate ET offset via US rules using dateutil isn't guaranteed.
        # Simpler: assume ET = UTC-4 during NFL season.
        kickoff_local = dt_et.replace(hour=tm.hour, minute=tm.minute)
        kickoff_utc = kickoff_local + timedelta(hours=4) * -1  # ET->UTC: add 4 hours (UTC = ET + 4). We'll correct if needed.
        return kickoff_local, kickoff_utc

    df["kickoff_local"], df["kickoff_utc"] = zip(*df.apply(parse_kick, axis=1))

    # Decide "current week": choose the minimum week whose kickoff_utc is within [today-1d, today+6d] range.
    now_utc = dt.now(timezone.utc).replace(tzinfo=None)
    window_start = now_utc - timedelta(days=1)
    window_end = now_utc + timedelta(days=6)
    cand = df[(df["kickoff_utc"] >= window_start) & (df["kickoff_utc"] <= window_end)]
    if cand.empty:
        # fallback: next upcoming game(s) after now
        cand = df[df["kickoff_utc"] >= now_utc].sort_values("kickoff_utc").head(16)
    week = int(cand.iloc[0]["week"])
    return df[df["week"] == week].copy()

def get_stadium_map() -> pd.DataFrame:
    """Scrape Wikipedia list of current NFL stadiums -> map Team(s) -> (Stadium, Location, Roof type)."""
    url = "https://en.wikipedia.org/wiki/List_of_current_NFL_stadiums"
    tables = read_tables(url)
    if not tables:
        raise RuntimeError("Could not read NFL stadiums Wikipedia table.")
    # Find table with columns Name | Team(s) | Location | Roof type
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("name" in c for c in cols) and any("team" in c for c in cols) and any("location" in c for c in cols) and any("roof" in c for c in cols):
            df = t.copy()
            break
    if df is None:
        df = tables[0].copy()
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "name" in lc: colmap[c] = "stadium"
        elif "team" in lc: colmap[c] = "teams"
        elif "location" in lc: colmap[c] = "location"
        elif "roof" in lc: colmap[c] = "roof"
    df = df.rename(columns=colmap)
    df = df[[c for c in ["stadium","teams","location","roof"] if c in df.columns]].dropna()
    # explode teams
    rows = []
    for _, r in df.iterrows():
        teams = re.split(r",|/| and ", str(r["teams"]))
        for tm in teams:
            tm = tm.strip()
            if not tm: continue
            rows.append({"team_name": tm, "stadium": r["stadium"], "location": r["location"], "roof": r["roof"]})
    out = pd.DataFrame(rows).drop_duplicates(subset=["team_name"])
    # Map name to canonical abbr
    out["team"] = out["team_name"].map(TEAM_CODE)
    out = out.dropna(subset=["team"])
    # Extract city (before comma)
    out["city"] = out["location"].apply(lambda s: str(s).split(",")[0].strip())
    return out[["team","stadium","roof","city"]]

def geocode_city(city: str) -> Optional[Tuple[float,float]]:
    """Use Open-Meteo geocoding API to get lat/lon for a US city name."""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language":"en", "format":"json", "country":"US"}
        r = requests.get(url, params=params, headers=HDRS, timeout=30); r.raise_for_status()
        data = r.json()
        if data.get("results"):
            res = data["results"][0]
            return float(res["latitude"]), float(res["longitude"])
    except Exception:
        return None
    return None

def nws_hourly(lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Fetch NWS hourly forecast DataFrame with columns [time, temp_f, wind_mph, precip_prob, summary]."""
    try:
        meta = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=HDRS, timeout=30); meta.raise_for_status()
        grid = meta.json()
        url = grid["properties"]["forecastHourly"]
        r = requests.get(url, headers=HDRS, timeout=30); r.raise_for_status()
        js = r.json()
        periods = js["properties"]["periods"]
        rows = []
        for p in periods:
            t = pd.to_datetime(p["startTime"])
            temp_f = p.get("temperature")
            wind = p.get("windSpeed","0 mph")
            try:
                wind_mph = float(re.findall(r"[0-9]+", wind)[0])
            except Exception:
                wind_mph = None
            pop = p.get("probabilityOfPrecipitation", {}).get("value", None)  # 0-100
            summary = p.get("shortForecast","")
            rows.append({"time": t, "temp_f": temp_f, "wind_mph": wind_mph, "precip_prob": pop, "summary": summary})
        return pd.DataFrame(rows)
    except Exception:
        return None

# Open-Meteo fallback
WCODE = {
    0:"Clear", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Depositing rime fog",
    51:"Drizzle: light", 53:"Drizzle: moderate", 55:"Drizzle: dense",
    56:"Freezing drizzle: light", 57:"Freezing drizzle: dense",
    61:"Rain: slight", 63:"Rain: moderate", 65:"Rain: heavy",
    66:"Freezing rain: light", 67:"Freezing rain: heavy",
    71:"Snow: slight", 73:"Snow: moderate", 75:"Snow: heavy",
    77:"Snow grains",
    80:"Rain showers: slight", 81:"Rain showers: moderate", 82:"Rain showers: violent",
    85:"Snow showers: slight", 86:"Snow showers: heavy",
    95:"Thunderstorm: slight/moderate", 96:"Thunderstorm with hail", 99:"Thunderstorm with heavy hail"
}

def openmeteo_hourly(lat: float, lon: float) -> Optional[pd.DataFrame]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,weathercode",
            "temperature_unit":"fahrenheit", "windspeed_unit":"mph", "timezone":"UTC"
        }
        r = requests.get(url, params=params, headers=HDRS, timeout=30); r.raise_for_status()
        js = r.json()
        times = js["hourly"]["time"]
        temps = js["hourly"]["temperature_2m"]
        winds = js["hourly"]["wind_speed_10m"]
        pops = js["hourly"].get("precipitation_probability", [None]*len(times))
        codes = js["hourly"].get("weathercode", [None]*len(times))
        rows = []
        for i, t in enumerate(times):
            ti = pd.to_datetime(t)
            temp_f = temps[i]
            wind_mph = winds[i]
            pop = pops[i] if pops else None
            code = codes[i]
            summary = WCODE.get(code, "")
            rows.append({"time": ti, "temp_f": temp_f, "wind_mph": wind_mph, "precip_prob": pop, "summary": summary})
        return pd.DataFrame(rows)
    except Exception:
        return None

def nearest_hour(df: pd.DataFrame, when_utc: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx = (df["time"] - when_utc).abs().idxmin()
    return df.loc[idx]

def roof_normalize(roof: str) -> str:
    s = str(roof).lower()
    if "retract" in s: return "Retractable"
    if "dome" in s or "translucent" in s or "fixed" in s: return "Dome/Fixed"
    return "Open"

def main(out_csv: str="weather.csv"):
    # 1) Pick current NFL week from PFR schedule
    week_games = get_current_week_games()
    week = int(week_games["week"].iloc[0])
    # Build home/away per game. PFR lists Winner and Loser; we need to deduce home via original table? Safer approach:
    # PFR tables also include 'Home/Away' marker sometimes; if absent, we infer via cross-join of unique matchups by looking for '@' in original text.
    # For robustness, re-scan raw HTML for "TeamA @ TeamB" patterns.
    url = "https://www.pro-football-reference.com/years/2025/games.htm"
    raw = requests.get(url, headers=HDRS, timeout=45).text
    # Build a matchup index by date/time: simplistic but serviceable
    # We'll derive pairs directly from df by assuming 'winner' vs 'loser' doesn't imply home/away; instead, compute home/away using a second pass:
    # PFR tables include columns 'Winner/tie' and 'Loser/tie' and may include '@' in the 'game location' cell; read another set of tables if available.
    # For this script, we will instead use an external inference: home team is the 'Loser' if the 'winner' row contains '@' marker else fallback to Wikipedia mapping later.
    # To avoid brittleness, we will fetch team list for the given week from nfl.com by-week page if available; else assume 'loser' is home when 'winner' contains '@'.

    # Minimal viable mapping: derive both teams, then later join stadium by 'home' using a heuristic: check stadium city alignment to team's home city in stadium map.
    # Extract both teams per row using existing columns:
    week_games = week_games.rename(columns={"winner":"winner_name","loser":"loser_name"})
    # Map to canonical abbr from TEAM_CODE (full names)
    week_games["winner_abbr"] = week_games["winner_name"].map(TEAM_CODE)
    week_games["loser_abbr"] = week_games["loser_name"].map(TEAM_CODE)
    # Some rows might be future and have blank winner/loser; drop those
    week_games = week_games.dropna(subset=["winner_abbr","loser_abbr"]).copy()

    # Heuristic: for schedule rows before games are played, PFR often leaves scores blank but still lists pairings under 'Winner/Loser' columns in alphabetical or home/away order varies.
    # We'll fall back to another PFR table variant that includes "Visitor, Home". Try to parse again:
    tables2 = read_tables(url)
    vhome = None
    for t in tables2:
        cols = [str(c).lower() for c in t.columns]
        if any("visitor" in c for c in cols) and any("home" in c for c in cols):
            vhome = t.rename(columns={t.columns[0]:"week"} if len(t.columns)>0 else {})
            break
    if vhome is not None and "week" in vhome.columns:
        cmap = {}
        for c in vhome.columns:
            lc = str(c).lower()
            if "week" in lc: cmap[c]="week"
            elif "visitor" in lc: cmap[c]="visitor"
            elif "home" in lc: cmap[c]="home"
            elif "date" in lc: cmap[c]="date"
            elif "time" in lc: cmap[c]="time_et"
        vhome = vhome.rename(columns=cmap)
        vhome = vhome[[c for c in ["week","visitor","home","date","time_et"] if c in vhome.columns]]
        vhome = vhome[vhome["week"]==week]
        vhome["visitor_abbr"] = vhome["visitor"].map(TEAM_CODE)
        vhome["home_abbr"] = vhome["home"].map(TEAM_CODE)
        vhome = vhome.dropna(subset=["visitor_abbr","home_abbr"])
        sched = vhome[["home_abbr","visitor_abbr","date","time_et"]].rename(columns={"home_abbr":"team","visitor_abbr":"opponent"})
    else:
        # fallback: use winner/loser names as teams (order unknown); assume 'loser' is home (weak), but better than nothing.
        sched = week_games[["loser_abbr","winner_abbr","date","time_et"]].rename(columns={"loser_abbr":"team","winner_abbr":"opponent"})

    # 2) Stadium map
    stad = get_stadium_map()  # columns: team, stadium, roof, city

    # Merge schedule with stadium info
    games = sched.merge(stad, on="team", how="left")
    # Parse kickoff UTC timestamp
    def to_utc(d, t):
        t = (t or "1:00 PM").replace("ET","").strip()
        try:
            d0 = pd.to_datetime(d).to_pydatetime()
        except Exception:
            d0 = dt.fromisoformat(str(d))
        try:
            tm = pd.to_datetime(t).time()
        except Exception:
            tm = pd.to_datetime("1:00 PM").time()
        local = d0.replace(hour=tm.hour, minute=tm.minute)
        # Convert ET -> UTC (+4) during NFL season (DST). This is a simplification.
        return (local + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    games["forecast_datetime_utc"] = games.apply(lambda r: to_utc(r.get("date"), r.get("time_et")), axis=1)

    # 3) Weather per game
    out_rows = []
    for _, r in games.iterrows():
        team = r["team"]
        opp = r["opponent"]
        stadium = r.get("stadium", "")
        roof = roof_normalize(r.get("roof",""))
        city = r.get("city","")
        when_utc = pd.to_datetime(r["forecast_datetime_utc"])

        temp_f = wind_mph = precip_prob = None
        summary = ""

        # Geocode city
        latlon = geocode_city(city) if city else None
        if latlon:
            lat, lon = latlon
            # Try NWS first
            nws = nws_hourly(lat, lon)
            row = None
            if nws is not None and not nws.empty:
                row = nearest_hour(nws, when_utc)
            if row is None:
                # Fallback Open-Meteo
                om = openmeteo_hourly(lat, lon)
                if om is not None and not om.empty:
                    row = nearest_hour(om, when_utc)
            if row is not None:
                summary = row.get("summary","")
                temp_f = row.get("temp_f", None)
                wind_mph = row.get("wind_mph", None)
                precip_prob = row.get("precip_prob", None)

        # If dome, blank weather fields
        if roof == "Dome/Fixed":
            summary = ""
            temp_f = wind_mph = precip_prob = None

        out_rows.append({
            "team": team, "opponent": opp, "week": int(games["week"].iloc[0]) if "week" in games.columns else None,
            "stadium": stadium, "roof": roof, "forecast_summary": summary,
            "temp_f": temp_f, "wind_mph": wind_mph, "precip_prob": precip_prob,
            "forecast_datetime_utc": when_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        })

    df_out = pd.DataFrame(out_rows, columns=[
        "team","opponent","week","stadium","roof","forecast_summary","temp_f","wind_mph","precip_prob","forecast_datetime_utc"
    ])
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df_out)} rows.")

if __name__ == "__main__":
    out_csv = "weather.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out_csv)
