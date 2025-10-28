#!/usr/bin/env python3
# build_injuries_weekly.py
# Fetch current week's NFL injuries & practice participation (free sources), auto-detecting week.
# Primary source: NFL.com Injuries index (week selector). Fallbacks to related NFL.com roundups when needed.
#
# Output CSV schema:
#   player,team,week,game_status,practice_status,body_part,designation,report_date
#
# Citations you can include in your docs:
# - NFL.com Injuries (current week): https://www.nfl.com/injuries/
# - NFL.com weekly inactives / injuries-to-monitor posts (same week)
# - Pro-Football-Reference 2025 Games (for week detection if needed): https://www.pro-football-reference.com/years/2025/games.htm
#
# Usage:
#   python build_injuries_weekly.py            # writes injuries.csv
#   python build_injuries_weekly.py data/injuries.csv
#
import sys, io, re, json
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import requests

HDRS = {"User-Agent":"Mozilla/5.0 (+github.com/your-org/your-repo)"}

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

def read(url: str) -> str:
    r = requests.get(url, headers=HDRS, timeout=45)
    r.raise_for_status()
    return r.text

def read_tables(url: str) -> List[pd.DataFrame]:
    html = read(url)
    try:
        return pd.read_html(io.StringIO(html))
    except ValueError:
        return []

def detect_current_week() -> int:
    """Try to read 'Week X' from NFL.com/injuries/, fallback to PFR games window (closest week)."""
    try:
        html = read("https://www.nfl.com/injuries/")
        m = re.search(r'Week\s+(\d+)\s+of the\s+(\d{4})\s+Season', html, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback via PFR: pick week containing today's date range
    try:
        tables = read_tables("https://www.pro-football-reference.com/years/2025/games.htm")
        df = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("week" in c for c in cols) and any("date" in c for c in cols):
                df = t; break
        if df is not None and "Week" in df.columns and "Date" in df.columns:
            # choose the week with dates closest to today
            d = df.dropna(subset=["Week","Date"]).copy()
            d["dt"] = pd.to_datetime(d["Date"], errors="coerce")
            today = pd.Timestamp.utcnow().normalize()
            grp = d.groupby("Week")["dt"].apply(lambda s: (s - today).abs().min())
            wk = grp.sort_values().index[0]
            return int(wk)
    except Exception:
        pass
    # Last resort
    return 1

def parse_nfl_injuries_week(week: int) -> pd.DataFrame:
    """Parse NFL.com Injuries for a given week. Returns normalized DataFrame."""
    url = f"https://www.nfl.com/injuries/"
    html = read(url)
    # NFL.com uses one page with filter; tables should be present for current week. If not current, try query param variants.
    tables = pd.read_html(io.StringIO(html))
    # Heuristic: select tables that have Player and Game Status columns
    frames = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("player" in c for c in cols) and any("game status" in c for c in cols):
            frames.append(t)
    if not frames:
        # try fallback paths (archived style)
        return pd.DataFrame(columns=["player","team","week","game_status","practice_status","body_part","designation","report_date"])
    df = pd.concat(frames, ignore_index=True)
    # Normalize columns
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "player" in lc: colmap[c] = "player"
        elif "team" in lc or "club" in lc: colmap[c] = "team_name"
        elif "injury" in lc: colmap[c] = "injury"
        elif "practice" in lc: colmap[c] = "practice_status"
        elif "game status" in lc: colmap[c] = "game_status"
        elif "status" == lc: colmap[c] = "designation"
    d = df.rename(columns=colmap)
    # Keep essentials; some tables may not include team; we derive team from section headers if necessary (future enhancement)
    keep = [c for c in ["player","team_name","injury","practice_status","game_status","designation"] if c in d.columns]
    d = d[keep].copy()
    # Split injury field into body_part (first token before comma/paren)
    if "injury" in d.columns:
        d["body_part"] = d["injury"].astype(str).str.split("[,(/]").str[0].str.strip().replace({"":"Unknown"})
    else:
        d["body_part"] = "Unknown"
    # Canonicalize team
    if "team_name" in d.columns:
        d["team"] = d["team_name"].map(TEAM_CODE)
    else:
        d["team"] = pd.NA
    d["week"] = week
    # Designation: prefer game_status when present (Out/Doubtful/Questionable)
    if "game_status" in d.columns:
        d["designation"] = d["game_status"]
    # Report date = today UTC
    d["report_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Final selection
    out = d.rename(columns={"injury":"_inj"})
    out = out[["player","team","week","game_status","practice_status","body_part","designation","report_date"]].copy()
    # Drop rows with no player
    out = out.dropna(subset=["player"]).reset_index(drop=True)
    return out

def main(out_csv: str="injuries.csv"):
    week = detect_current_week()
    df = parse_nfl_injuries_week(week)
    # Deduplicate on player+team
    if not df.empty:
        df = df.drop_duplicates(subset=["player","team"])
    else:
        # Ensure headers
        df = pd.DataFrame(columns=["player","team","week","game_status","practice_status","body_part","designation","report_date"])
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows for week {week}.")

if __name__ == "__main__":
    out = "injuries.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out)
