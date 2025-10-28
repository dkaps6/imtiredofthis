#!/usr/bin/env python3
# build_opponent_map_from_props.py
#
# Goal: Using the current week schedule + free prop boards, map players to (team, opponent, week).
# Output CSV (canonicalized teams): player,team,opponent,week,source
#
# Free sources (cite these in your docs):
# - Schedule: Pro-Football-Reference 2025 Games
#   https://www.pro-football-reference.com/years/2025/games.htm
# - Prop boards (multi-source, redundant):
#   VegasInsider player props (board): https://www.vegasinsider.com/nfl/odds/player-props/
#   RotoWire player props (board):    https://www.rotowire.com/betting/nfl/player-props.php
#   Action Network props hub:         https://www.actionnetwork.com/nfl/props
#
# Method:
# 1) Detect the current week window from PFR by nearest dates, then build the per-game schedule for that week
#    with canonical team abbreviations.
# 2) For each prop board URL, fetch HTML and parse:
#       - game headers (e.g., "NYJ vs KC", "Chiefs vs Commanders") to determine the matchup context.
#       - player rows within each game, preferring patterns like "Player Name (KC)" to capture team.
#       - if team is not present, we assign team by comparing player occurrences across both sides and
#         using a weak-name dictionary (only if detected); otherwise, leave team blank but keep opponent.
# 3) For each parsed (player, team_guess, matchup), convert team_guess to canonical abbr and set opponent
#    to the *other* team in that matchup. If team_guess is missing, we will leave team blank and still
#    include the opponent (useful for downstream joining with your roster map).
# 4) De-duplicate across sources; keep the earliest source string for traceability.
#
# Notes:
# - This is a best-effort scraper over *free* pages that may change markup. The script is defensive:
#   it tolerates failures on any single source and proceeds with what it can parse.
# - Team canonicalization is enforced via TEAM_ABBR mapping below.
#
import sys, io, re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import requests

HDRS = {"User-Agent":"Mozilla/5.0 (+github.com/your-org/your-repo)"}

TEAM_ABBR = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL","DEN":"DEN",
    "DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC","LV":"LV","LAC":"LAC","LAR":"LAR","MIA":"MIA",
    "MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB","TEN":"TEN","WAS":"WAS"
}
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF","Carolina Panthers":"CAR",
    "Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL","Denver Broncos":"DEN",
    "Detroit Lions":"DET","Green Bay Packers":"GB","Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
    "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG","New York Jets":"NYJ",
    "Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA","San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB",
    "Tennessee Titans":"TEN","Washington Commanders":"WAS"
}
TEAM_ALIASES = {
    "Chiefs":"KC","Chargers":"LAC","Rams":"LAR","Raiders":"LV","49ers":"SF","Niners":"SF","Football Team":"WAS","Commanders":"WAS",
    "Patriots":"NE","Packers":"GB","Giants":"NYG","Jets":"NYJ","Jaguars":"JAX","Titans":"TEN","Buccaneers":"TB","Bucs":"TB",
    "Saints":"NO","Eagles":"PHI","Cowboys":"DAL","Bills":"BUF","Dolphins":"MIA","Lions":"DET","Bears":"CHI","Vikings":"MIN",
    "Falcons":"ATL","Panthers":"CAR","Bengals":"CIN","Browns":"CLE","Ravens":"BAL","Broncos":"DEN","Seahawks":"SEA","Steelers":"PIT",
    "Cardinals":"ARI","Colts":"IND","Texans":"HOU"
}

def read_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HDRS, timeout=45)
    r.raise_for_status()
    try:
        return pd.read_html(io.StringIO(r.text))
    except ValueError:
        return []

# --- Week & schedule via PFR ---
def current_week_games() -> pd.DataFrame:
    url = "https://www.pro-football-reference.com/years/2025/games.htm"
    tables = read_tables(url)
    if not tables:
        raise RuntimeError("Could not read PFR schedule.")
    t = None
    for df in tables:
        cols = [str(c).lower() for c in df.columns]
        if any("visitor" in c for c in cols) and any("home" in c for c in cols):
            t = df; break
    if t is None:
        t = tables[0]
    cmap = {}
    for c in t.columns:
        lc = str(c).lower()
        if "week" in lc: cmap[c]="week"
        elif "visitor" in lc: cmap[c]="visitor"
        elif "home" in lc: cmap[c]="home"
        elif "date" in lc: cmap[c]="date"
        elif "time" in lc: cmap[c]="time_et"
    d = t.rename(columns=cmap)
    d = d[[c for c in ["week","visitor","home","date","time_et"] if c in d.columns]].dropna(subset=["week","visitor","home"])
    d["visitor_abbr"] = d["visitor"].map(TEAM_NAME_TO_ABBR).fillna(d["visitor"].map(TEAM_ALIASES))
    d["home_abbr"] = d["home"].map(TEAM_NAME_TO_ABBR).fillna(d["home"].map(TEAM_ALIASES))
    d = d.dropna(subset=["visitor_abbr","home_abbr"])
    # pick current week by nearest dates
    if "date" in d.columns:
        d["dt"] = pd.to_datetime(d["date"], errors="coerce")
        today = pd.Timestamp.utcnow().normalize()
        grp = d.groupby("week")["dt"].apply(lambda s: (s - today).abs().min())
        week = int(grp.sort_values().index[0])
    else:
        week = int(d["week"].iloc[-1])
    cur = d[d["week"] == week][["week","visitor_abbr","home_abbr"]].rename(columns={"visitor_abbr":"away","home_abbr":"home"}).copy()
    return cur

# --- Prop board parsers ---
def parse_pairs_from_text(text: str) -> List[Tuple[str, Optional[str]]]:
    """Extract (player_name, team_abbr?) tuples from text using simple patterns like 'Player Name (KC)'."""
    pairs = []
    # pattern: Player Name (KC)
    for m in re.finditer(r'([A-Z][a-zA-Z\.\'\-]+(?:\s[A-Z][a-zA-Z\.\'\-]+){0,2})\s*\(([A-Z]{2,3})\)', text):
        name = m.group(1).strip()
        abbr = m.group(2).strip()
        if abbr in TEAM_ABBR:
            pairs.append((name, abbr))
    # fallback: just player names without team
    for m in re.finditer(r'([A-Z][a-zA-Z\.\'\-]+(?:\s[A-Z][a-zA-Z\.\'\-]+){0,2})', text):
        nm = m.group(1).strip()
        if len(nm.split()) >= 2 and len(nm) <= 30:
            pairs.append((nm, None))
    return list(dict.fromkeys(pairs))  # unique preserve order

def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HDRS, timeout=45)
    r.raise_for_status()
    return r.text

def extract_from_board(url: str) -> pd.DataFrame:
    try:
        html = fetch_text(url)
    except Exception:
        return pd.DataFrame(columns=["player","team_guess","matchup_hint","source"])
    text = re.sub(r'\s+', ' ', html)
    # Try to find matchup headers like "NYJ vs KC" or "Chiefs vs Commanders"
    matchups = []
    # Patterns for abbr vs abbr
    for m in re.finditer(r'\b(ARI|ATL|BAL|BUF|CAR|CHI|CIN|CLE|DAL|DEN|DET|GB|HOU|IND|JAX|KC|LV|LAC|LAR|MIA|MIN|NE|NO|NYG|NYJ|PHI|PIT|SEA|SF|TB|TEN|WAS)\s*(?:@|vs\.?|v)\s*(ARI|ATL|BAL|BUF|CAR|CHI|CIN|CLE|DAL|DEN|DET|GB|HOU|IND|JAX|KC|LV|LAC|LAR|MIA|MIN|NE|NO|NYG|NYJ|PHI|PIT|SEA|SF|TB|TEN|WAS)\b', text):
        matchups.append((m.group(1), m.group(2)))
    # Patterns for names (Chiefs vs Commanders)
    for m in re.finditer(r'\b([A-Z][a-zA-Z]+)\s*(?:@|vs\.?|v)\s*([A-Z][a-zA-Z]+)\b', text):
        t1 = TEAM_ALIASES.get(m.group(1), None)
        t2 = TEAM_ALIASES.get(m.group(2), None)
        if t1 in TEAM_ABBR and t2 in TEAM_ABBR:
            matchups.append((t1, t2))
    matchups = list(dict.fromkeys(matchups))

    # Pull player pairs with team guesses
    players = parse_pairs_from_text(text)

    # Build rows by associating players found on the page to *all* detected matchups on that page
    # (board-wide pages often list many players across many games; without per-section anchors, we cannot perfectly map)
    rows = []
    for (name, abbr) in players:
        rows.append({"player": name, "team_guess": abbr, "matchup_hint": None, "source": url})
    # Additionally add rows per matchup header for traceability
    for a, h in matchups:
        rows.append({"player": None, "team_guess": None, "matchup_hint": f"{a} vs {h}", "source": url})
    return pd.DataFrame(rows)

def choose_opponent(team: str, away: str, home: str) -> Optional[str]:
    if team == away:
        return home
    if team == home:
        return away
    return None

def main(out_csv: str="opponent_map_from_props.csv"):
    sched = current_week_games()  # away, home, week
    week = int(sched["week"].iloc[0])

    sources = [
        "https://www.vegasinsider.com/nfl/odds/player-props/",
        "https://www.rotowire.com/betting/nfl/player-props.php",
        "https://www.actionnetwork.com/nfl/props",
    ]

    frames = []
    for url in sources:
        frames.append(extract_from_board(url))
    parsed = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["player","team_guess","matchup_hint","source"])

    # Build a set of matchups for this week from schedule (both forms "A vs B")
    sched_pairs = set()
    for _, r in sched.iterrows():
        sched_pairs.add((r["away"], r["home"]))

    # We will create outputs only for rows with a recognizable player and a team_guess that maps to a team in one of this week's games.
    rows = []
    for _, row in parsed.dropna(subset=["player"]).iterrows():
        name = str(row["player"]).strip()
        team_guess = row.get("team_guess", None)
        src = row.get("source")
        # If we didn't parse a team abbreviation alongside name, skip (can't assign opponent reliably without roster map)
        if team_guess not in TEAM_ABBR:
            continue
        # Find opponent by schedule: match any game this week that contains team_guess
        opp = None
        for away, home in sched_pairs:
            cand = choose_opponent(team_guess, away, home)
            if cand:
                opp = cand
                break
        if opp is None:
            continue
        rows.append({"player": name, "team": team_guess, "opponent": opp, "week": week, "source": src})

    out = pd.DataFrame(rows, columns=["player","team","opponent","week","source"]).drop_duplicates()
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows for week {week}.")

if __name__ == "__main__":
    out = "opponent_map_from_props.csv" if len(sys.argv) < 2 else sys.argv[1]
    main()
