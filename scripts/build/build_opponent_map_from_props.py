#!/usr/bin/env python3
# build_opponent_map_from_props.py
# Build opponent mapping by week from APISports (no scraping).
# Output: data/opponent_map_from_props.csv -> player,team,week,opponent (props step may fill player/opp later)

import os, sys, time, json, requests, pandas as pd
from pathlib import Path

APISPORTS_KEY = os.environ.get("APISPORTS_KEY")
USER_AGENT = {"User-Agent": "FullSlate/CI (+github-actions)"}

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA",
    "San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

def fetch_games_apisports(season: int) -> pd.DataFrame:
    if not APISPORTS_KEY:
        print("[opponent_map] WARN: APISPORTS_KEY missing; writing header-only CSV.")
        return pd.DataFrame(columns=["away_abbr","home_abbr","week","season"])
    url = "https://v1.american-football.api-sports.io/games"
    params = {"league":"NFL","season":season}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers={**USER_AGENT, "x-apisports-key": APISPORTS_KEY}, timeout=45)
            r.raise_for_status()
            data = r.json().get("response", [])
            rows = []
            for g in data:
                try:
                    away = TEAM_NAME_TO_ABBR.get(g["teams"]["away"]["name"])
                    home = TEAM_NAME_TO_ABBR.get(g["teams"]["home"]["name"])
                    wk = int(g.get("week") or g.get("round") or 0)
                    if away and home and wk:
                        rows.append({"away_abbr":away,"home_abbr":home,"week":wk,"season":season})
                except Exception:
                    continue
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[opponent_map] attempt {attempt+1}/3 error: {e}")
            time.sleep(2*(attempt+1))
    return pd.DataFrame(columns=["away_abbr","home_abbr","week","season"])

def expand_to_team_opp(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["team","opponent","week","season"])
    a = games.rename(columns={"away_abbr":"team","home_abbr":"opponent"})[["team","opponent","week","season"]]
    b = games.rename(columns={"home_abbr":"team","away_abbr":"opponent"})[["team","opponent","week","season"]]
    return pd.concat([a,b], ignore_index=True).drop_duplicates()

def main(season: int):
    games = fetch_games_apisports(season)
    team_opp = expand_to_team_opp(games)
    out = Path("data") / "opponent_map_from_props.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    team_opp.to_csv(out, index=False)
    print(f"[opponent_map] wrote {out} ({len(team_opp)} rows).")

if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    main(season)
