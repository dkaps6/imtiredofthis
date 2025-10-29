#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

APISPORTS_KEY = os.environ.get("APISPORTS_KEY")
UA = {"User-Agent": "FullSlate/CI (+github-actions)"}

TEAM_NAME_TO_ABBR = {
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


def fetch_games_apisports(season: int) -> pd.DataFrame:
    """Fetch NFL games for a season using the API-Sports American Football endpoint."""
    columns = ["away_abbr", "home_abbr", "week", "season", "game_timestamp"]
    if not APISPORTS_KEY:
        return pd.DataFrame(columns=columns)

    url = "https://v1.american-football.api-sports.io/games"
    params = {"league": "NFL", "season": season}

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params=params,
                headers={**UA, "x-apisports-key": APISPORTS_KEY},
                timeout=45,
            )
            resp.raise_for_status()
            rows = []
            for game in resp.json().get("response", []):
                away_name = game.get("teams", {}).get("away", {}).get("name")
                home_name = game.get("teams", {}).get("home", {}).get("name")
                week_raw = game.get("week") or game.get("round") or 0
                try:
                    week_val = int(week_raw)
                except (TypeError, ValueError):
                    week_val = 0
                game_timestamp = game.get("date") or game.get("game") or ""
                away = TEAM_NAME_TO_ABBR.get(away_name)
                home = TEAM_NAME_TO_ABBR.get(home_name)
                if away and home and week_val:
                    rows.append(
                        {
                            "away_abbr": away,
                            "home_abbr": home,
                            "week": week_val,
                            "season": season,
                            "game_timestamp": game_timestamp,
                        }
                    )
            return pd.DataFrame(rows, columns=columns)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame(columns=columns)


def expand_to_team_opp(games: pd.DataFrame) -> pd.DataFrame:
    """Expand away/home games to team/opponent rows."""
    columns = ["team", "opponent", "week", "season", "game_timestamp"]
    if games is None or games.empty:
        return pd.DataFrame(columns=columns)

    away = games.rename(
        columns={"away_abbr": "team", "home_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]
    home = games.rename(
        columns={"home_abbr": "team", "away_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]
    return pd.concat([away, home], ignore_index=True).drop_duplicates(subset=columns)


def infer_default_season() -> int:
    """Return a reasonable default NFL season based on today's date."""
    today = pd.Timestamp.utcnow()
    return today.year if today.month >= 3 else today.year - 1


def main() -> None:
    # Allow optional season argument (first numeric argument wins)
    season = None
    for arg in sys.argv[1:]:
        try:
            season = int(arg)
            break
        except (TypeError, ValueError):
            continue
    if season is None:
        season = infer_default_season()

    games = fetch_games_apisports(season)
    team_map = expand_to_team_opp(games)

    cols = ["player", "team", "opponent", "week", "season", "game_timestamp"]
    if team_map.empty:
        out_df = pd.DataFrame(columns=cols)
    else:
        team_map.insert(0, "player", "")
        if "game_timestamp" not in team_map:
            team_map["game_timestamp"] = ""
        out_df = team_map[cols]

    out_path = Path("data") / "opponent_map_from_props.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(
        f"[build_opponent_map_from_props] wrote {out_path} with {len(out_df)} rows (season={season})."
    )


if __name__ == "__main__":
    main()
