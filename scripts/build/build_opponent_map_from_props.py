#!/usr/bin/env python3
import sys
from pathlib import Path

from io import BytesIO

import pandas as pd
import requests

UA = {"User-Agent": "FullSlate/CI (+github-actions)"}
SCHEDULE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv"
)

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


def get_nfl_schedule(season: int) -> pd.DataFrame:
    """Fetch nflverse schedule CSV and return rows for the requested season."""

    resp = requests.get(SCHEDULE_URL, headers=UA, timeout=60)
    resp.raise_for_status()

    schedule = pd.read_csv(BytesIO(resp.content))
    if "season" not in schedule.columns:
        raise RuntimeError("nflverse schedules.csv is missing the 'season' column")

    season_schedule = schedule[schedule["season"] == season].copy()
    if season_schedule.empty:
        raise RuntimeError(
            f"nflverse schedules.csv returned no games for season={season}"
        )

    return season_schedule


def expand_to_team_opp(schedule: pd.DataFrame) -> pd.DataFrame:
    """Expand schedule rows to team/opponent rows using nflverse data."""

    if schedule is None or schedule.empty:
        raise RuntimeError("nflverse schedule returned no games to expand")

    home_col = next(
        (col for col in ["team_home", "home_team", "home"] if col in schedule.columns),
        None,
    )
    away_col = next(
        (col for col in ["team_away", "away_team", "away"] if col in schedule.columns),
        None,
    )
    if not home_col or not away_col:
        raise RuntimeError("nflverse schedule is missing home/away team columns")

    if "week" not in schedule.columns:
        raise RuntimeError("nflverse schedule is missing the 'week' column")

    timestamp_col = next(
        (
            col
            for col in [
                "gameday",
                "gamedate",
                "game_day",
                "game_date",
                "kickoff",
                "game_time",
                "gametime",
            ]
            if col in schedule.columns
        ),
        None,
    )

    games = schedule[[away_col, home_col, "week", "season"]].copy()
    games.rename(columns={away_col: "away_abbr", home_col: "home_abbr"}, inplace=True)

    for col in ["away_abbr", "home_abbr"]:
        games[col] = games[col].apply(
            lambda val: TEAM_NAME_TO_ABBR.get(
                str(val).strip(), str(val).strip().upper()
            )
            if pd.notna(val)
            else val
        )

    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games = games.dropna(subset=["week"]).astype({"week": int})

    if timestamp_col:
        games["game_timestamp"] = (
            schedule.loc[games.index, timestamp_col].fillna("").astype(str)
        )
    else:
        games["game_timestamp"] = ""

    columns = ["team", "opponent", "week", "season", "game_timestamp"]
    away = games.rename(
        columns={"away_abbr": "team", "home_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]
    home = games.rename(
        columns={"home_abbr": "team", "away_abbr": "opponent"}
    )[["team", "opponent", "week", "season", "game_timestamp"]]

    team_map = pd.concat([away, home], ignore_index=True).drop_duplicates(subset=columns)

    if team_map.empty:
        raise RuntimeError("expanded nflverse schedule produced no team/opponent rows")

    return team_map


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

    schedule = get_nfl_schedule(season)
    team_map = expand_to_team_opp(schedule)

    cols = ["player", "team", "opponent", "week", "season", "game_timestamp"]
    team_map.insert(0, "player", "")
    if "game_timestamp" not in team_map:
        team_map["game_timestamp"] = ""
    out_df = team_map[cols]

    out_path = Path("data") / "opponent_map_from_props.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_df.empty:
        raise RuntimeError("opponent map DataFrame is empty; refusing to write CSV")
    out_df.to_csv(out_path, index=False)
    print(
        f"[build_opponent_map_from_props] wrote {out_path} with {len(out_df)} rows (season={season})."
    )


if __name__ == "__main__":
    main()
