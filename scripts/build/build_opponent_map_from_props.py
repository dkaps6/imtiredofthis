#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
from nfl_data_py import import_schedules

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
    df = import_schedules([season])
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG"])]
    rename_map = {
        "home_team": "team_home",
        "away_team": "team_away",
        "game_date": "gameday",
        "venue": "stadium",
        "site_city": "location",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    required = ["week", "team_home", "team_away"]
    missing = [c for c in required if c not in df.columns]
    if missing or df.empty:
        raise RuntimeError(
            f"Schedule missing columns {missing} or empty for season {season}"
        )
    keep = [
        "week",
        "team_home",
        "team_away",
    ] + [c for c in ["game_id", "gameday", "stadium", "location"] if c in df.columns]
    return df[keep].reset_index(drop=True)


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
    schedule = schedule.copy()
    schedule["season"] = season
    team_map = expand_to_team_opp(schedule)

    roles_path = Path("data") / "roles_ourlads.csv"
    players = None
    if roles_path.exists():
        try:
            players = pd.read_csv(roles_path)
        except Exception as e:
            print(f"[build_opponent_map_from_props] failed to load roles_ourlads.csv: {e}")
            players = None

    if (
        players is not None
        and not players.empty
        and {"team", "player"}.issubset(players.columns)
    ):
        print(
            "[build_opponent_map_from_props] expanding per-player opponent map using roles_ourlads.csv "
            f"({len(players)} players)"
        )

        # keep just team + player, normalize team code
        players = (
            players.loc[:, ["team", "player"]]
            .dropna(subset=["team", "player"])
            .copy()
        )
        players["team"] = (
            players["team"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # normalize team/opponent/week on the schedule side before merge
        team_map["team"] = (
            team_map["team"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        team_map["opponent"] = (
            team_map["opponent"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # explode schedule per player
        merged = team_map.merge(players, on="team", how="left")

        # keep week, do NOT drop it
        out_df = merged.loc[
            :,
            ["player", "team", "opponent", "week", "season", "game_timestamp"],
        ].copy()

    else:
        print(
            "[build_opponent_map_from_props] WARNING: roles_ourlads.csv missing or invalid, writing team-only mapping"
        )
        team_only = team_map.loc[
            :, ["team", "opponent", "week", "season", "game_timestamp"]
        ].copy()
        team_only.insert(0, "player", "")
        out_df = team_only

    # final cleanup
    out_df = out_df.dropna(subset=["team", "opponent", "week"]).reset_index(drop=True)

    out_path = Path("data") / "opponent_map_from_props.csv"
    out_df.to_csv(out_path, index=False)
    print(
        f"[build_opponent_map_from_props] wrote {len(out_df)} player/opponent rows → {out_path}"
    )

    # debug sample
    debug_dir = out_path.parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "opponent_sample.csv").write_text(
        out_df.head(50).to_csv(index=False)
    )
    print(
        f"[build_opponent_map_from_props] wrote debug sample → {debug_dir/'opponent_sample.csv'}"
    )

if __name__ == "__main__":
    main()
