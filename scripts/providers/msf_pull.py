#!/usr/bin/env python3
"""
MySportsFeeds v2.1 pull.
ENV: MSF_API_KEY
Outputs:
  data/msf_team.csv    (placeholder)
  data/msf_player.csv  with: player, team, route_rate?, target_share?, rush_share?, yprr?, ypc, ypa
Note: exact availability varies by plan; we fill what we can and leave others NaN.
"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

def main(season: int):
    key = os.getenv("MSF_API_KEY", "")
    if not key:
        print("[msf_pull] MSF_API_KEY missing; skipping.")
        return 0

    try:
        from mysportsfeeds import MySportsFeeds
    except Exception as e:
        print("[msf_pull] mysportsfeeds package not installed:", e)
        return 0

    msf = MySportsFeeds(version="2.1", verbose=False)
    # For v2.1, password is usually "MYSPORTSFEEDS"
    msf.authenticate(key, "MYSPORTSFEEDS")

    # Season format e.g. '2025-regular'
    season_str = f"{season}-regular"

    # Player game logs can provide rushing/receiving/passing stats for aggregation
    try:
        print(f"[msf_pull] fetching gamelogs for {season_str}…")
        data = msf.msf_get_data(league="nfl", season=season_str, feed="player_gamelogs", format="json")
    except Exception as e:
        print("[msf_pull] API call failed:", e)
        return 0

    try:
        games = data.get("gamelogs", [])
        rows = []
        for g in games:
            pl = g.get("player", {})
            team = g.get("team", {}).get("abbreviation", "")
            stats = g.get("stats", {})
            name = pl.get("firstName","") + " " + pl.get("lastName","")
            rows.append({
                "player": name.strip(),
                "team": str(team).upper(),
                "targets": stats.get("Receiving", {}).get("Targets", {}).get("#text", 0),
                "receptions": stats.get("Receiving", {}).get("Receptions", {}).get("#text", 0),
                "receiving_yards": stats.get("Receiving", {}).get("Yards", {}).get("#text", 0),
                "rushing_attempts": stats.get("Rushing", {}).get("Attempts", {}).get("#text", 0),
                "rushing_yards": stats.get("Rushing", {}).get("Yards", {}).get("#text", 0),
                "passing_yards": stats.get("Passing", {}).get("Yards", {}).get("#text", 0),
                "passing_attempts": stats.get("Passing", {}).get("Attempts", {}).get("#text", 0),
            })
        df = pd.DataFrame(rows)
        for c in ("targets","receptions","receiving_yards","rushing_attempts","rushing_yards","passing_yards","passing_attempts"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        team_game = df.groupby(["team"], as_index=False).agg(
            team_targets=("targets","sum"),
            team_rush_att=("rushing_attempts","sum"),
        )
        agg = df.groupby(["team","player"], as_index=False).agg(
            targets=("targets","sum"),
            rush_att=("rushing_attempts","sum"),
            rec_yds=("receiving_yards","sum"),
            rush_yds=("rushing_yards","sum"),
            pass_yds=("passing_yards","sum"),
            pass_att=("passing_attempts","sum"),
        ).merge(team_game, on="team", how="left")

        agg["target_share"] = agg["targets"] / agg["team_targets"].replace(0, np.nan)
        agg["rush_share"]   = agg["rush_att"] / agg["team_rush_att"].replace(0, np.nan)
        agg["yprr"]         = np.nan  # not available directly
        agg["ypc"]          = agg["rush_yds"] / agg["rush_att"].replace(0, np.nan)
        agg["ypa"]          = agg["pass_yds"] / agg["pass_att"].replace(0, np.nan)
        agg["route_rate"]   = np.nan

        outp = agg[["player","team","route_rate","target_share","rush_share","yprr","ypc","ypa"]]
        Path("data").mkdir(exist_ok=True)
        pd.DataFrame(columns=["team"]).to_csv("data/msf_team.csv", index=False)  # placeholder
        outp.to_csv("data/msf_player.csv", index=False)
        print(f"[msf_pull] wrote data/msf_player.csv rows={len(outp)}")
    except Exception as e:
        print("[msf_pull] parse/normalize failed:", e)
    return 0

if __name__ == "__main__":
    season = int(os.getenv("SEASON", "2025"))
    sys.exit(main(season))
