#!/usr/bin/env python3
"""
Materialize GSIS-style fallbacks using nfl_data_py.
Outputs:
  data/gsis_team.csv      (currently empty shell; extend if needed)
  data/gsis_player.csv    with: player, team, route_rate, target_share, rush_share, yprr, ypc, ypa
"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

def main(season: int):
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print("[gsis_pull] nfl_data_py not available:", e)
        return 0

    # Pull weekly player stats and PBP-derived targets
    print(f"[gsis_pull] fetching weekly player data for {season}…")
    wk = nfl.import_weekly_data([season])  # can be large
    if wk is None or wk.empty:
        print("[gsis_pull] weekly_data empty; writing empty CSVs")
        Path("data").mkdir(exist_ok=True)
        pd.DataFrame(columns=["team"]).to_csv("data/gsis_team.csv", index=False)
        pd.DataFrame(columns=["player","team"]).to_csv("data/gsis_player.csv", index=False)
        return 0

    # Normalize names/teams
    wk["player_display_name"] = wk["player_display_name"].astype(str)
    wk["recent_team"] = wk["recent_team"].astype(str).str.upper()

    # Approximations:
    # - target_share: sum(targets)/sum(team pass attempts for games where player active)
    # - rush_share:   sum(rush_att)/sum(team rush attempts)
    # - route_rate:   routes_per_dropback if available; else NaN (nfl_data_py may not have routes)
    cols = {
        "targets":"targets", "receptions":"receptions",
        "attempts":"attempts", "completions":"completions",
        "passing_yards":"passing_yards", "rushing_yards":"rushing_yards",
        "rushing_attempts":"rushing_attempts", "receiving_yards":"receiving_yards",
    }
    for c in cols.values():
        if c not in wk.columns:
            wk[c] = 0

    # team-level denominators per game
    team_game = wk.groupby(["team","week"], as_index=False).agg(
        team_targets=("targets","sum"),
        team_rush_att=("rushing_attempts","sum"),
        team_pass_att=("attempts","sum"),
    )
    wk = wk.rename(columns={"recent_team":"team","player_display_name":"player"})
    wk = wk.merge(team_game, on=["team","week"], how="left")

    agg = wk.groupby(["team","player"], as_index=False).agg(
        targets=("targets","sum"),
        rush_att=("rushing_attempts","sum"),
        rec_yds=("receiving_yards","sum"),
        rush_yds=("rushing_yards","sum"),
        pass_yards=("passing_yards","sum"),
        att=("attempts","sum"),
        team_targets=("team_targets","sum"),
        team_rush_att=("team_rush_att","sum"),
        team_pass_att=("team_pass_att","sum"),
    )

    agg["target_share"] = (agg["targets"] / agg["team_targets"].replace(0,np.nan))
    agg["rush_share"]   = (agg["rush_att"] / agg["team_rush_att"].replace(0,np.nan))
    agg["yprr"]         = np.nan  # not available here reliably
    agg["ypc"]          = agg["rush_yds"] / agg["rush_att"].replace(0,np.nan)
    agg["ypa"]          = agg["pass_yards"] / agg["att"].replace(0,np.nan)
    agg["route_rate"]   = np.nan

    outp = agg[["player","team","route_rate","target_share","rush_share","yprr","ypc","ypa"]].copy()
    Path("data").mkdir(exist_ok=True)
    pd.DataFrame(columns=["team"]).to_csv("data/gsis_team.csv", index=False)  # placeholder
    outp.to_csv("data/gsis_player.csv", index=False)
    print(f"[gsis_pull] wrote data/gsis_player.csv rows={len(outp)}")
    return 0

if __name__ == "__main__":
    season = int(os.getenv("SEASON", "2025"))
    sys.exit(main(season))
