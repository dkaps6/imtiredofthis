#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSIS/nfl_data_py “light” pull.
- Imports weekly data for a season (your workflow passes 2025).
- Writes:
    data/gsis_weekly_<season>.csv (existing)
    data/nflgsis_team_form.csv    (schema-only unless you add real mappings)
    data/nflgsis_player_form.csv  (schema-only unless you add real mappings)

If you later decide to aggregate real team/player metrics from weekly data,
map into the same schemas where marked.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except Exception:
    nfl = None

DATA_DIR = "data"

TEAM_COLS = [
    "team","def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
    "light_box_rate","heavy_box_rate"
]
PLAYER_COLS = [
    "player","team",
    "tgt_share","route_rate","rush_share",
    "yprr","ypt","ypc","ypa",
    "receptions_per_target",
    "rz_share","rz_tgt_share","rz_rush_share",
]

def _safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def _write_csv_with_schema(path: str, df: pd.DataFrame, cols):
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out.columns = [c.lower() for c in out.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(path, index=False)
    return out

def pull_weekly(season: int) -> pd.DataFrame:
    if nfl is None:
        print("[gsis_pull] nfl_data_py not available; producing empty shell", file=sys.stderr)
        return pd.DataFrame(columns=["season"])
    print(f"[gsis_pull] importing weekly data for season={season}")
    df = nfl.import_weekly_data([season], downcast=True)
    if "season" in df.columns:
        df = df[df["season"].astype(int) == int(season)].copy()
    else:
        df["season"] = int(season)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, help="Season year, e.g., 2025")
    args = parser.parse_args()

    _safe_mkdir(DATA_DIR)

    # weekly dump (existing behavior)
    try:
        wk = pull_weekly(args.season)
        csv_path = os.path.join(DATA_DIR, f"gsis_weekly_{args.season}.csv")
        if len(wk):
            wk.to_csv(csv_path, index=False)
            print(f"[gsis_pull] wrote {csv_path} rows={len(wk)}")
        else:
            print("[gsis_pull] no rows; wrote nothing")
    except Exception as e:
        print(f"[gsis_pull] ERROR (weekly): {e}", file=sys.stderr)

    # team/player enrichers — schema-only unless you add mapping below
    team_df = pd.DataFrame(columns=["team"])
    player_df = pd.DataFrame(columns=["player","team"])

    # TODO (optional): derive real aggregates from `wk` and fill columns before writing.
    _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_team_form.csv"), team_df, TEAM_COLS)
    _write_csv_with_schema(os.path.join(DATA_DIR,"nflgsis_player_form.csv"), player_df, PLAYER_COLS)
    print("[gsis_pull] wrote nflgsis_team_form.csv & nflgsis_player_form.csv (schema)")

if __name__ == "__main__":
    main()
