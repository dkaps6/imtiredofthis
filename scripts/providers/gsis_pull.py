#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight GSIS/nflverse weekly pull for a given season.
Writes data/gsis_weekly_<season>.parquet (and CSV for transparency).
Hard clamps season to the one passed; your workflow passes 2025.
"""

import argparse
import os
import sys

import pandas as pd

try:
    import nfl_data_py as nfl
except Exception:
    nfl = None


OUT_DIR = "data"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pull_weekly(season: int) -> pd.DataFrame:
    if nfl is None:
        print("[gsis_pull] nfl_data_py not available; producing empty shell", file=sys.stderr)
        return pd.DataFrame(columns=["season"])

    print(f"[gsis_pull] importing weekly data for season={season}")
    df = nfl.import_weekly_data([season], downcast=True)

    # clamp
    if "season" in df.columns:
        df = df[df["season"].astype(int) == int(season)].copy()
    else:
        df["season"] = int(season)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int, help="Season year, e.g., 2025")
    args = parser.parse_args()

    _ensure_dir(OUT_DIR)
    try:
        df = pull_weekly(args.season)
        pq_path = os.path.join(OUT_DIR, f"gsis_weekly_{args.season}.parquet")
        csv_path = os.path.join(OUT_DIR, f"gsis_weekly_{args.season}.csv")

        if len(df):
            try:
                df.to_parquet(pq_path, index=False)
            except Exception:
                pass
            df.to_csv(csv_path, index=False)
            print(f"[gsis_pull] wrote {csv_path} rows={len(df)}")
        else:
            print("[gsis_pull] no rows; wrote nothing")
    except Exception as e:
        print(f"[gsis_pull] ERROR: {e}", file=sys.stderr)
        # keep CI alive if this is optional in your job
        raise


if __name__ == "__main__":
    main()
