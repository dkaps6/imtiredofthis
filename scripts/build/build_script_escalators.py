#!/usr/bin/env python3
# build_script_escalators.py
#
# Compute weekly team game-script shares from free play-by-play (2025):
#   team,week,lead_script_pct,trail_script_pct,neutral_script_pct,garbage_time_pct
#
# Definitions (explicit):
# - Offensive play: qb_dropback == 1 OR rush_attempt == 1 (excludes no-plays/penalties).
# - Lead script: score_differential > +7 (posteam leading by more than one score), excluding garbage time.
# - Trail script: score_differential < -7 (posteam trailing by more than one score), excluding garbage time.
# - Neutral script: |score_differential| <= 7 AND quarter in {1,2,3}, excluding garbage time.
# - Garbage time: quarter == 4 AND |score_differential| > 16.
# Notes: Plays in Q4 with |score_differential| <= 16 that are not >+7 or <-7 are not counted in neutral;
#        shares may not sum exactly to 1.0 by design (transparent by definition).
#
# Free sources to cite:
# - nflverse/nflfastR 2025 PBP (public GitHub release asset): https://github.com/nflverse/nflverse-data/releases/download/pbp/pbp_2025.csv.gz
# - PBP field dictionary (qtr, score_differential, posteam, qb_dropback, rush_attempt): https://nflreadr.nflverse.com/articles/dictionary_pbp.html
#
import sys

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025


def main(out_csv: str = "script_escalators.csv"):
    pbp = get_pbp_2025()
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        raise RuntimeError("PBP fetch failed")

    needed = [
        "week",
        "qtr",
        "score_differential",
        "posteam",
        "qb_dropback",
        "rush_attempt",
    ]
    for c in needed:
        if c not in pbp.columns:
            raise RuntimeError(f"Required column missing: {c}")
    # Normalize
    pbp["qtr"] = pd.to_numeric(pbp["qtr"], errors="coerce").fillna(0).astype(int)
    pbp["score_differential"] = pd.to_numeric(
        pbp["score_differential"], errors="coerce"
    )
    for col in ["qb_dropback", "rush_attempt"]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0).astype(int)

    # Offensive snaps
    off = pbp[(pbp["qb_dropback"] == 1) | (pbp["rush_attempt"] == 1)].copy()
    off = off.dropna(subset=["posteam", "week"])

    # Flags
    off["is_garbage"] = (
        (off["qtr"] == 4) & (off["score_differential"].abs() > 16)
    ).astype(int)
    # Exclude garbage when tagging lead/trail/neutral
    non_garb = off["is_garbage"] == 0
    off["is_lead"] = ((off["score_differential"] > 7) & non_garb).astype(int)
    off["is_trail"] = ((off["score_differential"] < -7) & non_garb).astype(int)
    off["is_neutral"] = (
        (off["score_differential"].abs() <= 7) & (off["qtr"].isin([1, 2, 3])) & non_garb
    ).astype(int)

    rows = []
    for (team, wk), dfw in off.groupby(["posteam", "week"]):
        n = len(dfw)
        lead = dfw["is_lead"].mean() if n else np.nan
        trail = dfw["is_trail"].mean() if n else np.nan
        neutral = dfw["is_neutral"].mean() if n else np.nan
        garbage = dfw["is_garbage"].mean() if n else np.nan
        rows.append(
            {
                "team": team,
                "week": int(wk),
                "lead_script_pct": None if pd.isna(lead) else round(float(lead), 4),
                "trail_script_pct": None if pd.isna(trail) else round(float(trail), 4),
                "neutral_script_pct": (
                    None if pd.isna(neutral) else round(float(neutral), 4)
                ),
                "garbage_time_pct": (
                    None if pd.isna(garbage) else round(float(garbage), 4)
                ),
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "team",
            "week",
            "lead_script_pct",
            "trail_script_pct",
            "neutral_script_pct",
            "garbage_time_pct",
        ],
    ).sort_values(["team", "week"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")


if __name__ == "__main__":
    out = "script_escalators.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out)
