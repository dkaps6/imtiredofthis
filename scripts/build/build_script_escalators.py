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
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025
from scripts.utils.pbp_threshold import get_dynamic_min_rows


def _maybe_warn(df: pd.DataFrame, pbp: pd.DataFrame, label: str) -> None:
    total_games = 0
    if "game_id" in pbp.columns:
        total_games = int(pbp["game_id"].dropna().nunique())
    elif {"week", "posteam"}.issubset(pbp.columns):
        total_games = int(pbp[["week", "posteam"]].dropna().drop_duplicates().shape[0] // 2)
    min_dynamic = max(2000, total_games * 150)
    if len(df) < min_dynamic:
        print(
            f"[builder WARNING] {label} low sample size ({len(df)} rows < {min_dynamic}), writing partial output anyway"
        )


def main(out_csv: str = str(Path("data") / "script_escalators.csv")):
    min_rows_target = get_dynamic_min_rows()
    pbp = get_pbp_2025(min_rows=20000)
    print(
        f"[script_escalators] PBP loaded rows: {len(pbp)} (soft target {min_rows_target})"
    )
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        print(
            "[builder WARNING] script_escalators.csv generated from limited PBP sample (<=1000 rows)"
        )

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
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_warn(out, pbp, out_path.name)
    out.to_csv(out_path, index=False)
    print(f"[builder] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    out = (
        str(Path("data") / "script_escalators.csv")
        if len(sys.argv) < 2
        else sys.argv[1]
    )
    main(out)
