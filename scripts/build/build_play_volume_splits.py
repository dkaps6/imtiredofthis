#!/usr/bin/env python3
# build_play_volume_splits.py
#
# Output: play_volume_splits.csv with columns:
#   team,week,plays_offense,plays_defense,seconds_per_play,neutral_situation_rate
#
# Free data source:
# - nflverse/nflfastR play-by-play 2025 season CSV (public GitHub Releases)
#   https://github.com/nflverse/nflverse-data/releases  (pbp_2025.csv.gz)
#   Dictionary / fields (qtr, score_differential, posteam, defteam, qb_dropback, rush_attempt, game_id, play_id, game_seconds_remaining):
#   https://nflreadr.nflverse.com/articles/dictionary_pbp.html
#   https://nflfastr.com/reference/fast_scraper.html
#
# Metric definitions:
# - Offensive plays: plays where (qb_dropback==1) OR (rush_attempt==1), counted for posteam.
# - Defensive plays: same plays counted for defteam.
# - Seconds per play (team offense): for each team-week, sort offensive plays by (game_id, play_id)
#   and sum deltas of game_seconds_remaining BETWEEN that team's consecutive offensive plays;
#   divide by offensive plays count. This approximates time between snaps including stoppages.
# - Neutral situation rate (offense): share of offensive plays taken when |score_differential| <= 7 AND qtr in {1,2,3}.
#
import sys

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025
from scripts.utils.pbp_threshold import (
    enforce_min_rows,
    get_dynamic_min_rows,
)


def compute_seconds_per_play(off_df: pd.DataFrame) -> float:
    """Approximate seconds per offensive play using game_seconds_remaining deltas between consecutive plays by same team & game."""
    if off_df.empty:
        return np.nan
    d = off_df.sort_values(["game_id", "play_id"]).copy()
    # compute delta within each game for the same team
    d["prev_sec"] = d.groupby("game_id")["game_seconds_remaining"].shift(1)
    # We only want deltas between consecutive OFFENSIVE plays; ensure previous row is same team's offense
    d["prev_posteam"] = d.groupby("game_id")["posteam"].shift(1)
    dd = d[d["prev_posteam"] == d["posteam"]].copy()
    dd["delta"] = dd["prev_sec"] - dd["game_seconds_remaining"]
    dd = dd[
        (dd["delta"].notna()) & (dd["delta"] >= 0) & (dd["delta"] < 600)
    ]  # guardrails
    if dd.empty:
        return np.nan
    return float(dd["delta"].mean())


def main(out_csv: str = "play_volume_splits.csv"):
    min_rows_target = get_dynamic_min_rows()
    pbp = get_pbp_2025(min_rows=20000)
    print(f"[play_volume_splits] PBP rows: {len(pbp)} (soft target {min_rows_target})")
    enforce_min_rows(pbp, min_rows_target)
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        raise RuntimeError("PBP fetch failed")

    # ensure needed columns exist
    needed = [
        "week",
        "qtr",
        "score_differential",
        "posteam",
        "defteam",
        "qb_dropback",
        "rush_attempt",
        "game_id",
        "play_id",
        "game_seconds_remaining",
    ]
    for c in needed:
        if c not in pbp.columns:
            raise RuntimeError(f"Required column missing: {c}")

    # Normalize numeric flags
    for col in ["qb_dropback", "rush_attempt", "qtr"]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0).astype(int)
    pbp["score_differential"] = pd.to_numeric(
        pbp["score_differential"], errors="coerce"
    )

    # Valid offensive plays
    pbp["is_off_play"] = (
        (pbp["qb_dropback"] == 1) | (pbp["rush_attempt"] == 1)
    ).astype(int)

    # Compute per team-week
    rows = []
    weeks = sorted(pbp["week"].dropna().unique().tolist())
    teams = sorted(
        set(pbp["posteam"].dropna().unique().tolist())
        | set(pbp["defteam"].dropna().unique().tolist())
    )
    for wk in weeks:
        wk_df = pbp[pbp["week"] == wk]
        for tm in teams:
            off = wk_df[(wk_df["posteam"] == tm) & (wk_df["is_off_play"] == 1)]
            deff = wk_df[(wk_df["defteam"] == tm) & (wk_df["is_off_play"] == 1)]

            plays_offense = int(len(off))
            plays_defense = int(len(deff))

            # seconds per play (offense, all situations)
            spp = compute_seconds_per_play(off)

            # neutral situation rate (offense): |score_differential| <= 7 & qtr in {1,2,3}
            if plays_offense > 0:
                neutral_mask = (off["score_differential"].abs() <= 7) & (
                    off["qtr"].isin([1, 2, 3])
                )
                neutral_rate = float(neutral_mask.mean())
            else:
                neutral_rate = np.nan

            rows.append(
                {
                    "team": tm,
                    "week": int(wk),
                    "plays_offense": plays_offense,
                    "plays_defense": plays_defense,
                    "seconds_per_play": None if pd.isna(spp) else round(spp, 2),
                    "neutral_situation_rate": (
                        None if pd.isna(neutral_rate) else round(neutral_rate, 4)
                    ),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "team",
            "week",
            "plays_offense",
            "plays_defense",
            "seconds_per_play",
            "neutral_situation_rate",
        ],
    )
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")


if __name__ == "__main__":
    out = "play_volume_splits.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out)
