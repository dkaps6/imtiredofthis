#!/usr/bin/env python3
# build_coverage_penalties.py
#
# Produce weekly defensive penalties from free play-by-play (2025 season).
# Output CSV schema (one row per DEF team per week):
#   team,week,def_penalties,def_penalty_yards,def_dpi_count,def_holding_count
#
# Free sources (cite in your docs/readme):
# - nflverse/nflfastR play-by-play 2025 (public GitHub release asset):
#   https://github.com/nflverse/nflverse-data/releases/download/pbp/pbp_2025.csv.gz
# - PBP field dictionary (penalty, penalty_yards, penalty_team, defteam, week, desc):
#   https://nflreadr.nflverse.com/articles/dictionary_pbp.html
#
# Notes:
# - We count only **defensive** penalties (credited to the defense).
#   Primary key is penalty_team == defteam when penalty == 1.
#   If penalty_team is missing, we infer defensive responsibility via text patterns in `desc`
#   (e.g., "defensive holding", "defensive pass interference").
# - DPI detection: case-insensitive match for "pass interference" on plays flagged as defensive.
# - Defensive holding detection: case-insensitive match for "defensive holding".
#
import sys

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025
from scripts.utils.pbp_threshold import get_dynamic_min_rows


def main(out_csv: str = "coverage_penalties.csv"):
    min_rows = get_dynamic_min_rows()
    pbp = get_pbp_2025(min_rows=min_rows)
    print(f"[coverage_penalties] PBP rows: {len(pbp)} (min_rows={min_rows})")
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        raise RuntimeError("PBP fetch failed")

    needed = ["week", "defteam", "penalty", "penalty_yards", "penalty_team", "desc"]
    for c in needed:
        if c not in pbp.columns:
            raise RuntimeError(f"Required column missing: {c}")
    # Normalize
    pbp["penalty"] = (
        pd.to_numeric(pbp["penalty"], errors="coerce").fillna(0).astype(int)
    )
    pbp["penalty_yards"] = (
        pd.to_numeric(pbp["penalty_yards"], errors="coerce").fillna(0).astype(int)
    )
    pbp["desc"] = pbp["desc"].astype(str)
    pbp["penalty_team"] = pbp["penalty_team"].astype(str)

    # Consider only flagged plays
    flags = pbp[pbp["penalty"] == 1].copy()

    # Determine defensive penalties
    # Primary: penalty_team matches defteam
    def_flag = flags[flags["penalty_team"] == flags["defteam"]].copy()

    # Secondary: missing/blank penalty_team but text implies defensive foul
    missing_team = flags[~(flags["penalty_team"] == flags["defteam"])].copy()
    # Sometimes penalty_team is 'nan' as string; treat as missing
    def_text_mask = (
        missing_team["desc"]
        .str.lower()
        .str.contains(
            r"(defensive\s+holding|defensive\s+pass\s+interference|illegal\s+contact|defensive\s+offside|neutral\s+zone\s+infraction|roughing\s+the\s+passer|illegal\s+hands\s+to\s+the\s+face)",
            regex=True,
        )
    )
    inferred_def = missing_team[def_text_mask].copy()
    def_all = pd.concat([def_flag, inferred_def], ignore_index=True)

    # DPI and Defensive Holding counts (on defensive penalties only)
    low = def_all["desc"].str.lower()
    dpi_mask = low.str.contains(r"pass\s+interference")
    dh_mask = low.str.contains(r"defensive\s+holding")

    # Aggregate by team-week
    grp = def_all.groupby(["defteam", "week"], dropna=False)
    out = (
        grp.agg(
            def_penalties=("penalty", "size"),
            def_penalty_yards=("penalty_yards", "sum"),
            def_dpi_count=("penalty", lambda s: int(dpi_mask.loc[s.index].sum())),
            def_holding_count=("penalty", lambda s: int(dh_mask.loc[s.index].sum())),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # Sort & write
    out = out.sort_values(["team", "week"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")


if __name__ == "__main__":
    out = "coverage_penalties.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out)
