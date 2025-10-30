#!/usr/bin/env python3
# build_run_pass_funnel.py
#
# Compute weekly defensive splits from free nflverse/nflfastR play-by-play (2025)
# and derive opponent funnel indicators:
#   team,week,opp_run_lean,opp_pass_lean,run_success_allowed,pass_success_allowed,run_epa_allowed,pass_epa_allowed
#
# Definitions (transparent & reproducible)
# - run_success_allowed: share of rush plays vs this defense with EPA > 0 in that week.
# - pass_success_allowed: share of pass plays vs this defense with EPA > 0 in that week.
#   (Success = EPA > 0 is a common convention in the analytics community.)
# - run_epa_allowed / pass_epa_allowed: mean EPA/play allowed by this defense on rush / pass.
# - opp_run_lean: (team's run_epa_allowed) minus the league-average run EPA allowed **that week**.
# - opp_pass_lean: (team's pass_epa_allowed) minus the league-average pass EPA allowed **that week**.
#   Positive values indicate the defense is worse than league average in that phase ⇒ an "advantage" for the opponent to attack there.
#
# Free sources:
# - Play-by-play release (pbp_2025.csv.gz): https://github.com/nflverse/nflverse-data/releases
#   Direct asset (subject to release tag): https://github.com/nflverse/nflverse-data/releases/download/pbp/pbp_2025.csv.gz
# - Field dictionary (pass_attempt, rush_attempt, epa, score_differential, etc.):
#   https://nflreadr.nflverse.com/articles/dictionary_pbp.html
#
import sys

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025
from scripts.utils.pbp_threshold import (
    enforce_min_rows,
    get_dynamic_min_rows,
)


def main(out_csv: str = "run_pass_funnel.csv"):
    min_rows_target = get_dynamic_min_rows()
    pbp = get_pbp_2025(min_rows=20000)
    print(f"[run_pass_funnel] PBP rows: {len(pbp)} (soft target {min_rows_target})")
    enforce_min_rows(pbp, min_rows_target)
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        raise RuntimeError("PBP fetch failed")

    # Required columns
    required = ["week", "defteam", "pass_attempt", "rush_attempt", "epa"]
    for c in required:
        if c not in pbp.columns:
            raise RuntimeError(f"Missing required column: {c}")
    # normalize flags
    for c in ["pass_attempt", "rush_attempt"]:
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
    pbp["epa"] = pd.to_numeric(pbp["epa"], errors="coerce")

    # Filter to offensive snaps (rush or pass attempt)
    plays = pbp[(pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1)].copy()

    rows = []
    for wk, wdf in plays.groupby("week"):
        # League averages for week
        league_run = wdf[wdf["rush_attempt"] == 1]["epa"].mean()
        league_pass = wdf[wdf["pass_attempt"] == 1]["epa"].mean()

        for team, tdf in wdf.groupby("defteam"):
            r = tdf[tdf["rush_attempt"] == 1]
            p = tdf[tdf["pass_attempt"] == 1]

            # Success rates (EPA > 0)
            run_sr = np.nan if r.empty else float((r["epa"] > 0).mean())
            pass_sr = np.nan if p.empty else float((p["epa"] > 0).mean())

            # EPA allowed
            run_epa = np.nan if r.empty else float(r["epa"].mean())
            pass_epa = np.nan if p.empty else float(p["epa"].mean())

            # Opponent leans relative to league weekly average
            opp_run_lean = np.nan if pd.isna(run_epa) else float(run_epa - league_run)
            opp_pass_lean = (
                np.nan if pd.isna(pass_epa) else float(pass_epa - league_pass)
            )

            rows.append(
                {
                    "team": team,
                    "week": int(wk),
                    "opp_run_lean": (
                        None if pd.isna(opp_run_lean) else round(opp_run_lean, 4)
                    ),
                    "opp_pass_lean": (
                        None if pd.isna(opp_pass_lean) else round(opp_pass_lean, 4)
                    ),
                    "run_success_allowed": (
                        None if pd.isna(run_sr) else round(run_sr, 4)
                    ),
                    "pass_success_allowed": (
                        None if pd.isna(pass_sr) else round(pass_sr, 4)
                    ),
                    "run_epa_allowed": None if pd.isna(run_epa) else round(run_epa, 4),
                    "pass_epa_allowed": (
                        None if pd.isna(pass_epa) else round(pass_epa, 4)
                    ),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "team",
            "week",
            "opp_run_lean",
            "opp_pass_lean",
            "run_success_allowed",
            "pass_success_allowed",
            "run_epa_allowed",
            "pass_epa_allowed",
        ],
    ).sort_values(["team", "week"])
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows.")


if __name__ == "__main__":
    out = "run_pass_funnel.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out)
