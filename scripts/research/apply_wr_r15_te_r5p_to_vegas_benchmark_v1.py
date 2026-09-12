#!/usr/bin/env python3
"""Apply the promoted WR-R15/TE-R5P entitlement models to the existing
full-stack Vegas benchmark cohort, closing the gap disclosed in
data/backtests/full_stack_vegas_benchmark_v1/README.md ("WR-R15/TE-R5P are
not yet included in the non_qb numbers").

Research only. Reuses the exact grading arithmetic from
scripts/backtest/grade_full_stack_vegas_benchmark_v1.py (same PLAY/LEAN gate,
same signal thresholds) -- this does not invent a new grading rule.

Inputs (already-computed, real, canonical artifacts -- nothing re-run):
- rich per-row projection detail from the original full-stack benchmark build
  (mc_proj/ml_proj/state_proj/ensemble_proj/position/etc, one row per
  player-week-market before the slim columns were trimmed for the committed
  non_qb_detail.csv)
- the real historical market props archive used for that same benchmark
- wr_r15_confirmation_predictions.csv -- the canonical WR-R15 OOS validation
  artifact (run 34238301577, artifact 10061328722, the exact run/artifact IDs
  WR-R15's own production certification is pinned to). Covers 2023-2024 only
  -- WR-R15 was never validated for 2025 (scientific_confirmation_seasons:
  [2023, 2024] is the frozen production contract), so 2025 WR receiving rows
  are correctly left on the base ensemble, not silently "confirmed."
- te_r5_oos_player_casebook.csv -- the canonical TE-R5 OOS validation artifact
  (run 34132127351, artifact 10022512461). Covers 2023-2025, full overlap.

Output: a revised detail/summary pair plus a rerun of the situational
edge-hunt slice specifically on receptions, to see whether the candidate
UNDER edge found in situational_edge_hunt_v1.py holds, sharpens, or breaks
once these models are actually applied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade

DETAIL = Path("/tmp/nonqb_final_grade/full_stack_vegas_benchmark_detail.csv")
PROPS = Path("/tmp/nonqb_archive_out/historical_market_props.csv")
WR_R15_PRED = Path("/tmp/wr_r15_artifact/backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv")
TE_R5_CASEBOOK = Path("/tmp/te_r5_artifact/data/backtests/te_r5_participation_entitlement_v1/te_r5_oos_player_casebook.csv")

OUT_DIR = Path("docs/research/overnight")
WR_R15_CONFIRMED_SEASONS = {2023, 2024}


def main() -> int:
    det = pd.read_csv(DETAIL, low_memory=False)
    props = pd.read_csv(PROPS, low_memory=False)

    # Reduce back to the columns grade() actually needs, starting from the
    # ORIGINAL ensemble_proj so untouched rows are byte-identical to the
    # committed benchmark.
    proj = det[[
        "player_x", "player_clean_key", "team", "opponent", "season", "week",
        "position", "event_id", "market", "mc_proj", "ml_proj", "state_proj",
        "ensemble_proj", "actual", "game_id",
    ]].copy()
    proj = proj.rename(columns={"player_x": "player"})
    proj["adjusted_proj"] = proj["ensemble_proj"]
    proj["adjustment_applied"] = "NONE"

    # --- WR-R15: 2023-2024 only, WR position, receptions/rec_yards ---
    wr = pd.read_csv(WR_R15_PRED, low_memory=False)
    wr = wr.loc[wr["variant"].eq("WR_R15_WR1_ANCHORED_PARTICIPATION")].copy()
    wr_rec = wr[["team", "player_clean_key", "season", "week", "mc_receptions"]].rename(columns={"mc_receptions": "wr_r15_value"})
    wr_rec["market"] = "receptions"
    wr_yds = wr[["team", "player_clean_key", "season", "week", "mc_rec_yards"]].rename(columns={"mc_rec_yards": "wr_r15_value"})
    wr_yds["market"] = "rec_yards"
    wr_long = pd.concat([wr_rec, wr_yds], ignore_index=True)
    wr_long = wr_long.loc[wr_long["season"].isin(WR_R15_CONFIRMED_SEASONS)]

    proj = proj.merge(wr_long, on=["team", "player_clean_key", "season", "week", "market"], how="left")
    wr_mask = proj["position"].eq("WR") & proj["wr_r15_value"].notna()
    proj.loc[wr_mask, "adjusted_proj"] = proj.loc[wr_mask, "wr_r15_value"]
    proj.loc[wr_mask, "adjustment_applied"] = "WR_R15_WR1_ANCHORED_PARTICIPATION"
    proj = proj.drop(columns=["wr_r15_value"])

    # --- TE-R5P: 2023-2025, TE position, receptions/rec_yards ---
    te = pd.read_csv(TE_R5_CASEBOOK, low_memory=False)
    te_rec = te[["team", "player_key", "season", "week", "candidate_receptions_r5"]].rename(
        columns={"player_key": "player_clean_key", "candidate_receptions_r5": "te_r5_value"}
    )
    te_rec["market"] = "receptions"
    te_yds = te[["team", "player_key", "season", "week", "candidate_rec_yards_r5"]].rename(
        columns={"player_key": "player_clean_key", "candidate_rec_yards_r5": "te_r5_value"}
    )
    te_yds["market"] = "rec_yards"
    te_long = pd.concat([te_rec, te_yds], ignore_index=True)

    proj = proj.merge(te_long, on=["team", "player_clean_key", "season", "week", "market"], how="left")
    te_mask = proj["position"].eq("TE") & proj["te_r5_value"].notna()
    proj.loc[te_mask, "adjusted_proj"] = proj.loc[te_mask, "te_r5_value"]
    proj.loc[te_mask, "adjustment_applied"] = "TE_R5P_PARTICIPATION_ENTITLEMENT"
    proj = proj.drop(columns=["te_r5_value"])

    print(f"WR-R15 applied to {int(wr_mask.sum())} rows (2023-2024 WR receptions/rec_yards)")
    print(f"TE-R5P applied to {int(te_mask.sum())} rows (2023-2025 TE receptions/rec_yards)")

    # Re-grade with the adjusted projection, using the exact same grading
    # arithmetic as the committed benchmark.
    detail_new, summary_new = grade(proj, props, proj_col="adjusted_proj")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_new.to_csv(OUT_DIR / "non_qb_detail_wr_r15_te_r5p_applied.csv", index=False)
    summary_new.to_csv(OUT_DIR / "non_qb_summary_wr_r15_te_r5p_applied.csv", index=False)

    print("\n=== NEW SUMMARY (receptions/rec_yards only, STRONG_ONLY_PLAY_TIER) ===")
    s = summary_new.loc[summary_new.market.isin(["receptions", "rec_yards"]) & summary_new.tier.eq("STRONG_ONLY_PLAY_TIER")]
    print(s.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
