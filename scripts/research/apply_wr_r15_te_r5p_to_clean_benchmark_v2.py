#!/usr/bin/env python3
"""Apply the promoted WR-R15/TE-R5P entitlement models to the IDENTITY-CLEAN
full-stack Vegas benchmark cohort (PR #541/#542), the same method as
apply_wr_r15_te_r5p_to_vegas_benchmark_v1.py used on the corrupted cohort,
now re-run on data known to be correctly matched to real games.

This directly answers the "you only tested the base ensemble, not my
layered research" gap in Claude's clean-cohort re-grade
(CLEAN_BENCHMARK_INDEPENDENT_REGRADE_V1_RESULT.md): that re-grade covered
only mc_proj/ml_proj/state_proj + frozen ensemble weights, not WR-R15/TE-R5P.

Research only. Reuses the exact grading arithmetic from
scripts/backtest/grade_full_stack_vegas_benchmark_v1.py -- no new grading
rule invented.

Inputs:
- docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv --
  the identity-clean graded detail from PR #541/#542 (retains
  mc_proj/ml_proj/state_proj/ensemble_proj/game_id from before grading)
- docs/research/overnight/clean_v1_historical_market_props.csv -- the same
  clean historical props archive used to build that detail
- wr_r15_confirmation_predictions.csv -- canonical WR-R15 OOS validation
  artifact (run 34238301577, artifact 10061328722). 2023-2024 only by its
  own frozen production contract.
- te_r5_oos_player_casebook.csv -- canonical TE-R5 OOS validation artifact
  (run 34132127351, artifact 10022512461). 2023-2025, full overlap.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade

DETAIL = Path("docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv")
PROPS = Path("docs/research/overnight/clean_v1_historical_market_props.csv")
WR_R15_PRED = Path("/tmp/wr_r15_artifact/backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv")
TE_R5_CASEBOOK = Path("/tmp/te_r5_artifact/data/backtests/te_r5_participation_entitlement_v1/te_r5_oos_player_casebook.csv")

OUT_DIR = Path("docs/research/overnight")
WR_R15_CONFIRMED_SEASONS = {2023, 2024}


def main() -> int:
    det = pd.read_csv(DETAIL, low_memory=False)
    props = pd.read_csv(PROPS, low_memory=False)
    det.columns = [c.strip().lower() for c in det.columns]

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

    detail_new, summary_new = grade(proj, props, proj_col="adjusted_proj")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_new.to_csv(OUT_DIR / "clean_v2_non_qb_detail_wr_r15_te_r5p_applied.csv", index=False)
    summary_new.to_csv(OUT_DIR / "clean_v2_non_qb_summary_wr_r15_te_r5p_applied.csv", index=False)

    print("\n=== NEW SUMMARY (receptions/rec_yards only, STRONG_ONLY_PLAY_TIER) ===")
    s = summary_new.loc[summary_new.market.isin(["receptions", "rec_yards"]) & summary_new.tier.eq("STRONG_ONLY_PLAY_TIER")]
    print(s.to_string(index=False))

    print("\n=== FULL SUMMARY, ALL MARKETS, ALL TIERS ===")
    print(summary_new.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
