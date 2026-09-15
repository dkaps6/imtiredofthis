#!/usr/bin/env python3
"""One-off independent verification of GPT-5.6's structural claims about the
exact WR-R15 authority artifact (run 34238301577 / artifact 10061328722):
row counts, WR1-anchor coverage, and whether pred_targets sums are conserved
across the graded confirmation-predictions subset. Source-only, no WR
receiving-yard outcome interpretation, no production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=False)
    ap.add_argument("--conservation", type=Path, required=False)
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions, low_memory=False)
    pred.columns = [str(c).strip().lower() for c in pred.columns]
    print("=== wr_r15_confirmation_predictions.csv ===")
    print(f"total rows: {len(pred)}")
    print("rows per variant:")
    print(pred["variant"].value_counts())
    cand = pred.loc[pred["variant"].eq("WR_R15_WR1_ANCHORED_PARTICIPATION")].copy()
    print(f"\ncandidate variant rows: {len(cand)}")
    print("candidate rows by season:")
    print(cand["season"].value_counts().sort_index())
    print(f"\ncandidate wr_rank==1 rows: {int(cand['wr_rank'].eq(1).sum())}")
    team_games = cand[["season", "week", "team"]].drop_duplicates()
    print(f"distinct (season,week,team) team-games in candidate rows: {len(team_games)}")
    anchor_team_games = cand.loc[cand["wr_rank"].eq(1), ["season", "week", "team"]].drop_duplicates()
    print(f"distinct team-games WITH a wr_rank==1 row present: {len(anchor_team_games)}")
    print(f"team-games MISSING a wr_rank==1 row: {len(team_games) - len(anchor_team_games)}")

    print("\npred_targets sum by team-game, candidate vs baseline (first 5 largest diffs):")
    base = pred.loc[pred["variant"].eq("M38_EXPLICIT_BASELINE")].copy() if pred["variant"].isin(["M38_EXPLICIT_BASELINE"]).any() else pd.DataFrame()
    if len(base):
        cand_sum = cand.groupby(["season", "week", "team"])["pred_targets"].sum().rename("cand_sum")
        base_sum = base.groupby(["season", "week", "team"])["pred_targets"].sum().rename("base_sum")
        both = pd.concat([cand_sum, base_sum], axis=1).dropna()
        both["abs_diff"] = (both["cand_sum"] - both["base_sum"]).abs()
        print(both.sort_values("abs_diff", ascending=False).head(5))
        print(f"max abs diff: {both['abs_diff'].max()}")
    else:
        print("(no M38_EXPLICIT_BASELINE variant rows found in this file)")

    if args.features and args.features.exists():
        feat = pd.read_csv(args.features, low_memory=False)
        feat.columns = [str(c).strip().lower() for c in feat.columns]
        print(f"\n=== wr_r15_confirmation_features.csv ===")
        print(f"total rows: {len(feat)}")
        if "baseline_wr_rank" in feat.columns:
            print("baseline_wr_rank value counts (min value shown):")
            print(f"min baseline_wr_rank: {feat['baseline_wr_rank'].min()}")

    if args.conservation and args.conservation.exists():
        cons = pd.read_csv(args.conservation, low_memory=False)
        cons.columns = [str(c).strip().lower() for c in cons.columns]
        print(f"\n=== wr_r15_conservation_audit.csv ===")
        print(f"total rows (team-games): {len(cons)}")
        for c in ["max_anchor_delta", "max_secondary_gap", "max_wr_gap", "max_team_gap"]:
            if c in cons.columns:
                print(f"{c} max: {cons[c].abs().max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
