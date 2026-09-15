#!/usr/bin/env python3
"""Independent verification of GPT-5.6's Phase-4A "canonical modeled WR room"
claim (Issue #535 comment 5673487481): that among the WR-R15 authority
artifact's 1,088 candidate team-games, only 71 have a candidate-prediction
identity set that exactly equals {WR1 anchor (if present)} union {every
WR2+ identity in wr_r15_confirmation_features.csv for that team-game}.

Source-only structural check. No receiving-yard outcome interpretation, no
production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"

# Candidate identity columns, tried in order, first one present in BOTH
# frames wins. Printed either way so a mismatch is visible, not silent.
IDENTITY_COL_CANDIDATES = [
    "player_clean_key",
    "player_id",
    "gsis_id",
    "nflverse_player_id",
    "receiver_player_id",
    "player_name",
]


def _pick_identity_col(pred: pd.DataFrame, feat: pd.DataFrame) -> str | None:
    for c in IDENTITY_COL_CANDIDATES:
        if c in pred.columns and c in feat.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions, low_memory=False)
    pred.columns = [str(c).strip().lower() for c in pred.columns]
    feat = pd.read_csv(args.features, low_memory=False)
    feat.columns = [str(c).strip().lower() for c in feat.columns]

    print("=== columns ===")
    print("predictions columns:", list(pred.columns))
    print("features columns:", list(feat.columns))

    id_col = _pick_identity_col(pred, feat)
    print(f"\nidentity column selected: {id_col!r}")
    if id_col is None:
        print("FAIL: no shared identity column found between predictions and features -- "
              "cannot reproduce the 71-team-game claim mechanically. Reporting columns only.")
        return 1

    cand = pred.loc[pred["variant"].eq(CANDIDATE_VARIANT)].copy()
    team_games = cand[["season", "week", "team"]].drop_duplicates()
    print(f"\ncandidate team-games: {len(team_games)}")

    anchor_present = cand.loc[cand["wr_rank"].eq(1), ["season", "week", "team"]].drop_duplicates()
    n_anchor_present = len(anchor_present)
    print(f"team-games with wr_rank==1 present: {n_anchor_present}")

    # WR2+ canonical membership per team-game, from the features file.
    if "baseline_wr_rank" in feat.columns:
        wr2plus_feat = feat.loc[feat["baseline_wr_rank"].ge(2)].copy()
    else:
        wr2plus_feat = feat.copy()
    print(f"WR2+ feature rows (canonical WR2+ room source): {len(wr2plus_feat)}")

    anchor_ids = (
        cand.loc[cand["wr_rank"].eq(1), ["season", "week", "team", id_col]]
        .drop_duplicates()
    )
    pred_ids = cand[["season", "week", "team", id_col]].drop_duplicates()
    feat_ids = wr2plus_feat[["season", "week", "team", id_col]].drop_duplicates()

    canonical = (
        pd.concat([anchor_ids, feat_ids], ignore_index=True)
        .drop_duplicates(["season", "week", "team", id_col])
    )

    complete = 0
    incomplete_examples = []
    for key, cgrp in canonical.groupby(["season", "week", "team"]):
        canonical_set = set(cgrp[id_col])
        pgrp = pred_ids.loc[
            pred_ids["season"].eq(key[0]) & pred_ids["week"].eq(key[1]) & pred_ids["team"].eq(key[2])
        ]
        pred_set = set(pgrp[id_col])
        if pred_set == canonical_set:
            complete += 1
        elif len(incomplete_examples) < 5:
            incomplete_examples.append((key, len(canonical_set), len(pred_set), len(canonical_set - pred_set)))

    print(f"\ntotal team-games evaluated (union of prediction + canonical keys): {canonical['season'].count() and team_games.shape[0]}")
    print(f"team-games where candidate prediction identity set EXACTLY equals canonical room set: {complete}")
    print(f"team-games NOT exactly matching: {team_games.shape[0] - complete}")
    print("\nfirst few mismatches (key, n_canonical, n_predicted, n_missing_from_predictions):")
    for ex in incomplete_examples:
        print(" ", ex)

    print(f"\nGPT's claim: 71 / 1088 team-games truly complete.")
    print(f"Independently computed: {complete} / {team_games.shape[0]} team-games truly complete.")
    if complete == 71 and team_games.shape[0] == 1088:
        print("MATCH: independent computation confirms the 71/1088 claim exactly.")
    else:
        print("MISMATCH: independent computation does NOT match the claimed 71/1088 -- needs reconciliation before Phase 4A freeze.")

    # --- Diagnostic 2: raw row-count parity (NOT deduplicated), to test
    # whether GPT's "71" came from comparing raw feature ROW counts (which
    # can exceed distinct-identity counts if a player has >1 feature row
    # per team-game, e.g. one per market/event_id) rather than distinct
    # identity SETS.
    raw_feat_counts = wr2plus_feat.groupby(["season", "week", "team"]).size().rename("n_feat_rows_raw")
    raw_pred_counts = cand.groupby(["season", "week", "team"]).size().rename("n_pred_rows_raw")
    anchor_flag = anchor_present.assign(has_anchor=1).set_index(["season", "week", "team"])["has_anchor"]
    rowcount = pd.concat([raw_feat_counts, raw_pred_counts, anchor_flag], axis=1).fillna(0)
    rowcount["expected_full_room_rows"] = rowcount["n_feat_rows_raw"] + rowcount["has_anchor"]
    rowcount_complete = int((rowcount["n_pred_rows_raw"] == rowcount["expected_full_room_rows"]).sum())
    print(f"\n[diagnostic] raw-row-count parity (undeduplicated feature rows + anchor flag vs candidate row count): "
          f"{rowcount_complete} / {len(rowcount)} team-games match by RAW ROW COUNT (not identity set).")

    # Full sorted list of exactly-complete team-game keys (identity-set
    # definition) so this can be diffed directly against GPT's own list.
    # Also flag anchor presence per key to test GPT's explanation (comment
    # 5673565674) that the 79-71 gap is entirely anchor-absent team-games
    # where the (empty) prediction set trivially equals the (anchor-less)
    # canonical set.
    anchor_keys = set(zip(anchor_present["season"], anchor_present["week"], anchor_present["team"]))
    complete_keys = []
    for key, cgrp in canonical.groupby(["season", "week", "team"]):
        canonical_set = set(cgrp[id_col])
        pgrp = pred_ids.loc[
            pred_ids["season"].eq(key[0]) & pred_ids["week"].eq(key[1]) & pred_ids["team"].eq(key[2])
        ]
        if set(pgrp[id_col]) == canonical_set:
            complete_keys.append(key)
    complete_keys.sort()
    print(f"\nfull sorted list of {len(complete_keys)} identity-set-complete team-game keys (season, week, team), anchor flag:")
    for k in complete_keys:
        print(" ", k, "anchor_present" if k in anchor_keys else "NO_ANCHOR")

    complete_with_anchor = [k for k in complete_keys if k in anchor_keys]
    complete_without_anchor = [k for k in complete_keys if k not in anchor_keys]
    print(f"\ncomplete AND anchor-present (GPT's stricter 'full WR room' definition): {len(complete_with_anchor)}")
    print(f"complete but anchor-ABSENT (the disputed subset): {len(complete_without_anchor)}")
    print(f"\nGPT's claim: the 79-71={79-71} gap is entirely the anchor-absent set.")
    if len(complete_with_anchor) == 71 and len(complete_without_anchor) == 8:
        print("CONFIRMED: anchor-present-and-complete count is exactly 71, anchor-absent-and-complete count is exactly 8. GPT's explanation is correct.")
    else:
        print(f"NOT CONFIRMED: anchor-present-and-complete = {len(complete_with_anchor)}, anchor-absent-and-complete = {len(complete_without_anchor)} -- does not exactly match GPT's explanation, needs further digging.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
