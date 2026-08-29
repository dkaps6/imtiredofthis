#!/usr/bin/env python3
"""Freeze the corrected M69 artifact into immutable qb_frontier_canonical_v1.

This is a one-time data migration. It MUST NOT rebuild M59-M65. Instead it
consumes the exact corrected authoritative M69 artifact and writes a compact
canonical CSV plus provenance manifest for M70+ research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SNAPSHOT_ID = "qb_frontier_canonical_v1"
SOURCE_RUN_ID = 33262455637
SOURCE_ARTIFACT_ID = 9717868133
SOURCE_ARTIFACT_DIGEST = "sha256:fbe7500d04e971c0d83ac468cecdf022407c3179e3904334521e380afa4070f4"
SOURCE_M69_HEAD = "6de9030c60d481f9783a8b08926a3273afa39fd8"
SOURCE_M69_MERGE = "a081b24d60d6cc78954ed8ffd25fab57d1f98bc7"
EXPECTED_SNAPSHOT_SHA256 = "396d924a308fb05baff9a44d7485fe87d3bf008a9134b65a66c06a7704a8f87c"

COLUMNS = [
    "season","week","game_id","team","opponent","player_clean_key",
    "market_spread","market_abs_spread","market_total","market_team_implied","market_opp_implied","market_moneyline","is_home",
    "actual_pass_yards","pred_pass_yards","actual_attempts","pred_attempts","actual_ypa","pred_ypa",
    "attempts_raw","ypa_contextual","m64_pass_raw_reference","m64_pass_raw_point_product","pred_point_product",
    "pass_residual_actual_minus_pred","attempt_residual_actual_minus_pred","ypa_residual_actual_minus_pred","abs_pass_error","cat75","cat100",
    "volume_yard_contribution","efficiency_yard_contribution","interaction_yard_contribution","mc_point_product_remainder",
    "mechanism","recoverability",
    "m64_pred_drives","m64_pred_plays_per_drive","m64_pred_dropback_rate_neutral","m64_pred_dropback_rate_gamescript","m64_pred_attempt_conversion","m64_pred_qb_attempt_share",
    "m64_attempts_generative_neutral","m64_attempts_generative_gamescript","m64_pass_generative_neutral","m64_pass_generative_gamescript",
    "m64_actual_team_drives","m64_actual_plays_per_drive","m64_actual_dropback_rate","m64_actual_attempt_conversion","m64_actual_team_pass_attempts","m64_actual_qb_attempt_share",
    "m65_actual_neutral_share","m65_actual_trailing_share","m65_actual_leading_share",
    "m65_actual_neutral_dropback_rate","m65_actual_trailing_dropback_rate","m65_actual_leading_dropback_rate",
    "m65_pred_neutral_share","m65_pred_trailing_share","m65_pred_leading_share",
    "m65_pred_neutral_dropback_rate","m65_pred_trailing_dropback_rate","m65_pred_leading_dropback_rate","m65_pred_dropback_rate",
    "m65_attempts_state_ridge","m65_pass_state_ridge",
    "opening_first15_dbr_mean8","opening_q1_dbr_mean8","playcaller_opening_first15_dbr_mean8","playcaller_opening_q1_dbr_mean8",
    "playcaller_changed_since_last_game","playcaller_prior_games_allteams","playcaller_prior_games_team","playcaller_new_to_team",
    "opp_coverage_man_rate","opp_coverage_zone_rate","opp_pressure_rate_generated","opp_def_pass_epa","opp_success_rate_def","opponent_force_pass","opp_explosive_play_rate_allowed",
    "actual_drives","actual_scrimmage_plays","actual_plays_per_drive","actual_dropbacks","actual_pbp_pass_attempts","actual_dropback_rate","actual_attempt_conversion",
    "actual_first10_dbr","actual_first15_dbr","actual_first_drive_dbr","actual_first2drives_dbr","actual_q1_dbr","actual_first_half_dbr","actual_first15_vs_rest_dbr",
    "actual_neutral_share","actual_trailing8_share","actual_leading8_share","actual_neutral_dbr","actual_trailing8_dbr","actual_leading8_dbr",
    "actual_rush_epa","actual_pass_epa","actual_rush_success","actual_pass_success","actual_sack_rate","actual_scramble_rate","actual_turnovers",
    "realized_def_man_rate","realized_def_zone_rate","realized_def_box_mean","realized_def_heavy_box_rate","realized_def_light_box_rate","realized_def_pass_rushers_mean","realized_def_pressure_rate",
    "realized_cover0_rate","realized_cover1_rate","realized_cover2_rate","realized_cover3_rate","realized_cover4_rate","realized_cover6_rate",
    "trailing_share_surprise","leading_share_surprise","opening_baseline_playcaller","opening_deviation_vs_playcaller","opening_regime",
]

SOURCE_FILES = [
    "m69_summary/m69_game_forensic_atlas.csv",
    "m69_summary/m69_precommitted_interpretation.csv",
    "m65_summary/m65_game_level.csv",
    "m64_summary/m64_game_level.csv",
    "m68_source/m68_pregame_new_information_features.csv",
    "2024/m62/m62_enriched_games_2024.csv",
    "2025/m62/m62_enriched_games_2025.csv",
    "2024/qb_both_raw/qb_both_raw_walkforward_trace.csv",
    "2025/qb_both_raw/qb_both_raw_walkforward_trace.csv",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    atlas_path = a.artifact_root / "m69_summary/m69_game_forensic_atlas.csv"
    if not atlas_path.exists():
        raise RuntimeError(f"missing authoritative M69 atlas: {atlas_path}")
    x = pd.read_csv(atlas_path, low_memory=False)
    missing = [c for c in COLUMNS if c not in x.columns]
    if missing:
        raise RuntimeError(f"canonical v1 missing required columns: {missing}")

    snap = x[COLUMNS].copy().sort_values(["season","week","team","player_clean_key"]).reset_index(drop=True)
    if len(snap) != 643 or snap.duplicated(["season","week","team","player_clean_key"]).any():
        raise RuntimeError("canonical v1 row/key invariant failed")
    if snap.season.value_counts().to_dict() != {2024: 332, 2025: 311}:
        raise RuntimeError(f"canonical v1 season invariant failed: {snap.season.value_counts().to_dict()}")
    if int(snap.cat75.sum()) != 188 or int(snap.cat100.sum()) != 96:
        raise RuntimeError("canonical v1 tail-count invariant failed")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = a.out_dir / f"{SNAPSHOT_ID}.csv"
    snap.to_csv(csv_path, index=False, float_format="%.10g")
    snapshot_sha = sha256(csv_path)
    if snapshot_sha != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError(f"canonical v1 snapshot hash drift: {snapshot_sha}")

    source_meta = {}
    for rel in SOURCE_FILES:
        p = a.artifact_root / rel
        if not p.exists():
            raise RuntimeError(f"canonical v1 source missing: {rel}")
        source_meta[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

    manifest = {
        "snapshot_id": SNAPSHOT_ID,
        "schema_version": 1,
        "row_count": int(len(snap)),
        "seasons": [2024, 2025],
        "columns": len(COLUMNS),
        "snapshot_file": csv_path.name,
        "snapshot_sha256": snapshot_sha,
        "source_m69_workflow_run_id": SOURCE_RUN_ID,
        "source_m69_artifact_id": SOURCE_ARTIFACT_ID,
        "source_m69_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_m69_head_commit": SOURCE_M69_HEAD,
        "source_m69_merge_commit": SOURCE_M69_MERGE,
        "purpose": "Immutable canonical 2024-2025 QB research foundation. M70+ must load this snapshot instead of rebuilding M59-M65 unless a documented upstream foundation bug creates v2.",
        "invariants": {
            "target_games": 643,
            "season_2024_rows": 332,
            "season_2025_rows": 311,
            "cat75_games": 188,
            "cat100_games": 96,
        },
        "source_files": source_meta,
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{SNAPSHOT_ID}] rows={len(snap)} cols={len(COLUMNS)} sha256={snapshot_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
