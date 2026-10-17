#!/usr/bin/env python3
"""PRISM Stage-2 2023 independent translation source audit V1.

Source/lineage only. No target-game outcome is scored and no predictive model is
fit. A common Stage-1 pregame-universe blocker fails every Stage-2 translation
cell closed even when a downstream entitlement source exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_40s_information_frontier as m76
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.research.audit_player_active_role_participation_v1 import normalize_roster

VERSION = "PRISM_STAGE2_2023_TRANSLATION_SOURCE_AUDIT_V1"
TARGET_SEASON = 2023
TARGET_WEEKS = tuple(range(5, 19))
EXPECTED_M78_SHA = "d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
CELLS = ("RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")


def num(x):
    return pd.to_numeric(x, errors="coerce")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--te-coefficients", type=Path, required=True)
    ap.add_argument("--te-folds", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_meta: list[dict] = []

    # Exact frozen inactive authority.
    inactive_sha = file_sha(args.inactives)
    if inactive_sha != EXPECTED_M78_SHA:
        raise RuntimeError(
            f"frozen M78 inactive SHA drift {inactive_sha} != {EXPECTED_M78_SHA}"
        )
    ina = read_csv(args.inactives, "frozen M78 inactive authority")
    need = {"season", "week", "team", "inactive_tokens", "inactive_count"}
    if need - set(ina.columns):
        raise RuntimeError(f"inactive authority missing {sorted(need-set(ina.columns))}")
    ina["season"] = num(ina["season"]).astype(int)
    inactive_seasons = sorted(ina["season"].unique().tolist())
    inactive_2023 = ina.loc[ina["season"].eq(TARGET_SEASON)].copy()

    # Actual 2023 roster source probe. This is identity/source characterization
    # only; target-game outcome columns are never loaded.
    roster_error = ""
    try:
        raw_roster = m78.load_weekly_rosters(TARGET_SEASON, source_meta)
        roster = normalize_roster(raw_roster, TARGET_SEASON)
        roster_target = roster.loc[roster["week"].isin(TARGET_WEEKS)].copy()
        roster_weeks = sorted(roster_target["week"].unique().tolist())
        roster_stable_id = (
            float(roster_target["player_id"].astype(str).str.strip().ne("").mean())
            if len(roster_target) else 0.0
        )
    except Exception as exc:
        roster_target = pd.DataFrame()
        roster_weeks = []
        roster_stable_id = 0.0
        roster_error = f"{type(exc).__name__}:{exc}"

    # Actual historical offensive-snap source probe. No target value is scored.
    snap_error = ""
    try:
        as_of = datetime.now(timezone.utc).isoformat()
        raw_snaps = m76.download_table(
            "pfr_snap_counts",
            TARGET_SEASON,
            m76.release_urls("snaps", TARGET_SEASON),
            source_meta,
            as_of,
        )
        if raw_snaps.empty:
            snap_rows = 0
            snap_weeks = []
        else:
            snap = raw_snaps.copy()
            snap.columns = [str(col).strip().lower() for col in snap.columns]
            snap_rows = int(len(snap))
            snap_weeks = (
                sorted(num(snap["week"]).dropna().astype(int).unique().tolist())
                if "week" in snap.columns else []
            )
    except Exception as exc:
        snap_rows = 0
        snap_weeks = []
        snap_error = f"{type(exc).__name__}:{exc}"

    # TE-R5P OOS source lineage.
    coeff = read_csv(args.te_coefficients, "TE-R5P coefficients")
    folds = read_csv(args.te_folds, "TE-R5P folds")
    if "test_season" not in coeff.columns or "test_season" not in folds.columns:
        raise RuntimeError("TE-R5P source missing test_season")
    coeff_2023 = coeff.loc[num(coeff["test_season"]).eq(TARGET_SEASON)].copy()
    fold_2023 = folds.loc[num(folds["test_season"]).eq(TARGET_SEASON)].copy()
    te_oos_2023 = bool(len(coeff_2023) and len(fold_2023))
    train_seasons = sorted(
        set(
            part.strip()
            for value in fold_2023.get("train_seasons", pd.Series(dtype=str)).astype(str)
            for part in value.split(",")
            if part.strip()
        )
    )
    te_training_precedes_target = bool(
        train_seasons
        and all(int(s) < TARGET_SEASON for s in train_seasons if s.isdigit())
    )

    stage1_gates = {
        "weekly_roster_source_nonempty": bool(len(roster_target)),
        "weekly_roster_target_weeks_5_18_complete": roster_weeks == list(TARGET_WEEKS),
        "stable_roster_identity_ge_099": roster_stable_id >= 0.99,
        "historical_snap_source_nonempty": snap_rows > 0,
        "historical_snap_weeks_cover_5_18": set(TARGET_WEEKS).issubset(set(snap_weeks)),
        "exact_official_inactive_2023_source_exists": len(inactive_2023) > 0,
        "exact_official_inactive_target_week_coverage_ge_099": False,
        "official_inactive_roster_identity_bridge_ge_095": False,
        "chronology_violations_zero": True,
    }

    # The missing official-inactive authority is a mandatory common Stage-1
    # pregame-universe gate. Do not continue to a scored/reconstructed proxy.
    common_stage1_ready = all(stage1_gates.values())

    entitlement_lineage = {
        "RB_RUSHING": {
            "status": "NOT_SCORED__COMMON_STAGE1_BLOCK",
            "canonical_parent_family": "EXISTING_RULE_RUSH_SHARE_TOP5_PLUS_RESIDUAL",
            "proxy_substituted": False,
        },
        "RB_RECEIVING": {
            "status": "NOT_SCORED__COMMON_STAGE1_BLOCK",
            "canonical_parent_family": "EXISTING_HISTORICAL_TARGET_ENTITLEMENT",
            "proxy_substituted": False,
        },
        "TE_RECEIVING": {
            "status": (
                "TE_R5P_2023_OOS_LINEAGE_PRESENT__COMMON_STAGE1_BLOCK"
                if te_oos_2023 and te_training_precedes_target
                else "TE_R5P_2023_OOS_LINEAGE_BLOCKED"
            ),
            "test_season_2023_fold_present": te_oos_2023,
            "train_seasons": train_seasons,
            "training_precedes_target": te_training_precedes_target,
            "proxy_substituted": False,
        },
    }

    per_cell = []
    for cell in CELLS:
        cell_source_ready = bool(
            common_stage1_ready
            and (
                cell != "TE_RECEIVING"
                or (te_oos_2023 and te_training_precedes_target)
            )
        )
        per_cell.append({
            "cell": cell,
            "stage1_common_source_ready": common_stage1_ready,
            "source_ready": cell_source_ready,
            "disposition": (
                f"{cell}_STAGE2_2023_TRANSLATION_SOURCE_READY"
                if cell_source_ready
                else f"{cell}_STAGE2_2023_TRANSLATION_SOURCE_BLOCKED"
            ),
            "blocked_reason": (
                ""
                if cell_source_ready
                else "EXACT_2023_OFFICIAL_GAME_DAY_INACTIVE_AUTHORITY_UNAVAILABLE_FOR_FROZEN_STAGE1_UNIVERSE"
            ),
        })

    ready = [x["cell"] for x in per_cell if x["source_ready"]]
    if len(ready) == len(CELLS):
        disposition = "PRISM_STAGE2_2023_TRANSLATION_SOURCE_READY_ALL"
    elif ready:
        disposition = "PRISM_STAGE2_2023_TRANSLATION_SOURCE_READY_PARTIAL"
    else:
        disposition = "PRISM_STAGE2_2023_TRANSLATION_SOURCE_BLOCKED"

    result = {
        "version": VERSION,
        "target_season": TARGET_SEASON,
        "target_weeks": [5, 18],
        "disposition": disposition,
        "ready_cells": ready,
        "stage1_common_source_ready": common_stage1_ready,
        "stage1_gates": stage1_gates,
        "frozen_inactive_sha256": inactive_sha,
        "frozen_inactive_seasons": inactive_seasons,
        "frozen_inactive_2023_teamweeks": int(len(inactive_2023)),
        "roster_probe": {
            "rows_weeks_5_18": int(len(roster_target)),
            "weeks": roster_weeks,
            "stable_identity_rate": roster_stable_id,
            "source_error": roster_error,
        },
        "snap_probe": {
            "raw_rows": snap_rows,
            "weeks": snap_weeks,
            "source_error": snap_error,
        },
        "entitlement_lineage": entitlement_lineage,
        "per_cell": per_cell,
        "hard_blocker": (
            "FROZEN_STAGE1_REQUIRES_EXACT_OFFICIAL_GAME_DAY_INACTIVE_IDENTITY__"
            "CANONICAL_M78_AUTHORITY_CONTAINS_2024_2025_ONLY__NO_2023_EQUIVALENT"
        ),
        "next_if_blocked": (
            "PREPARE_FUTURE_PROSPECTIVE_STAGE2_MARKET_TRANSLATION_TRACE__"
            "DO_NOT_TOUCH_IMMUTABLE_WEEK6_LOCKS"
        ),
        "integrity": {
            "target_game_outcomes_scored": 0,
            "target_game_outcomes_used_as_predictors": 0,
            "outcomes_2026_used": 0,
            "sportsbook_inputs": 0,
            "oddsapi_calls": 0,
            "models_fit": 0,
            "threshold_searches": 0,
            "projection_changes": 0,
            "production_changed": False,
            "historical_proxy_substituted": False,
        },
    }

    pd.DataFrame(source_meta).to_csv(
        args.out_dir / "prism_stage2_2023_source_probe_snapshots.csv", index=False
    )
    pd.DataFrame(per_cell).to_csv(
        args.out_dir / "prism_stage2_2023_source_cells.csv", index=False
    )
    (args.out_dir / "prism_stage2_2023_translation_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
