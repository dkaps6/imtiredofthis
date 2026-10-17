#!/usr/bin/env python3
"""PRISM Stage-2 market translation independent-history source screen V1.

Source/lineage only. No target outcomes are scored. The first hard gate is the
exact frozen official-inactive authority required by qualified Stage 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

VERSION = "PRISM_STAGE2_MARKET_TRANSLATION_INDEPENDENT_HISTORY_SOURCE_SCREEN_V1"
INACTIVE_SHA = "d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
TARGET_SEASON = 2023
EXPECTED_PARENT_SEASONS = {2024: 544, 2025: 544}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.inactives.exists() or args.inactives.stat().st_size <= 0:
        raise RuntimeError(f"missing frozen inactive authority: {args.inactives}")

    digest = sha256(args.inactives)
    if digest != INACTIVE_SHA:
        raise RuntimeError(f"frozen inactive SHA drift: {digest}")

    x = pd.read_csv(args.inactives, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    need = {"season", "week", "team"}
    if need - set(x.columns):
        raise RuntimeError(f"inactive authority missing {sorted(need-set(x.columns))}")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate inactive team-week identity")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    season_counts = {
        int(s): int(n)
        for s, n in x.groupby("season", dropna=True).size().items()
        if pd.notna(s)
    }
    parent_exact = all(season_counts.get(s, 0) == n for s, n in EXPECTED_PARENT_SEASONS.items())
    target_rows = int(season_counts.get(TARGET_SEASON, 0))

    gates = {
        "frozen_inactive_sha_exact": digest == INACTIVE_SHA,
        "parent_2024_2025_shape_exact": parent_exact and len(x) == 1088,
        "target_2023_official_inactive_authority_present": target_rows > 0,
        "target_2023_complete_regular_teamweek_authority": target_rows >= 544,
    }

    ready = all(gates.values())
    disposition = (
        "PRISM_STAGE2_MARKET_TRANSLATION_2023_SOURCE_READY_FOR_DEEP_AUDIT"
        if ready
        else "PRISM_STAGE2_MARKET_TRANSLATION_2023_SOURCE_BLOCKED"
    )
    terminal_reason = (
        ""
        if ready
        else (
            "QUALIFIED_STAGE1_REQUIRES_EXACT_OFFICIAL_INACTIVE_PREGAME_EXCLUSION__"
            "FROZEN_AUTHORITY_HAS_NO_COMPLETE_2023_COVERAGE__DO_NOT_APPROXIMATE"
        )
    )

    result = {
        "version": VERSION,
        "target_season": TARGET_SEASON,
        "status": "SOURCE_SCREEN_COMPLETE",
        "disposition": disposition,
        "inactive_authority_path": str(args.inactives),
        "inactive_authority_sha256": digest,
        "inactive_authority_rows": int(len(x)),
        "inactive_authority_season_counts": season_counts,
        "target_2023_rows": target_rows,
        "gates": gates,
        "terminal_reason": terminal_reason,
        "stage2_cells_requested": ["RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING"],
        "stage1_true_pregame_universe_required": True,
        "target_game_outcomes_scored": 0,
        "target_game_snaps_used_to_define_active_state": 0,
        "new_model_fits": 0,
        "threshold_searches": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "week6_outcomes_used": 0,
        "week6_locks_changed": False,
        "production_changed": False,
        "next_if_blocked": (
            "PREPARE_PROSPECTIVE_STAGE2_MARKET_TRANSLATION_MECHANICS__"
            "NO_WEEK6_LOCK_MUTATION"
        ),
    }

    (args.out_dir / "stage2_market_translation_independent_history_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(
        [{"season": s, "teamweek_rows": n} for s, n in sorted(season_counts.items())]
    ).to_csv(args.out_dir / "inactive_authority_season_coverage.csv", index=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
