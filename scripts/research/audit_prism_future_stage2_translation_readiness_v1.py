#!/usr/bin/env python3
"""Readiness-only probe for the next prospective PRISM Stage-2 market translation.

This probe can NEVER create a scientific lock. It reuses the frozen future-week
player-state source audit to determine whether a future target has enough
strictly-prior source state to *be eligible for lock construction later*.

No target outcomes, sportsbook inputs, OddsAPI, fits, threshold search, or
production writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.research.audit_player_state_future_week_coverage_v1 import audit

VERSION = "PRISM_FUTURE_STAGE2_TRANSLATION_READINESS_V1"


def adjudicate(season: int, week: int, out_dir: Path) -> tuple[dict, object, object, object]:
    if season == 2026 and week <= 6:
        raise RuntimeError("readiness probe refuses immutable Week-6-or-earlier 2026 targets")

    try:
        rows, pos, rooms, parent = audit(season, week)
        parent_ready = parent.get("disposition") == "PLAYER_STATE_FUTURE_WEEK_READY"
        exact_roster = bool(parent.get("target_week_roster_source_ready", False))
        prior_complete = bool(parent.get("prior_week_source_completion", {}).get("complete", False))
        chrono = int(parent.get("chronology_violations", -1))
        source_ready = bool(parent_ready and exact_roster and prior_complete and chrono == 0)
        blocker = "" if source_ready else "PLAYER_STATE_SOURCE_GATES_NOT_ALL_READY"
    except Exception as exc:
        rows = pos = rooms = None
        parent = None
        source_ready = False
        blocker = f"{type(exc).__name__}: {exc}"

    result = {
        "version": VERSION,
        "status": (
            "PRISM_FUTURE_STAGE2_TRANSLATION_SOURCE_READY__NO_SCIENTIFIC_LOCK_CREATED"
            if source_ready
            else "PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED"
        ),
        "season": int(season),
        "target_week": int(week),
        "source_ready": bool(source_ready),
        "source_blocker": blocker,
        "player_state_result": parent,
        "scientific_lock_created": False,
        "stage2_lock_rows_created": 0,
        "market_translation_rows_created": 0,
        "target_outcomes_read": 0,
        "week6_outcomes_read_by_probe": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "threshold_searches": 0,
        "production_changed": False,
        "boundary": (
            "readiness only; even SOURCE_READY does not authorize a lock unless the "
            "separately frozen chronology contract is satisfied before target outcomes"
        ),
    }
    return result, rows, pos, rooms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--target-week", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result, rows, pos, rooms = adjudicate(args.season, args.target_week, args.out_dir)
    if rows is not None:
        rows.to_csv(args.out_dir / "player_state_readiness_rows.csv", index=False)
    if pos is not None:
        pos.to_csv(args.out_dir / "player_state_position_coverage.csv", index=False)
    if rooms is not None:
        rooms.to_csv(args.out_dir / "player_state_rb_rooms.csv", index=False)
    (args.out_dir / "prism_future_stage2_translation_readiness_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
