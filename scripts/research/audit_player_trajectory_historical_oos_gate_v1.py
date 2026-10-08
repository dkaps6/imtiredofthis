#!/usr/bin/env python3
"""Source-only Gate 0 for frozen WR/TE trajectory historical integration.

Never reads player outcome data, odds, gameplay logs, or 2026 Week-5 results.
This is NOT a historical backtest or an OOS certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED = {
    "te": (
        Path("data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json"),
        "TE_R5P_PRODUCTION_MODEL_V1",
    ),
    "wr": (
        Path("data/models/wr_r15_production_model_v1/wr_r15_production_model_v1.json"),
        "WR_R15_PRODUCTION_MODEL_V1",
    ),
}
DEFAULT_SEASONS = (2023, 2024, 2025)
VERSION = "PLAYER_TARGET_SHARE_TRAJECTORY_HISTORICAL_INTEGRATION_GATE_V1"


def read_model(path: Path, expected_version: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing frozen specialist model: {path}")
    raw = path.read_bytes()
    if not raw:
        raise RuntimeError(f"empty frozen specialist model: {path}")
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid frozen specialist JSON: {path}") from exc
    if not isinstance(doc, dict) or doc.get("model_version") != expected_version:
        raise RuntimeError(f"frozen model version mismatch: {path}")
    seasons = doc.get("training_seasons")
    if not isinstance(seasons, list) or not seasons or any(
        type(s) is not int or s < 1900 or s > 2100 for s in seasons
    ) or len(seasons) != len(set(seasons)):
        raise RuntimeError(f"invalid model training-seasons contract: {path}")
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "model_version": expected_version,
        "training_seasons": sorted(seasons),
    }


def gate(
    *,
    te_path: Path = EXPECTED["te"][0],
    wr_path: Path = EXPECTED["wr"][0],
    target_seasons: tuple[int, ...] = DEFAULT_SEASONS,
) -> dict[str, Any]:
    seasons = tuple(target_seasons)
    if not seasons or len(seasons) != len(set(seasons)) or any(
        type(s) is not int or s < 1900 or s > 2100 for s in seasons
    ):
        raise ValueError("invalid target seasons")
    assets = {
        "te": read_model(te_path, EXPECTED["te"][1]),
        "wr": read_model(wr_path, EXPECTED["wr"][1]),
    }
    per_season = []
    any_overlap = False
    for season in sorted(seasons):
        overlapping = sorted(name for name, asset in assets.items()
                             if season in asset["training_seasons"])
        any_overlap |= bool(overlapping)
        per_season.append({
            "season": season,
            "overlapping_specialists": overlapping,
            "declared_training_membership_nonoverlap": not bool(overlapping),
            "independent_oos_validated": False,
            "historical_trajectory_requires_four_prior_same_team_season_games": True,
        })
    return {
        "version": VERSION,
        "status": (
            "HISTORICAL_INTEGRATION_DIAGNOSTIC_ONLY__SPECIALIST_TRAINING_OVERLAP"
            if any_overlap else
            "SOURCE_PARITY_AND_AS_OF_GATE_REQUIRED__MEMBERSHIP_NONOVERLAP"
        ),
        "research_only": True,
        "production_changed": False,
        "sportsbook_inputs_used": False,
        "target_outcomes_read": False,
        "week5_2026_outcomes_read": False,
        "parameters_fit": 0,
        "candidate_seasons": sorted(seasons),
        "assets": assets,
        "season_dispositions": per_season,
        "historical_grading_executed": False,
        "independent_oos_certified": False,
        "next_action": (
            "Verify point-in-time M38/TE-R5P/WR-R15 baseline and roster/source "
            "parity; any overlapping-season test must remain diagnostic only. "
            "Preserve original prospective Week-5+ gates."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--te-model", type=Path, default=EXPECTED["te"][0])
    p.add_argument("--wr-model", type=Path, default=EXPECTED["wr"][0])
    p.add_argument("--target-seasons", default="2023,2024,2025")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    try:
        years = tuple(int(x.strip()) for x in args.target_seasons.split(","))
        result = gate(te_path=args.te_model, wr_path=args.wr_model, target_seasons=years)
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
