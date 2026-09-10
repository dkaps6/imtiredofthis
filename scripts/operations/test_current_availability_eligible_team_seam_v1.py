#!/usr/bin/env python3
"""Regression for the frozen availability-aware current-team coverage seam."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from scripts.utils.eligible_team_set_v1 import validate_current_team_set

NFL32 = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]
ELIGIBLE30 = [t for t in NFL32 if t not in {"NE", "SEA"}]


def must_fail(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f"expected RuntimeError: {label}")


def main() -> int:
    old = os.environ.pop("ACTIVE_ROLES_CSV", None)
    try:
        # Mode 1: no explicit availability seam preserves the original 32-team contract.
        legacy = validate_current_team_set(NFL32, label="legacy fixture")
        assert legacy["mode"] == "LEGACY_32_TEAM"
        assert legacy["observed_teams"] == 32
        must_fail(lambda: validate_current_team_set(ELIGIBLE30, label="legacy missing-game fixture"), "legacy 30 teams")

        # Mode 2: explicit active-role artifact is sole current eligible-team authority.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "roles_current_production_eligible_v1.csv"
            pd.DataFrame({"team": ELIGIBLE30, "player": [f"p{i}" for i in range(30)]}).to_csv(p, index=False)
            os.environ["ACTIVE_ROLES_CSV"] = str(p)
            explicit = validate_current_team_set(ELIGIBLE30, label="explicit fixture")
            assert explicit["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
            assert explicit["expected_teams"] == 30
            assert explicit["observed_teams"] == 30
            assert explicit["canonical_games"] == 15
            assert set(explicit["teams"]) == set(ELIGIBLE30)
            must_fail(
                lambda: validate_current_team_set(ELIGIBLE30[:-1], label="explicit missing-team fixture"),
                "explicit missing team",
            )
            must_fail(
                lambda: validate_current_team_set(ELIGIBLE30 + ["NE"], label="explicit extra-team fixture"),
                "explicit extra team",
            )
    finally:
        if old is None:
            os.environ.pop("ACTIVE_ROLES_CSV", None)
        else:
            os.environ["ACTIVE_ROLES_CSV"] = old
    print("CURRENT_AVAILABILITY_ELIGIBLE_TEAM_SEAM_REGRESSION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
