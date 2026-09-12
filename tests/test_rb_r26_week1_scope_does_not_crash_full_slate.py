"""RB R26 receptions refinement is Week-1-only, same as RB P3 rushing.

A non-Week-1 Full Slate run must never abort the whole pricing job just
because the R26 route isn't qualified for that week -- it must fail closed
on the R26 route only (keep the pre-R26 V4 receptions result) and let
everything else price normally. This mirrors the identical fix already
locked in for RB P3 in test_rb_pricing_adapter_v1.py.
"""
from __future__ import annotations

from pathlib import Path


def test_r26_module_gates_on_week_before_calling_the_frozen_adapter():
    src = Path("scripts/run_pricing_with_full_roster_universe_v5_production.py").read_text(encoding="utf-8")
    assert "promoted RB R26 receptions refinement is Week-1-only" in src
    assert "if weeks[0] != 1:" in src
    assert "return v4_result" in src


def test_r26_lineage_stamp_does_not_require_r26_artifacts_outside_week1():
    src = Path("scripts/run_pricing_with_full_roster_universe_v5_production.py").read_text(encoding="utf-8")
    assert "def _stamp_r26_pricing_lineage(*, week: int) -> dict:" in src
    assert "RB_R26_NOT_APPLICABLE_OUTSIDE_WEEK1" in src
    assert "week = int(resolve_week())" in src
    assert "_stamp_r26_pricing_lineage(week=week)" in src
