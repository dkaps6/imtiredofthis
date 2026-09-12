"""RB R22 receiving-tail refinement is Week-1-only, same as RB P3/R26.

A non-Week-1 Full Slate run must never abort the whole pricing job just
because the R22 route isn't qualified for that week -- it must fail closed
on the R22 route only (keep the pre-R22 V3 receiving-yard result) and let
everything else price and certify normally. This mirrors the identical
fixes already locked in for RB P3 (test_rb_pricing_adapter_v1.py) and RB
R26 (test_rb_r26_week1_scope_does_not_crash_full_slate.py).
"""
from __future__ import annotations

from pathlib import Path


def test_r22_module_gates_on_week_before_calling_the_frozen_adapter():
    src = Path("scripts/run_pricing_with_full_roster_universe_v4_production.py").read_text(encoding="utf-8")
    assert "promoted RB R22 receiving-tail refinement is Week-1-only" in src
    assert "if weeks[0] != 1:" in src
    assert "return v3_result" in src
    assert "_write_r22_not_applicable_adapter_stub" in src


def test_r22_lineage_stamp_does_not_require_r22_artifacts_outside_week1():
    src = Path("scripts/run_pricing_with_full_roster_universe_v4_production.py").read_text(encoding="utf-8")
    assert "def _stamp_pricing_lineage(*, week: int) -> dict:" in src
    assert "RB_R22_NOT_APPLICABLE_OUTSIDE_WEEK1" in src
    assert "_stamp_pricing_lineage(week=int(resolve_week()))" in src


def test_r22_v5_clarify_step_handles_non_week1_disposition():
    src = Path("scripts/run_pricing_with_full_roster_universe_v5_production.py").read_text(encoding="utf-8")
    assert "def _clarify_r22_lineage_after_r26(*, week: int) -> None:" in src
    assert "v4.R22_NOT_APPLICABLE" in src
    assert "_clarify_r22_lineage_after_r26(week=week)" in src


def test_market_model_lineage_v3_handles_non_week1_r22_disposition():
    src = Path("scripts/audit_market_model_lineage_v3.py").read_text(encoding="utf-8")
    assert 'adapter.get("disposition") == "RB_R22_NOT_APPLICABLE_OUTSIDE_WEEK1"' in src
    assert "rb_r22_receiving_tail_consumed" in src


def test_certified_stack_v3_handles_non_week1_r22_disposition():
    src = Path("scripts/validate_certified_full_slate_stack_v3.py").read_text(encoding="utf-8")
    assert 'adapter.get("disposition") == "RB_R22_NOT_APPLICABLE_OUTSIDE_WEEK1"' in src
    assert "R22_NOT_APPLICABLE_WITH_DECLARED_SCIENCE_LIMITATIONS" in src
