"""Downstream certification must use the same certified current football
universe as PlayerForm/opportunity/pricing.

The weekly schedule may still cover all league games, but once current
availability/timing has certified the active football team set, downstream
football artifacts must match that set exactly.  A late-Sunday four-team
football universe is therefore valid; silently carrying 32 teams is not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.operations import apply_current_availability_downstream_certification_seam_v1 as seam

STACK_V1 = Path("scripts/validate_certified_full_slate_stack_v1.py")
ALL_SEAM_TARGETS = [
    STACK_V1,
    Path("scripts/audit_market_model_lineage_v2_core.py"),
    Path("scripts/audit_market_model_lineage_v3.py"),
    Path("scripts/validate_certified_full_slate_stack_v2_core.py"),
    Path("scripts/validate_certified_full_slate_stack_v3.py"),
]


@pytest.fixture
def restore_stack_v1():
    originals = {p: p.read_text(encoding="utf-8") for p in ALL_SEAM_TARGETS}
    yield
    for p, text in originals.items():
        p.write_text(text, encoding="utf-8")


def test_football_universe_and_qb_checks_are_exact_current_scope(restore_stack_v1):
    seam.main()
    text = STACK_V1.read_text(encoding="utf-8")
    assert '_require(int(football.get("football_teams", 0)) == expected_team_count' in text
    assert '_require(int(football.get("canonical_games", 0)) == expected_game_count' in text
    assert '_require(int(c2.get("football_qb_rows", 0)) == expected_team_count' in text
    assert '_require(int(stamp.get("football_qbs", -1)) == expected_team_count' in text
    assert '_require(int(football.get("football_teams", 0)) >= expected_team_count' not in text
    assert '_require(int(c2.get("football_qb_rows", 0)) >= expected_team_count' not in text
    assert '_require(int(stamp.get("football_qbs", -1)) >= expected_team_count' not in text


def test_pricing_scope_fields_remain_exact_matches(restore_stack_v1):
    seam.main()
    text = STACK_V1.read_text(encoding="utf-8")
    assert '_require(int(stamp.get("current_team_scope_expected", -1)) == expected_team_count' in text
