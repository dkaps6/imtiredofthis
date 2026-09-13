"""validate_team_target_pool_full_universe_v2.py audits
football_simulation_universe.csv and target_entitlement_v1_trace.csv, both
explicitly sportsbook-independent full-league artifacts that the universe
builder (run_pricing_with_full_roster_universe_v1.py) hard-requires to cover
all 32 teams regardless of kickoff-timing pricing eligibility. The downstream
certification seam must never scope this file's team-count checks down to
the shrinking current-eligible count -- that made every real pricing run
fail the moment any team's game kicked off, since the 32-team universe never
shrinks while the eligible-for-pricing set does.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.operations import apply_current_availability_downstream_certification_seam_v1 as seam

TARGET_POOL = Path("scripts/validate_team_target_pool_full_universe_v2.py")

TOUCHED_BY_OTHER_TRANSFORMS = [
    Path("scripts/audit_market_model_lineage_v2_core.py"),
    Path("scripts/audit_market_model_lineage_v3.py"),
    Path("scripts/validate_certified_full_slate_stack_v1.py"),
    Path("scripts/validate_certified_full_slate_stack_v2_core.py"),
    Path("scripts/validate_certified_full_slate_stack_v3.py"),
]


@pytest.fixture
def restore_seam_targets():
    originals = {p: p.read_text(encoding="utf-8") for p in [TARGET_POOL, *TOUCHED_BY_OTHER_TRANSFORMS]}
    yield
    for p, text in originals.items():
        p.write_text(text, encoding="utf-8")


def test_target_pool_validator_is_never_patched_by_the_seam(restore_seam_targets):
    before = TARGET_POOL.read_text(encoding="utf-8")
    seam.main()
    after = TARGET_POOL.read_text(encoding="utf-8")
    assert before == after


def test_target_pool_validator_still_requires_all_32_teams():
    text = TARGET_POOL.read_text(encoding="utf-8")
    assert 'if frame["team"].nunique() != 32:' in text
    assert "expected_current_teams" not in text


def test_seam_no_longer_defines_a_target_pool_path_constant():
    assert not hasattr(seam, "TARGET_POOL")
