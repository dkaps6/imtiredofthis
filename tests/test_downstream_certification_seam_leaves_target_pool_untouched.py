"""validate_team_target_pool_full_universe_v2.py audits
football_simulation_universe.csv and target_entitlement_v1_trace.csv. It now
natively validates team coverage via eligible_team_set_v1.validate_current_team_set,
which tolerates the universe being a superset of the currently-eligible-for-
pricing teams (kickoff lockout narrows pricing eligibility, not the
football-only universe -- and the universe itself can legitimately be below
32 on a bye week). Two live production incidents motivated this:

1. A downstream certification seam used to scope this validator's team-count
   check down to an exact match on the shrinking current-eligible count,
   which made a real pricing run fail the moment any team's game kicked off
   ("target audit expected 4 teams, found 32").
2. Reverting to a hardcoded "exactly 32" then failed the standing bye-week
   replay artifact, whose real football_simulation_universe.csv legitimately
   has only 28 teams. Neither a shrinking exact-match nor a fixed 32 is
   correct; only a floor/superset check is.
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


def test_target_pool_validator_uses_the_native_floor_check_not_a_fixed_count():
    text = TARGET_POOL.read_text(encoding="utf-8")
    assert "validate_current_team_set(frame[\"team\"]" in text
    assert 'if frame["team"].nunique() != 32:' not in text
    assert "expected_current_teams" not in text


def test_seam_no_longer_defines_a_target_pool_path_constant():
    assert not hasattr(seam, "TARGET_POOL")
