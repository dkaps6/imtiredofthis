"""The target-pool validator owns its team-scope check natively.

It calls eligible_team_set_v1.validate_current_team_set on the materialized
football simulation universe.  Under explicit current availability that helper
requires exact equality with the certified current football teams; in legacy
mode it retains the historical 32-team contract.  The downstream certification
transformer must not patch this validator at runtime.
"""
from __future__ import annotations

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


def test_target_pool_validator_uses_native_current_team_set_check():
    text = TARGET_POOL.read_text(encoding="utf-8")
    assert "validate_current_team_set(frame[\"team\"]" in text
    assert 'if frame["team"].nunique() != 32:' not in text


def test_seam_no_longer_defines_a_target_pool_path_constant():
    assert not hasattr(seam, "TARGET_POOL")
