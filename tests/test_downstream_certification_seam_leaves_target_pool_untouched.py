"""Target-pool team scope is now native, not transformer-owned.

The validator calls validate_current_team_set directly. With the shared helper's
exact current-team contract, missing or extra current teams fail closed without
rewriting the target-pool validator during Full Slate.
"""
from pathlib import Path
import pytest
from scripts.operations import apply_current_availability_downstream_certification_seam_v1 as seam

TARGET_POOL = Path("scripts/validate_team_target_pool_full_universe_v2.py")
OTHER_TARGETS = [
    Path("scripts/audit_market_model_lineage_v2_core.py"),
    Path("scripts/audit_market_model_lineage_v3.py"),
    Path("scripts/validate_certified_full_slate_stack_v1.py"),
    Path("scripts/validate_certified_full_slate_stack_v2_core.py"),
    Path("scripts/validate_certified_full_slate_stack_v3.py"),
]

@pytest.fixture
def restore_seam_targets():
    originals = {p: p.read_text(encoding="utf-8") for p in [TARGET_POOL, *OTHER_TARGETS]}
    yield
    for p, text in originals.items():
        p.write_text(text, encoding="utf-8")


def test_target_pool_validator_is_not_rewritten_by_seam(restore_seam_targets):
    before = TARGET_POOL.read_text(encoding="utf-8")
    seam.main()
    assert TARGET_POOL.read_text(encoding="utf-8") == before


def test_target_pool_validator_uses_shared_exact_current_team_helper():
    text = TARGET_POOL.read_text(encoding="utf-8")
    assert 'from scripts.utils.eligible_team_set_v1 import validate_current_team_set' in text
    assert 'validate_current_team_set(frame["team"], label="target-pool football universe")' in text


def test_downstream_seam_no_longer_owns_target_pool_path():
    assert not hasattr(seam, "TARGET_POOL")
