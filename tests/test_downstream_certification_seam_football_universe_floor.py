"""Regression guards for the single certified current football universe.

The weekly schedule may remain full-league, but artifacts built from the explicit
current-role authority must exactly match that current team/game scope. These
checks prevent reintroducing the Sept. 13 superset/floor repair loop.
"""
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


def test_priced_qb_subset_is_bounded_by_current_scope(restore_stack_v1):
    seam.main()
    text = STACK_V1.read_text(encoding="utf-8")
    assert '_require(int(stamp.get("current_team_scope_expected", -1)) == expected_team_count' in text
    assert '0 < int(stamp.get("pass_yard_qbs", 0)) <= expected_team_count' in text
