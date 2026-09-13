"""Regression guard for a real production incident (2026-09-13, live run
34776845601): current_team_scope_expected (the certified-eligible team
count) legitimately shrinks over a game day as more games kick off -- it is
never required to equal the constant 32-team football-only universe, only
to never exceed it.
"""
import pytest

from scripts.audit_market_model_lineage_v1 import _validate_qb_c2_coverage_contract


def _c2(**overrides):
    base = {
        "football_qb_rows": 32,
        "selected_qb_rows": 30,
    }
    base.update(overrides)
    return base


def _stamp(**overrides):
    base = {
        "football_qbs": 32,
        "current_team_scope_expected": 32,
        "pass_yard_qbs": 26,
        "c2_selected_qbs": 24,
        "c2_selected_football_qbs": 30,
        "sportsbook_offer_coverage_defines_football_universe": False,
    }
    base.update(overrides)
    return base


def test_accepts_shrunk_current_scope_late_in_a_game_day():
    result = _validate_qb_c2_coverage_contract(_c2(), _stamp(current_team_scope_expected=4))
    assert result["football_qbs"] == 32


def test_accepts_full_32_team_current_scope():
    result = _validate_qb_c2_coverage_contract(_c2(), _stamp(current_team_scope_expected=32))
    assert result["football_qbs"] == 32


def test_rejects_current_scope_exceeding_football_universe():
    with pytest.raises(RuntimeError, match="exceeds the football-only universe"):
        _validate_qb_c2_coverage_contract(_c2(), _stamp(current_team_scope_expected=33))


def test_still_rejects_production_stamp_mismatch():
    with pytest.raises(RuntimeError, match="football universe differs"):
        _validate_qb_c2_coverage_contract(_c2(), _stamp(football_qbs=31))
