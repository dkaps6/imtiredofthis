"""QB C2 football coverage must match the certified current football scope.

Sportsbook pass-yard coverage may be a smaller subset, but the football C2
starter universe itself must not carry already-started/withheld teams.
"""
import pytest

from scripts.audit_market_model_lineage_v1 import _validate_qb_c2_coverage_contract


def _c2(**overrides):
    base = {
        "football_qb_rows": 4,
        "selected_qb_rows": 3,
    }
    base.update(overrides)
    return base


def _stamp(**overrides):
    base = {
        "football_qbs": 4,
        "current_team_scope_expected": 4,
        "pass_yard_qbs": 2,
        "c2_selected_qbs": 2,
        "c2_selected_football_qbs": 3,
        "sportsbook_offer_coverage_defines_football_universe": False,
    }
    base.update(overrides)
    return base


def test_accepts_exact_current_football_scope_with_smaller_priced_subset():
    result = _validate_qb_c2_coverage_contract(_c2(), _stamp())
    assert result["football_qbs"] == 4
    assert result["priced_qbs"] == 2


def test_rejects_football_universe_larger_than_current_scope():
    with pytest.raises(RuntimeError, match="differs from certified current team scope"):
        _validate_qb_c2_coverage_contract(
            _c2(football_qb_rows=32, selected_qb_rows=30),
            _stamp(football_qbs=32, current_team_scope_expected=4, c2_selected_football_qbs=30),
        )


def test_rejects_current_scope_larger_than_football_universe():
    with pytest.raises(RuntimeError, match="differs from certified current team scope"):
        _validate_qb_c2_coverage_contract(_c2(), _stamp(current_team_scope_expected=5))


def test_still_rejects_production_stamp_mismatch():
    with pytest.raises(RuntimeError, match="football universe differs"):
        _validate_qb_c2_coverage_contract(_c2(), _stamp(football_qbs=3))
