import numpy as np
import pandas as pd
import pytest

from scripts.modeling.rb_rush_synthesis_v1 import (
    RB_RUSH_SYNTHESIS_VERSION,
    WEEK1_ROUTE,
    WEEKS2_18_ROUTE,
    apply_p3,
    compose_p3_row,
)


def test_week1_is_exact_stack_override():
    out = compose_p3_row(week=1, stack_att=12.0, stack_yards=51.25, enriched_att=None)
    assert out["rb_synthesis_proj"] == 51.25
    assert out["rb_synthesis_route"] == WEEK1_ROUTE
    assert out["rb_synthesis_version"] == RB_RUSH_SYNTHESIS_VERSION
    assert out["rb_synthesis_applied"] == 1
    assert out["rb_ypc_fallback_used"] == 0


def test_weeks2_18_exact_enriched_opportunity_times_stack_efficiency():
    out = compose_p3_row(week=7, stack_att=10.0, stack_yards=47.5, enriched_att=13.0)
    assert out["rb_stack_implied_ypc"] == 4.75
    assert out["rb_synthesis_proj"] == 61.75
    assert out["rb_synthesis_route"] == WEEKS2_18_ROUTE
    assert out["rb_ypc_fallback_used"] == 0


def test_weeks2_18_uses_m94c_ypc_only_when_stack_efficiency_unavailable():
    out = compose_p3_row(
        week=2,
        stack_att=0.1,
        stack_yards=0.4,
        enriched_att=3.0,
        m94c_implied_ypc=4.2,
    )
    assert out["rb_stack_implied_ypc"] == 4.2
    assert out["rb_synthesis_proj"] == pytest.approx(12.6)
    assert out["rb_ypc_fallback_used"] == 1


def test_weeks2_18_fail_closed_without_enriched_opportunity():
    with pytest.raises(RuntimeError, match="enriched_att"):
        compose_p3_row(week=3, stack_att=10.0, stack_yards=45.0)


def test_weeks2_18_fail_closed_without_any_efficiency_source():
    with pytest.raises(RuntimeError, match="fallback is unavailable"):
        compose_p3_row(week=4, stack_att=0.0, stack_yards=0.0, enriched_att=2.0)


def test_apply_p3_preserves_rows_and_routes():
    frame = pd.DataFrame(
        [
            {"player": "A", "week": 1, "stack_att": 10.0, "stack_yards": 40.0, "enriched_att": np.nan},
            {"player": "B", "week": 8, "stack_att": 8.0, "stack_yards": 36.0, "enriched_att": 12.0},
        ]
    )
    out = apply_p3(frame)
    assert out["player"].tolist() == ["A", "B"]
    assert out["rb_synthesis_proj"].tolist() == [40.0, 54.0]
    assert out["rb_synthesis_route"].tolist() == [WEEK1_ROUTE, WEEKS2_18_ROUTE]


def test_contract_has_no_sportsbook_parameters():
    import inspect

    params = set(inspect.signature(compose_p3_row).parameters)
    forbidden = {"line", "odds", "over_odds", "under_odds", "consensus_line", "vegas"}
    assert not (params & forbidden)
