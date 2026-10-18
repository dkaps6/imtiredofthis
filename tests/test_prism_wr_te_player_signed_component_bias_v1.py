import pandas as pd

from scripts.research.audit_prism_wr_te_player_signed_component_bias_v1 import (
    audit_slice,
)


def _row(week, player, prior_rows=4, max_week=None, eligible=True):
    if max_week is None:
        max_week = week - 1
    return {
        "position": "WR",
        "season": 2023,
        "week": week,
        "team": "A",
        "player_clean_key": player,
        "prior_component_rows_available": prior_rows,
        "prior_component_min_week": max(1, max_week - prior_rows + 1),
        "prior_component_max_week": max_week,
        "source_eligible": eligible,
        "prior8_opportunity_error_mean": 1.0,
        "prior8_efficiency_error_mean": -1.0,
        "current_opportunity_error": 0.5,
        "current_efficiency_error": -0.5,
        "identity_gap": 0.0,
    }


def test_chronology_rejects_same_week_prior_max():
    rows = [_row(8, "p1", prior_rows=4, max_week=8)]
    # Repeat enough rows to avoid support being relevant to this unit assertion.
    x = pd.DataFrame(rows)
    rec = audit_slice(x, "WR", 2023)
    assert rec["chronology_violations"] == 1
    assert rec["source_gates"]["chronology"] is False


def test_signed_fields_are_source_integrity_not_predictive_score():
    rows = [_row(8, "p1")]
    x = pd.DataFrame(rows)
    rec = audit_slice(x, "WR", 2023)
    assert rec["signed_field_missingness"]["prior8_opportunity_error_mean"] == 0
    assert rec["signed_field_missingness"]["current_efficiency_error"] == 0
    assert "mean_delta" not in rec
    assert "correlation" not in rec


def test_prior_window_bounds_are_enforced():
    x = pd.DataFrame([
        _row(9, "p1", prior_rows=3),
        _row(9, "p2", prior_rows=9),
    ])
    rec = audit_slice(x, "WR", 2023)
    assert rec["chronology_violations"] == 2
    assert rec["source_gates"]["chronology"] is False
