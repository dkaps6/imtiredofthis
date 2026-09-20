from scripts.research.evaluate_event_regime_reliability_v1 import (
    FAMILIES,
    _primary_gate,
    _replication_gate,
)


def _m(rows=100, mae=1.0, rmse=1.0, p90=1.0, p95=1.0, sd=1.0, cat=0.10):
    return {
        "rows": rows,
        "mae": mae,
        "rmse": rmse,
        "bias": 0.0,
        "median_ae": 0.5,
        "p75_ae": 0.8,
        "p90_ae": p90,
        "p95_ae": p95,
        "residual_sd": sd,
        "catastrophic_miss_rate": cat,
    }


def test_frozen_primary_gate_passes_only_material_degradation():
    non = _m()
    event = _m(mae=1.06, rmse=1.06, p90=1.06, p95=1.04, sd=1.04, cat=0.121)
    gate = _primary_gate(event, non)
    assert gate["event_rows_gate"]
    assert gate["mae_gate"]
    assert gate["rmse_gate"]
    assert gate["p90_gate"]
    assert gate["catastrophic_gate"]
    assert gate["season_gate_pass"]


def test_primary_gate_does_not_accept_sub_five_percent_mae_signal():
    non = _m()
    event = _m(mae=1.049, rmse=1.06, p90=1.06, p95=1.04, sd=1.04, cat=0.13)
    gate = _primary_gate(event, non)
    assert not gate["mae_gate"]
    assert not gate["season_gate_pass"]


def test_replication_gate_requires_directional_error_and_two_three_percent_metrics():
    non = _m()
    event = _m(mae=1.04, rmse=1.04, p90=1.01, p95=1.04, sd=1.00, cat=0.11)
    gate = _replication_gate(event, non)
    assert gate["mae_gate"]
    assert gate["rmse_gate"]
    assert gate["catastrophic_gate"]
    assert gate["three_pct_metric_count"] == 3
    assert gate["two_of_five_three_pct_gate"]
    assert gate["season_gate_pass"]


def test_source_thin_team_change_is_not_in_frozen_reliability_families():
    assert all(spec["event"] != "team_change_flag" for spec in FAMILIES.values())


def test_only_outcome_free_qualified_binary_events_enter_v1():
    allowed = {
        "joint_player_room_transition_flag",
        "target_room_churn_flag",
        "rush_room_churn_flag",
    }
    assert {spec["event"] for spec in FAMILIES.values()} <= allowed
