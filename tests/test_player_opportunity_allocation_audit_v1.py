from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_player_opportunity_allocation_audit_v1 as audit
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate


def _tiny_final() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "event_id": "G1", "team": "IND", "opponent": "HOU",
            "player": "Receiver One", "player_clean_key": "receiverone",
            "position": "WR", "entitlement_tgt_share": 0.25,
            "rules_tgt_share": 0.25, "rules_rush_share": 0.00,
            "rules_ypt": 8.0, "rules_ypc": 4.0,
            "rules_plays_est": 64.0, "rules_pass_rate": 0.60,
        },
        {
            "event_id": "G1", "team": "IND", "opponent": "HOU",
            "player": "Back One", "player_clean_key": "backone",
            "position": "RB", "entitlement_tgt_share": 0.12,
            "rules_tgt_share": 0.12, "rules_rush_share": 0.55,
            "rules_ypt": 6.5, "rules_ypc": 4.4,
            "rules_plays_est": 64.0, "rules_pass_rate": 0.60,
        },
    ])


def test_trace_wrapper_is_exact_simulation_noop():
    frame = _tiny_final()
    baseline = explicit_simulate(frame, iterations=300, seed=77)
    traced, trace = audit._trace_explicit_simulation(frame, iterations=300, seed=77)

    assert baseline.iterations == traced.iterations
    assert set(baseline.values) == set(traced.values)
    for key in baseline.values:
        assert np.array_equal(baseline.values[key], traced.values[key]), key

    assert set(trace["opportunity_type"]) == {"targets", "carries"}
    assert len(trace) == 4
    assert trace["predicted_opportunities"].ge(0).all()
    assert trace["predicted_team_opportunity_mean"].gt(0).all()
    assert trace["residual_probability"].between(0, 1).all()


def test_probability_transform_matches_canonical_residual_bucket():
    clean, raw_sum, probs, residual = audit._probability_transform(np.array([0.2, 0.3]))
    assert raw_sum == pytest.approx(0.5)
    assert clean.tolist() == pytest.approx([0.2, 0.3])
    assert probs.tolist() == pytest.approx([0.2, 0.3])
    assert residual == pytest.approx(0.5)

    clean, raw_sum, probs, residual = audit._probability_transform(np.array([0.8, 0.7]))
    assert raw_sum == pytest.approx(1.5)
    assert probs.sum() == pytest.approx(0.95)
    assert residual == pytest.approx(0.05)


def test_opportunity_bins_are_frozen_semantic_bins():
    assert audit._opportunity_bin("QB", "pass_attempts", 0) == "ZERO"
    assert audit._opportunity_bin("QB", "pass_attempts", 20) == "01_20"
    assert audit._opportunity_bin("QB", "pass_attempts", 31) == "31_40"
    assert audit._opportunity_bin("QB", "pass_attempts", 41) == "41_PLUS"

    assert audit._opportunity_bin("RB", "carries", 3) == "01_03"
    assert audit._opportunity_bin("RB", "carries", 8) == "04_08"
    assert audit._opportunity_bin("RB", "carries", 14) == "09_14"
    assert audit._opportunity_bin("RB", "carries", 15) == "15_PLUS"

    assert audit._opportunity_bin("WR", "targets", 2) == "01_02"
    assert audit._opportunity_bin("WR", "targets", 5) == "03_05"
    assert audit._opportunity_bin("WR", "targets", 8) == "06_08"
    assert audit._opportunity_bin("WR", "targets", 9) == "09_PLUS"


def test_summary_reports_compression_without_fitting():
    rows = pd.DataFrame({
        "position_family": ["WR"] * 4,
        "opportunity_type": ["targets"] * 4,
        "predicted_opportunities": [2.0, 3.0, 4.0, 5.0],
        "actual_opportunities": [0.0, 2.0, 6.0, 10.0],
        "linked_yards_error": [10.0, 5.0, -20.0, -40.0],
        "linked_count_error": [1.0, 0.5, -2.0, -4.0],
        "actual_opportunity_bin": ["ZERO", "01_02", "06_08", "09_PLUS"],
    })
    s = audit._build_summary(rows)
    overall = s.loc[s["actual_opportunity_bin"].eq("ALL")].iloc[0]
    assert overall["rows"] == 4
    assert overall["signed_bias"] == pytest.approx(-1.0)
    assert overall["opportunity_error_vs_linked_yards_error_pearson"] > 0.9
    assert overall["opportunity_error_vs_linked_count_error_pearson"] > 0.9


def test_zero_state_reports_fixed_thresholds_without_selecting_one():
    rows = pd.DataFrame({
        "position_family": ["TE"] * 5,
        "opportunity_type": ["targets"] * 5,
        "predicted_opportunities": [0.2, 0.7, 1.2, 3.0, 5.0],
        "actual_opportunities": [0.0, 0.0, 0.0, 2.0, 6.0],
    })
    z = audit._zero_state_audit(rows).iloc[0]
    assert "zero_precision_pred_lt_0_5" in z.index
    assert "zero_recall_pred_lt_0_5" in z.index
    assert "zero_precision_pred_lt_1_0" in z.index
    assert "zero_recall_pred_lt_1_0" in z.index
    assert z["actual_zero_rows"] == 3


def test_actual_for_fails_to_zero_only_downstream():
    amap = {
        ("IND", "playera", "rec_yards"): 7.0,
        ("IND", "playera", "rush_yards"): 11.0,
        ("IND", "qba", "pass_yards"): 33.0,
    }
    assert audit._actual_for(
        amap, team="IND", player_key="playera", opportunity_type="targets"
    ) == (7.0, "NFLVERSE_WEEKLY_STATS")
    assert audit._actual_for(
        amap, team="IND", player_key="playera", opportunity_type="carries"
    ) == (11.0, "NFLVERSE_WEEKLY_STATS")
    assert audit._actual_for(
        amap, team="IND", player_key="qba", opportunity_type="pass_attempts"
    ) == (33.0, "NFLVERSE_WEEKLY_STATS")

    actual, source = audit._actual_for(
        amap, team="IND", player_key="missing", opportunity_type="targets"
    )
    assert actual == 0.0
    assert source.endswith("_ZERO")
