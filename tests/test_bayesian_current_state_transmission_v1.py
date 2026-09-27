import pandas as pd
import pytest

from scripts.research.audit_bayesian_current_state_transmission_v1 import (
    _weight_audit,
    classify,
)


def _summary(rows):
    return pd.DataFrame(rows)


def test_frozen_effective_weight_example_is_exact():
    w = _weight_audit()
    assert w["2"]["playerform_current_weight"] == pytest.approx(2 / 6)
    assert w["2"]["bayes_current_weight_prior6plus"] == pytest.approx(2 / 11)
    assert w["2"]["bayes_to_playerform_weight_ratio"] == pytest.approx((2 / 11) / (2 / 6))


def test_systemic_mismatch_requires_all_three_metrics_and_week3_analogue():
    rows = []
    for pos, metric in [("RB", "rush_share"), ("WR", "tgt_share"), ("TE", "tgt_share")]:
        rows += [
            {"sample": "2024", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": 0.01},
            {"sample": "2025", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": 0.02},
            {
                "sample": "WEEK3_ANALOGUE_CURRENT_GAMES_2_POOLED",
                "position": pos,
                "metric": metric,
                "mean_ae_delta_bayes_minus_playerform": 0.03,
            },
        ]
    result = classify(_summary(rows))
    assert result["disposition"] == "BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED"
    assert all(x["season_status"] == "PLAYERFORM_BETTER" for x in result["per_metric"])


def test_metric_specific_mismatch_does_not_promote_systemic_claim():
    rows = []
    specs = {
        ("RB", "rush_share"): (0.01, 0.02, 0.03),
        ("WR", "tgt_share"): (-0.01, -0.02, -0.03),
        ("TE", "tgt_share"): (0.01, -0.01, 0.02),
    }
    for (pos, metric), (d24, d25, dw3) in specs.items():
        rows += [
            {"sample": "2024", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": d24},
            {"sample": "2025", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": d25},
            {
                "sample": "WEEK3_ANALOGUE_CURRENT_GAMES_2_POOLED",
                "position": pos,
                "metric": metric,
                "mean_ae_delta_bayes_minus_playerform": dw3,
            },
        ]
    result = classify(_summary(rows))
    assert result["disposition"] == "BAYESIAN_CURRENT_STATE_TRANSMISSION_METRIC_SPECIFIC_MISMATCH"


def test_no_mismatch_when_playerform_never_wins_both_seasons():
    rows = []
    for pos, metric in [("RB", "rush_share"), ("WR", "tgt_share"), ("TE", "tgt_share")]:
        rows += [
            {"sample": "2024", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": -0.01},
            {"sample": "2025", "position": pos, "metric": metric, "mean_ae_delta_bayes_minus_playerform": -0.02},
            {
                "sample": "WEEK3_ANALOGUE_CURRENT_GAMES_2_POOLED",
                "position": pos,
                "metric": metric,
                "mean_ae_delta_bayes_minus_playerform": -0.03,
            },
        ]
    result = classify(_summary(rows))
    assert result["disposition"] == "BAYESIAN_CURRENT_STATE_TRANSMISSION_NO_MISMATCH"
