"""Guards for the dispersion study.

It reports a null on its own hypothesis while proving the production constant wrong, so
both halves must stay falsifiable: the machinery has to be able to recover a known
dispersion, and the decision rule must not be satisfiable by one tail alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_player_dispersion_v1 as D


def _panel(n=800, sd_slope=0.30, sd_base=11.0, seed=0):
    """Panel whose true sigma really is sd_base + sd_slope*mu."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(15.0, 95.0, n)
    sigma = sd_base + sd_slope * mu
    return pd.DataFrame({
        "season": 2020, "pid": [f"p{i % 50}" for i in range(n)],
        "player": [f"P{i % 50}" for i in range(n)], "pos": "WR",
        "week": rng.integers(4, 19, n), "mu_hat": mu,
        "centrality": rng.uniform(0.08, 0.35, n),
        "y": mu + rng.normal(0, sigma),
    })


def test_frozen_constants_match_the_preregistration():
    assert D.FIT_SEASONS == (2020, 2021)
    assert D.HOLDOUT_SEASON == 2022
    assert 2022 not in D.REPORTED_SEASONS, "the clean holdout must not also be a reported season"
    assert D.PRODUCTION_SIGMA == 26.0
    assert D.MIN_PRIOR_TARGETS == 20
    assert D.SIGMA_FLOOR == 5.0
    assert D.TARGET_COVERAGE == 0.60
    assert D.CALIBRATED_TOLERANCE == 0.10


def test_holdout_is_disjoint_from_fit():
    assert D.HOLDOUT_SEASON not in D.FIT_SEASONS


def test_sigma_estimator_recovers_a_known_dispersion():
    """If the estimator cannot recover a planted sigma, the null means nothing."""
    panel = _panel(n=4000, sd_slope=0.30, sd_base=11.0, seed=5)
    beta = D.fit_sigma(panel, "S1")
    sigma = D.predict_sigma(panel, "S1", beta)
    truth = 11.0 + 0.30 * panel["mu_hat"].to_numpy(float)
    assert np.corrcoef(sigma, truth)[0, 1] > 0.99
    assert abs(np.median(sigma / truth) - 1.0) < 0.08, "scale is off by more than 8%"


def test_s0_ignores_every_player_property():
    """The production arm must be a constant -- that is the defect under test."""
    panel = _panel(n=100)
    sigma = D.predict_sigma(panel, "S0", None)
    assert np.all(sigma == D.PRODUCTION_SIGMA)
    assert sigma.std() == 0.0


def test_sigma_floor_is_enforced():
    panel = _panel(n=50)
    tiny = D.predict_sigma(panel, "S1", np.array([-100.0, 0.0]))
    assert np.all(tiny >= D.SIGMA_FLOOR)


def test_pinball_loss_is_asymmetric_in_the_right_direction():
    y = np.array([100.0])
    # at q=0.20 under-prediction should cost more than over-prediction
    assert D._pinball(y, np.array([90.0]), 0.20) < D._pinball(y, np.array([110.0]), 0.20)
    assert D._pinball(y, np.array([110.0]), 0.80) < D._pinball(y, np.array([90.0]), 0.80)


def test_decision_rule_cannot_be_satisfied_by_one_tail_alone():
    """The exact way this study failed: a floor win must not carry the verdict."""
    ev = {
        "S0": {"frac_well_calibrated": 0.40},
        "S1": {"frac_well_calibrated": 0.55, "frac_players_pin20_improved": 0.95,
               "frac_players_pin80_improved": 0.31, "frac_rows_on_sigma_floor": 0.0,
               "sigma_range": [20.0, 56.0]},
        "S2": {"frac_well_calibrated": 0.55, "frac_players_pin20_improved": 0.95,
               "frac_players_pin80_improved": 0.80, "frac_rows_on_sigma_floor": 0.0,
               "sigma_range": [20.0, 56.0]},
        "bias_by_mean_tercile": {"S0": {"high": {"floor_bias": 9.4}},
                                 "S1": {"high": {"floor_bias": -3.9}},
                                 "S2": {"high": {"floor_bias": -3.9}}},
    }
    out = D.decide(ev)
    assert out["S1"]["SUPPORTED"] is False, "one-tail win must not pass"
    assert out["S1"]["rule_2_pinball_majority_both_tails"] is False
    assert out["S2"]["SUPPORTED"] is True, "both tails improving must pass"


def test_main_refuses_to_score_without_the_preregistration(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="preregistration missing"):
        D.main()


def test_anomaly_gate_labels_before_evidence():
    assert D.state_label(0, 0, 0.0, 40.0) == "DNP_INACTIVE"
    assert D.state_label(3, 0, 10.0, 40.0) == "PARTIAL_GAME"
    assert D.state_label(5, 0, 38.0, 40.0) == "HEALTHY_FULL"
    assert D.state_label(0, 0, float("nan"), 40.0) == "UNRESOLVED_CONTEXT"


def test_lognormal_secondary_preserves_the_mean_and_is_right_skewed():
    """Its asymmetry must put the floor further from the mean than the ceiling."""
    panel = _panel(n=400, seed=9)
    sigma = D.predict_sigma(panel, "S1", D.fit_sigma(panel, "S1"))
    out = D.lognormal_secondary(panel, sigma)
    assert out["mean_downside_distance"] > out["mean_upside_distance"]
    assert "cannot change the verdict" in out["note"]
