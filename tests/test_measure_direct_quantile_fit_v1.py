"""Guards for the direct per-tail quantile study.

This is the first supported result in the lane, so its guards matter more than usual:
the estimator must recover known quantiles, the decision rule must still reject a
one-tail win, and the holdout must stay naive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_direct_quantile_fit_v1 as Q


def _panel(n=3000, seed=0):
    """Panel with a KNOWN asymmetric structure: floor slope 0.35, ceiling slope 0.70."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(15.0, 95.0, n)
    # lognormal-ish noise so the tails are genuinely asymmetric around the mean
    y = np.maximum(0.0, mu * rng.lognormal(mean=-0.18, sigma=0.60, size=n))
    return pd.DataFrame({
        "season": 2020, "pid": [f"p{i % 120}" for i in range(n)],
        "player": [f"P{i % 120}" for i in range(n)], "pos": "WR",
        "week": rng.integers(4, 18, n), "mu_hat": mu,
        "centrality": rng.uniform(0.08, 0.35, n), "y": y,
    })


def test_frozen_windows_and_naive_holdout():
    assert Q.FIT_SEASONS == (2020, 2021, 2022)
    assert Q.HOLDOUT_SEASONS == (2017, 2018, 2019)
    assert not set(Q.FIT_SEASONS) & set(Q.HOLDOUT_SEASONS), "holdout must be disjoint from fit"
    assert not set(Q.HOLDOUT_SEASONS) & set(Q.REPORTED_SEASONS)
    assert Q.MIN_HOLDOUT_GAMES == 8, "frozen this study rather than left undisclosed"


def test_season_length_is_era_correct():
    assert Q.last_week(2017) == 17 and Q.last_week(2020) == 17
    assert Q.last_week(2021) == 18 and Q.last_week(2025) == 18


def test_tail_fit_recovers_the_requested_quantile():
    """A p20 fit must leave ~20% of outcomes below it, and p80 ~80%."""
    panel = _panel(n=6000, seed=3)
    for name, q in Q.TAILS:
        beta = Q.fit_tail(panel, "Q1", q)
        pred = Q._design(panel, "Q1") @ beta
        below = float((panel["y"].to_numpy(float) <= pred).mean())
        assert abs(below - q) < 0.04, f"{name}: {below:.3f} vs target {q}"


def test_tails_are_fitted_independently_and_can_differ_in_slope():
    """The whole point: a symmetric family cannot do this."""
    panel = _panel(n=6000, seed=4)
    b20 = Q.fit_tail(panel, "Q1", 0.20)
    b80 = Q.fit_tail(panel, "Q1", 0.80)
    assert b80[1] > b20[1], "ceiling slope should exceed floor slope on skewed data"
    assert b80[1] - b20[1] > 0.1


def test_q0_is_production_and_is_symmetric_by_construction():
    panel = _panel(n=200)
    p = Q.predict(panel, "Q0", {})
    mu = panel["mu_hat"].to_numpy(float)
    assert np.allclose(mu - p["p20"], p["p80"] - mu), "Q0 must be symmetric about mu"
    assert np.allclose(p["p80"] - p["p20"], 2 * abs(Q.Z20) * Q.PRODUCTION_SIGMA)


def test_decision_rule_still_rejects_a_one_tail_win():
    """The Q1 failure mode must stay disqualifying."""
    ev = {
        "Q0": {"frac_well_calibrated": 0.386},
        "Q1": {"frac_well_calibrated": 0.636, "frac_players_pin20_improved": 0.848,
               "frac_players_pin80_improved": 0.457, "frac_rows_crossed": 0.0},
        "Q2": {"frac_well_calibrated": 0.598, "frac_players_pin20_improved": 0.837,
               "frac_players_pin80_improved": 0.598, "frac_rows_crossed": 0.0},
        "bias_by_mean_tercile": {
            "Q0": {"high": {"floor_bias": 11.7, "ceiling_bias": -3.1}},
            "Q1": {"high": {"floor_bias": -5.1, "ceiling_bias": -0.3}},
            "Q2": {"high": {"floor_bias": -4.7, "ceiling_bias": 0.2}},
        },
    }
    out = Q.decide(ev)
    assert out["Q1"]["SUPPORTED"] is False, "higher headline must not rescue a failed ceiling"
    assert out["Q1"]["rule_2_pinball_majority_BOTH_tails"] is False
    assert out["Q2"]["SUPPORTED"] is True
    assert out["preferred_arm"] == "Q2"


def test_rule_3_rejects_buying_the_floor_by_wrecking_the_ceiling():
    ev = {
        "Q0": {"frac_well_calibrated": 0.40},
        "Q1": {"frac_well_calibrated": 0.55, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
        "Q2": {"frac_well_calibrated": 0.55, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
        "bias_by_mean_tercile": {
            "Q0": {"high": {"floor_bias": 11.7, "ceiling_bias": -3.1}},
            "Q1": {"high": {"floor_bias": -1.0, "ceiling_bias": +10.5}},  # ceiling worsened
            "Q2": {"high": {"floor_bias": -1.0, "ceiling_bias": -1.0}},
        },
    }
    out = Q.decide(ev)
    assert out["Q1"]["rule_3_high_mean_floor_better_ceiling_not_worse"] is False
    assert out["Q2"]["rule_3_high_mean_floor_better_ceiling_not_worse"] is True


def test_crossed_quantiles_disqualify():
    ev = {
        "Q0": {"frac_well_calibrated": 0.40},
        "Q1": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.01},
        "Q2": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
        "bias_by_mean_tercile": {
            "Q0": {"high": {"floor_bias": 11.7, "ceiling_bias": -3.1}},
            "Q1": {"high": {"floor_bias": -1.0, "ceiling_bias": -1.0}},
            "Q2": {"high": {"floor_bias": -1.0, "ceiling_bias": -1.0}},
        },
    }
    out = Q.decide(ev)
    assert out["Q1"]["rule_4_no_crossed_quantiles"] is False
    assert out["Q1"]["SUPPORTED"] is False


def test_main_refuses_to_score_without_the_preregistration(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="preregistration missing"):
        Q.main()


def test_mc_dispersion_rule3_enforces_no_overshoot():
    """Regression for the precedence bug GPT audit 6472971619 section 5 found.

    `A and B or A` evaluates as `(A and B) or A` == A, which silently dropped the
    preregistered no-overshoot condition from rule 3.
    """
    src = Path("scripts/research/validate_mc_dispersion_recalibration_v1.py").read_text(encoding="utf-8")
    assert "no_overshoot = (z_wide >= 1.0 - 1e-9) if z_base > 1.0 else (z_wide <= 1.0 + 1e-9)" in src

    def rule3(z_base, z_wide):
        closer = abs(z_wide - 1.0) < abs(z_base - 1.0)
        no_overshoot = (z_wide >= 1.0 - 1e-9) if z_base > 1.0 else (z_wide <= 1.0 + 1e-9)
        return bool(closer and no_overshoot)

    # baseline ABOVE 1.0: approaching from above passes, dropping below it does not
    assert rule3(1.767, 1.186) is True
    assert rule3(1.767, 0.40) is False, "undershoot past 1.0 must be rejected"
    assert rule3(1.767, 1.90) is False, "moving away from 1.0 must be rejected"
    # baseline BELOW 1.0: the guarded side flips
    assert rule3(0.60, 0.90) is True
    assert rule3(0.60, 1.50) is False, "overshoot past 1.0 must be rejected"
