"""Guards for the walk-forward study.

The whole claim rests on predictions being out-of-sample by construction, so that is
what the guards check hardest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_direct_quantile_fit_v1 as V
from scripts.research import measure_walk_forward_quantiles_v1 as W


def test_form_is_imported_not_reimplemented():
    assert W._design is V._design
    assert W.fit_tail is V.fit_tail
    assert W.evaluate is V.evaluate
    assert W.TAILS is V.TAILS and W.ARMS is V.ARMS


def test_walk_forward_structure_is_frozen():
    assert W.FIRST_FIT_SEASON == 2013
    assert W.MIN_PRIOR_SEASONS == 3
    assert W.TARGET_SEASONS == tuple(range(2016, 2026))
    assert 2026 not in W.TARGET_SEASONS, "2026 is live and must be excluded"
    assert set(W.MARKETS) == {"rec_yards", "rush_yards", "receptions"}


def test_market_sigmas_match_production():
    from scripts.pricing_v2 import SIGMA
    for m, spec in W.MARKETS.items():
        assert spec["production_sigma"] == SIGMA[m], m


def test_coefficients_never_see_the_target_season_or_later():
    """The core anti-leakage property, asserted directly on the fitting calls."""
    seen = {}
    real_fit = W.fit_tail

    def spy(fit, arm, q):
        seen.setdefault(id(fit), sorted(fit["season"].unique().tolist()))
        return real_fit(fit, arm, q)

    rng = np.random.default_rng(0)
    panels = {}
    for yr in range(2013, 2019):
        n = 300
        panels[yr] = pd.DataFrame({
            "season": yr, "pid": [f"p{i % 40}" for i in range(n)],
            "player": [f"P{i % 40}" for i in range(n)], "pos": "WR",
            "week": rng.integers(4, 18, n), "mu_hat": rng.uniform(20, 90, n),
            "centrality": rng.uniform(.1, .3, n), "y": rng.uniform(0, 160, n)})
    W.fit_tail = spy
    try:
        pred, coefs = W.walk_forward(panels, "rec_yards")
    finally:
        W.fit_tail = real_fit

    for fit_seasons in seen.values():
        # every fit set must be strictly earlier than the earliest target it could serve
        assert max(fit_seasons) < max(coefs.keys()), "a fit set reached the target season"
    for target in coefs:
        assert target in W.TARGET_SEASONS
    assert pred["season"].min() >= min(W.TARGET_SEASONS)


def test_walk_forward_skips_targets_without_enough_prior_history():
    rng = np.random.default_rng(1)
    panels = {}
    for yr in (2013, 2014, 2016):          # 2015 missing, 2016 has only 2 prior seasons
        n = 200
        panels[yr] = pd.DataFrame({
            "season": yr, "pid": [f"p{i % 30}" for i in range(n)],
            "player": "x", "pos": "WR", "week": rng.integers(4, 18, n),
            "mu_hat": rng.uniform(20, 90, n), "centrality": rng.uniform(.1, .3, n),
            "y": rng.uniform(0, 160, n)})
    # 2016 has only 2 prior seasons, so NO target qualifies -- must fail closed loudly
    with pytest.raises(RuntimeError, match="no target season qualified"):
        W.walk_forward(panels, "rec_yards")


def test_rule_3_joint_cannot_be_won_by_trading_one_tail_for_the_other():
    """The sum must fall, which is the whole point of the joint form."""
    ev = {
        "Q0": {"frac_well_calibrated": 0.40},
        "Q1": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
        "Q2": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
               "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
        "bias_by_mean_tercile": {
            "Q0": {"high": {"floor_bias": 10.0, "ceiling_bias": -1.0}},
            # floor fixed but ceiling blown out: sum 11.0 -> 13.0, must FAIL
            "Q1": {"high": {"floor_bias": -1.0, "ceiling_bias": 12.0}},
            # both improved: sum 11.0 -> 4.0, must PASS
            "Q2": {"high": {"floor_bias": -2.0, "ceiling_bias": -2.0}},
        },
    }
    out = W.decide_dual(ev, production_sigma=24.0)
    assert out["Q1"]["rule_3_joint"] is False
    assert out["Q2"]["rule_3_joint"] is True
    assert out["H1_JOINT"] is True and out["preferred_JOINT"] == "Q2"


def test_rule_3_joint_tolerance_is_scaled_to_each_market():
    assert W.decide_dual({"Q0": {"frac_well_calibrated": 0.4},
                          "Q1": {"frac_well_calibrated": 0.5, "frac_players_pin20_improved": .9,
                                 "frac_players_pin80_improved": .9, "frac_rows_crossed": 0.0},
                          "Q2": {"frac_well_calibrated": 0.5, "frac_players_pin20_improved": .9,
                                 "frac_players_pin80_improved": .9, "frac_rows_crossed": 0.0},
                          "bias_by_mean_tercile": {k: {"high": {"floor_bias": 1.0, "ceiling_bias": 1.0}}
                                                   for k in ("Q0", "Q1", "Q2")}},
                         production_sigma=1.8)["rule_3_joint_tolerance"] == pytest.approx(0.45)


def test_both_verdicts_are_published_neither_suppressed():
    ev = {"Q0": {"frac_well_calibrated": 0.40},
          "Q1": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
                 "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
          "Q2": {"frac_well_calibrated": 0.60, "frac_players_pin20_improved": 0.9,
                 "frac_players_pin80_improved": 0.9, "frac_rows_crossed": 0.0},
          "bias_by_mean_tercile": {k: {"high": {"floor_bias": 1.0, "ceiling_bias": 1.0}}
                                   for k in ("Q0", "Q1", "Q2")}}
    out = W.decide_dual(ev, 26.0)
    for arm in ("Q1", "Q2"):
        assert "SUPPORTED_STRICT" in out[arm] and "SUPPORTED_JOINT" in out[arm]
        assert "rule_3_strict" in out[arm] and "rule_3_joint" in out[arm]
    assert "H1_STRICT" in out and "H1_JOINT" in out


def test_main_refuses_to_score_without_the_preregistration(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="preregistration missing"):
        W.main()
