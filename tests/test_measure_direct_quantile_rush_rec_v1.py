"""Guards for the rushing/receptions replication.

The value of a replication is that the form cannot drift, so the guards check exactly
that: the machinery is imported from the validated study, the market definitions match
production's constants, and the holdout stays naive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_direct_quantile_fit_v1 as V
from scripts.research import measure_direct_quantile_rush_rec_v1 as R


def test_form_is_imported_from_the_validated_study_not_reimplemented():
    """A replication must reuse the frozen machinery, not a copy of it."""
    assert R._design is V._design
    assert R.fit_tail is V.fit_tail
    assert R.evaluate is V.evaluate
    assert R.decide is V.decide
    assert R.TAILS is V.TAILS
    assert R.ARMS is V.ARMS
    assert R.MIN_HOLDOUT_GAMES == V.MIN_HOLDOUT_GAMES


def test_market_sigmas_match_production():
    """If pricing_v2's constants change, this study's Q0 arm is stale."""
    from scripts.pricing_v2 import SIGMA
    assert R.MARKETS["rush_yards"]["production_sigma"] == SIGMA["rush_yards"] == 24.0
    assert R.MARKETS["receptions"]["production_sigma"] == SIGMA["receptions"] == 1.8


def test_holdout_is_naive_for_every_market():
    assert R.HOLDOUT_SEASONS == (2014, 2015, 2016)
    assert not set(R.HOLDOUT_SEASONS) & set(R.FIT_SEASONS)
    assert not set(R.HOLDOUT_SEASONS) & set(R.REPORTED_SEASONS)
    # the receiving-yards study's holdout must NOT be reused as this one's holdout
    assert not set(R.HOLDOUT_SEASONS) & set(V.HOLDOUT_SEASONS)


def test_markets_are_judged_separately():
    """A win in one market must not be able to carry the other."""
    assert set(R.MARKETS) == {"rush_yards", "receptions"}
    for spec in R.MARKETS.values():
        assert spec["outcome"] and spec["volume"] and spec["positions"]


def test_q0_uses_each_markets_own_constant_and_is_symmetric():
    df = pd.DataFrame({"mu_hat": [40.0, 80.0], "centrality": [0.2, 0.3]})
    for market, spec in R.MARKETS.items():
        p = R.predict(df, "Q0", {}, spec["production_sigma"])
        mu = df["mu_hat"].to_numpy(float)
        assert np.allclose(mu - p["p20"], p["p80"] - mu), f"{market} Q0 must be symmetric"
        assert np.allclose(p["p80"] - p["p20"],
                           2 * abs(V.Z20) * spec["production_sigma"])


def test_receptions_is_flagged_discrete_and_rushing_is_not():
    """The discreteness disclosure must be in the code, not only the prose."""
    assert R.MARKETS["receptions"]["discrete"] is True
    assert R.MARKETS["rush_yards"]["discrete"] is False


def test_rushing_eligibility_is_volume_appropriate():
    """A 20-target threshold would be meaningless for carries."""
    assert R.MARKETS["rush_yards"]["min_prior_volume"] == 60
    assert R.MARKETS["rush_yards"]["min_game_volume"] == 5
    assert R.MARKETS["receptions"]["min_prior_volume"] == 20
    assert R.MARKETS["receptions"]["min_game_volume"] == 1


def test_decision_rule_is_the_same_four_as_the_validated_study():
    ev = {
        "Q0": {"frac_well_calibrated": 0.294},
        "Q1": {"frac_well_calibrated": 0.608, "frac_players_pin20_improved": 0.784,
               "frac_players_pin80_improved": 0.686, "frac_rows_crossed": 0.0},
        "Q2": {"frac_well_calibrated": 0.549, "frac_players_pin20_improved": 0.765,
               "frac_players_pin80_improved": 0.686, "frac_rows_crossed": 0.0},
        "bias_by_mean_tercile": {
            "Q0": {"high": {"floor_bias": 4.03, "ceiling_bias": -6.90}},
            "Q1": {"high": {"floor_bias": -15.03, "ceiling_bias": -1.84}},
            "Q2": {"high": {"floor_bias": -14.83, "ceiling_bias": -2.36}},
        },
    }
    out = R.decide(ev)
    # the real rushing outcome: rules 1/2/4 pass, rule 3 fails on a floor overshoot
    assert out["Q1"]["rule_1_more_calibrated_players"] is True
    assert out["Q1"]["rule_2_pinball_majority_BOTH_tails"] is True
    assert out["Q1"]["rule_3_high_mean_floor_better_ceiling_not_worse"] is False
    assert out["Q1"]["rule_4_no_crossed_quantiles"] is True
    assert out["H1_SUPPORTED"] is False, "a floor overshoot must not pass"


def test_main_refuses_to_score_without_the_preregistration(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="preregistration missing"):
        R.main()
