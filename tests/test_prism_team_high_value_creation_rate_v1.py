import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_team_high_value_creation_rate_v1 import (
    adjacent_binomial_crps,
    adjacent_binomial_pmf,
    prior_rate_state,
)


def test_adjacent_binomial_mixture_preserves_continuous_parent_mean():
    mu = 31.4
    p = 0.17
    pmf = adjacent_binomial_pmf(mu, p)
    k = np.arange(len(pmf), dtype=float)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.isclose(float((k * pmf).sum()), mu * p, atol=1e-12)
    assert len(pmf) - 1 == 32


def test_adjacent_binomial_integer_parent_reduces_to_finite_binomial():
    pmf = adjacent_binomial_pmf(10.0, 0.2)
    assert len(pmf) == 11
    assert np.isclose(pmf.sum(), 1.0)
    assert np.isclose(float((np.arange(11) * pmf).sum()), 2.0)


def test_crps_penalizes_actual_above_projected_finite_support():
    near = adjacent_binomial_crps(5.2, 0.2, 1)
    far = adjacent_binomial_crps(5.2, 0.2, 8)
    assert far > near


def test_prior_rate_state_is_strictly_prior_and_last_four():
    rows = []
    for week in range(1, 7):
        rows.append(
            {
                "season": 2025,
                "week": week,
                "game_id": f"g{week}",
                "team": "IND",
                "opponent": "TEN",
                "rush_source_present": True,
                "target_source_present": True,
                "team_rush_attempts": 20,
                "inside10_rush": week,
                "team_targets": 30,
                "red_zone_targets": 3,
            }
        )
        rows.append(
            {
                "season": 2025,
                "week": week,
                "game_id": f"g{week}",
                "team": "TEN",
                "opponent": "IND",
                "rush_source_present": True,
                "target_source_present": True,
                "team_rush_attempts": 20,
                "inside10_rush": 1,
                "team_targets": 30,
                "red_zone_targets": 2,
            }
        )
    team = pd.DataFrame(rows)
    spec = {
        "source_flag": "rush_source_present",
        "team_hv": "inside10_rush",
        "team_parent": "team_rush_attempts",
    }
    state = prior_rate_state(team, spec, 2025)
    w6 = state.loc[(state["week"] == 6) & (state["team"] == "IND")].iloc[0]
    # Last four prior IND games are Weeks 2-5: 2+3+4+5 over 4*20.
    assert w6["history_games"] == 4
    assert np.isclose(w6["p_team"], 14 / 80)
    assert w6["history_max_week"] == 5
    # Target Week 6 value (=6) cannot enter the predictor.
    assert np.isclose(w6["history_high_value_sum"], 14)
