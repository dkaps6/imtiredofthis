import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "evaluate_prism_wr_te_player_signed_component_bias_v1.py"
spec = importlib.util.spec_from_file_location("signed_bias", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _row(season, week, team, player, current, prior_mean=0.0, n=4):
    return {
        "season": season,
        "week": week,
        "team": team,
        "player_clean_key": player,
        "prior_component_rows_available": n,
        "prior_component_min_week": max(1, week - n),
        "prior_component_max_week": max(1, week - 1),
        "source_eligible": True,
        "identity_gap": 0.0,
        "prior8_opportunity_error_mean": prior_mean,
        "prior8_efficiency_error_mean": prior_mean,
        "current_opportunity_error": current,
        "current_efficiency_error": current,
        "position": "WR",
    }


def test_position_prior_excludes_target_player_and_same_week():
    x = pd.DataFrame([
        _row(2024, 1, "A", "alpha", 1.0),
        _row(2024, 1, "B", "bravo", 3.0),
        _row(2024, 2, "A", "alpha", 5.0, prior_mean=2.0),
        _row(2024, 2, "B", "bravo", 7.0, prior_mean=4.0),
    ])
    z = m.add_position_priors(x, "OPPORTUNITY")
    a2 = z.loc[(z["week"] == 2) & (z["player_clean_key"] == "alpha")].iloc[0]
    b2 = z.loc[(z["week"] == 2) & (z["player_clean_key"] == "bravo")].iloc[0]
    assert a2["position_prior_mean"] == pytest.approx(3.0)
    assert b2["position_prior_mean"] == pytest.approx(1.0)
    assert a2["position_prior_other_player_rows"] == 1
    assert b2["position_prior_other_player_rows"] == 1


def test_exact_four_pseudo_observation_shrinkage():
    x = pd.DataFrame([
        _row(2024, 1, "A", "alpha", 1.0),
        _row(2024, 1, "B", "bravo", 3.0),
        _row(2024, 2, "A", "alpha", 5.0, prior_mean=2.0, n=4),
    ])
    z = m.add_position_priors(x, "OPPORTUNITY")
    a2 = z.loc[(z["week"] == 2) & (z["player_clean_key"] == "alpha")].iloc[0]
    assert a2["position_prior_mean"] == pytest.approx(3.0)
    assert a2["hierarchical_signed_bias"] == pytest.approx(2.5)


def test_team_week_cluster_is_stable():
    x = pd.DataFrame([
        _row(2024, 3, "AAA", "alpha", 1.0),
        _row(2024, 3, "AAA", "bravo", 2.0),
    ])
    z = m.add_position_priors(x, "EFFICIENCY")
    assert z["team_week_cluster"].nunique() == 1
    assert z["team_week_cluster"].iloc[0] == "2024-3-AAA"
