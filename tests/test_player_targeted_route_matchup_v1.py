import numpy as np
import pandas as pd

from scripts.research.evaluate_player_targeted_route_matchup_v1 import (
    bh_adjust,
    matchup_features,
    residualize_depth,
)


def test_matchup_features_is_player_weighted_and_neutral_fallback():
    mix = {"SLANT": 0.75, "GO": 0.25}
    defense = {
        "SLANT": {"n": 20.0, "comp": 16.0, "yards": 180.0},
        "GO": {"n": 5.0, "comp": 2.0, "yards": 90.0},
    }
    league = {
        "SLANT": {"n": 100.0, "comp": 70.0, "yards": 700.0},
        "GO": {"n": 100.0, "comp": 40.0, "yards": 900.0},
    }
    dypt, dcatch = matchup_features(mix, defense, league)
    # GO defense cell is below the frozen n=10 floor, so only SLANT contributes delta.
    assert np.isclose(dypt, 0.75 * ((180.0 / 20.0) - (700.0 / 100.0)))
    assert np.isclose(dcatch, 0.75 * ((16.0 / 20.0) - (70.0 / 100.0)))


def test_bh_adjust_monotone_and_bounded():
    q = bh_adjust([0.001, 0.02, 0.04, 0.20, 0.50, 0.90])
    assert all(0 <= x <= 1 for x in q)
    assert q[0] <= q[1] <= q[2] <= q[3] <= q[4] <= q[5]


def test_depth_residualization_removes_linear_depth_component():
    q = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "prior_player_air_yards": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    z, coverage = residualize_depth(q, "x", "y")
    assert coverage == 1.0
    assert np.allclose(z["x_depth_resid"], 0.0, atol=1e-10)
    assert np.allclose(z["y_depth_resid"], 0.0, atol=1e-10)
