import numpy as np
import pandas as pd

from scripts.pricing_v2 import _projection, _team_volume, _fair_market_prob


def test_projection_is_independent_of_vegas_line():
    base = {
        "pace": 28.0,
        "proe": 0.0,
        "target_share": 0.25,
        "ypt": 9.0,
        "rush_share": 0.0,
        "def_pass_epa_opp": 0.0,
        "wind_mph": 0.0,
    }
    row_a = pd.Series({**base, "line": 60.5})
    row_b = pd.Series({**base, "line": 95.5})
    assert np.isclose(_projection(row_a, "rec_yards"), _projection(row_b, "rec_yards"))


def test_team_volume_uses_team_possession_scale_not_full_game_clock():
    tv = _team_volume(pd.Series({"pace": 28.0, "proe": 0.0}))
    assert 50 <= tv["plays"] <= 80
    assert tv["plays"] < 100


def test_two_way_market_is_devigged():
    over, under = _fair_market_prob(-110, -110)
    assert np.isclose(over, 0.5)
    assert np.isclose(under, 0.5)
