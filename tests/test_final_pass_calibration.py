import math

from scripts.backtest.run_final_pass_calibration import calibrated_share


def test_fixed_share_candidates_are_exact():
    assert calibrated_share("fixed_53", team_rate=0.61, league_rate=0.57) == 0.53
    assert calibrated_share("fixed_55", team_rate=0.61, league_rate=0.57) == 0.55
    assert calibrated_share("fixed_57", team_rate=0.61, league_rate=0.57) == 0.57


def test_team_identity_is_centered_on_55_not_historical_league_level():
    # Team is +4 points above its historical league; 10% identity should add
    # only 0.4 points to the 55% anchor, not inherit the 57% league baseline.
    out = calibrated_share("team_identity_10", team_rate=0.61, league_rate=0.57)
    assert math.isclose(out, 0.554, abs_tol=1e-12)


def test_larger_team_identity_weight_moves_farther_from_55():
    low = calibrated_share("team_identity_05", team_rate=0.61, league_rate=0.57)
    high = calibrated_share("team_identity_15", team_rate=0.61, league_rate=0.57)
    assert 0.55 < low < high
