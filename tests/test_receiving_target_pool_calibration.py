import numpy as np
from scripts.backtest.run_receiving_target_pool_calibration import _transform, _allocator_probabilities


def test_top7_keeps_only_seven_largest_shares():
    s=np.array([.30,.20,.15,.10,.08,.06,.04,.03,.02])
    out=_transform(s,"top7")
    assert np.count_nonzero(out)==7
    assert np.allclose(out[:7],s[:7])
    assert np.allclose(out[7:],0.0)


def test_minimum_share_gate_zeroes_fringe_players():
    s=np.array([.30,.04,.029,.01])
    out=_transform(s,"min_03")
    assert np.allclose(out,[.30,.04,0.0,0.0])


def test_cumulative_gate_keeps_players_until_threshold_reached():
    s=np.array([.40,.25,.15,.10,.05])
    out=_transform(s,"cum_80")
    assert np.allclose(out,[.40,.25,.15,0.0,0.0])


def test_allocator_caps_player_probability_at_95_percent_total():
    probs,gated_sum,normalized_sum,residual=_allocator_probabilities(np.array([.60,.50,.30]))
    assert np.isclose(gated_sum,1.40)
    assert np.isclose(normalized_sum,.95)
    assert np.isclose(probs.sum(),.95)
    assert np.isclose(residual,.05)
