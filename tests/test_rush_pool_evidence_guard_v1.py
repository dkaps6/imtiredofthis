import numpy as np
from scripts.research.rush_pool_evidence_guard_v1 import (
    FALLBACK_STATE, _baseline_mask, _candidate_mask, _player_probabilities
)


def test_week1_is_exact_noop():
    shares=np.array([.40,.30,.20,.10,.05,.04])
    states=np.array(["prior+current",FALLBACK_STATE,"prior_only",FALLBACK_STATE,"prior+current","prior_only"],dtype=object)
    assert np.array_equal(_baseline_mask(shares), _candidate_mask(shares, states, week=1))


def test_evidenced_players_fill_top5_before_fallback():
    shares=np.array([.40,.35,.30,.25,.20,.19])
    states=np.array(["prior+current",FALLBACK_STATE,"prior_only","prior+current","prior_only","prior+current"],dtype=object)
    got=_candidate_mask(shares,states,week=2)
    # evidenced indices 0,2,3,4,5 must all be selected; fallback index 1 cannot displace index 5
    assert got.tolist() == [True,False,True,True,True,True]


def test_fallback_fills_unused_slots():
    shares=np.array([.40,.35,.30,.25])
    states=np.array(["prior+current",FALLBACK_STATE,"prior_only",FALLBACK_STATE],dtype=object)
    got=_candidate_mask(shares,states,week=2)
    assert got.tolist() == [True,True,True,True]


def test_probability_cap_and_residual_semantics_preserved():
    shares=np.array([.50,.40,.30,.20,.10,.05])
    mask=_baseline_mask(shares)
    probs,resid=_player_probabilities(shares,mask)
    assert mask.sum() == 5
    assert abs(probs.sum() - .95) < 1e-12
    assert abs(resid - .05) < 1e-12
    assert abs(probs.sum()+resid-1.0) < 1e-12
