import numpy as np
from scripts.backtest.audit_receiving_opportunity import _allocator_probabilities


def test_allocator_probabilities_preserve_subcap_shares():
    probs, raw_sum, residual = _allocator_probabilities(np.array([0.40,0.25,0.10]))
    assert np.isclose(raw_sum,0.75)
    assert np.allclose(probs,[0.40,0.25,0.10])
    assert np.isclose(residual,0.25)


def test_allocator_probabilities_cap_overfull_target_pool():
    probs, raw_sum, residual = _allocator_probabilities(np.array([0.60,0.30,0.20]))
    assert np.isclose(raw_sum,1.10)
    assert np.isclose(probs.sum(),0.95)
    assert np.isclose(residual,0.05)
    assert np.isclose(probs[0],0.60*0.95/1.10)
