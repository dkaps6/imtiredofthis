import numpy as np
from scripts.research.audit_player_active_role_conditional_allocation_source_v1 import alloc_probs

def test_alloc_probs_preserve_residual_and_cap():
    p,r=alloc_probs([0.7,0.5,0.1])
    assert np.isclose(float(p.sum()+r),1.0)
    assert p.sum() <= 0.95 + 1e-12
    assert r >= 0.05 - 1e-12

def test_alloc_probs_zero_mass_is_all_residual():
    p,r=alloc_probs([0.0,0.0])
    assert np.allclose(p,[0.0,0.0])
    assert np.isclose(r,1.0)
