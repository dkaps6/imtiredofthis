import numpy as np

from scripts.research.evaluate_prism_rb_receiving_team_context_history_authority_reset_v1 import (
    enumerate_states,
)


def test_exact_state_enumeration_conserves_probability_and_room_mass():
    q=np.array([0.30,0.20],dtype=float)
    p=np.array([0.80,0.60],dtype=float)
    w,r,r0=enumerate_states(q,p)
    assert abs(float(w.sum())-1.0) <= 1e-12
    assert np.allclose(r.sum(axis=1)+r0,1.0,atol=1e-12)


def test_reset_changes_only_target_participation_input():
    q=np.array([0.30,0.20],dtype=float)
    baseline=np.array([1.0,0.70],dtype=float)
    candidate=baseline.copy()
    candidate[0]=0.60
    wb,rb,rb0=enumerate_states(q,baseline)
    wc,rc,rc0=enumerate_states(q,candidate)
    eb=wb@rb
    ec=wc@rc
    assert abs(float(wb.sum())-1.0) <= 1e-12
    assert abs(float(wc.sum())-1.0) <= 1e-12
    assert abs(float(eb.sum())+float(wb@rb0)-1.0) <= 1e-12
    assert abs(float(ec.sum())+float(wc@rc0)-1.0) <= 1e-12
    assert candidate[1] == baseline[1]
    assert candidate[0] != baseline[0]
