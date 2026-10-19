import numpy as np

from scripts.research.evaluate_prism_te_receiving_team_context_stage1_participation_authority_reset_v1 import (
    enumerate_states,
)


def test_te_state_enumeration_conserves():
    q=np.array([0.20,0.15],dtype=float)
    p=np.array([0.90,0.80],dtype=float)
    w,r,r0=enumerate_states(q,p)
    assert abs(float(w.sum())-1.0)<=1e-12
    assert np.allclose(r.sum(axis=1)+r0,1.0,atol=1e-12)


def test_te_reset_changes_only_target_p():
    q=np.array([0.20,0.15],dtype=float)
    base=np.array([0.95,0.80],dtype=float)
    cand=base.copy(); cand[0]=0.85
    wb,rb,rb0=enumerate_states(q,base)
    wc,rc,rc0=enumerate_states(q,cand)
    assert cand[1]==base[1]
    assert cand[0]!=base[0]
    assert abs(float(wb.sum())-1.0)<=1e-12
    assert abs(float(wc.sum())-1.0)<=1e-12
    assert abs(float((wb@rb).sum())+float(wb@rb0)-1.0)<=1e-12
    assert abs(float((wc@rc).sum())+float(wc@rc0)-1.0)<=1e-12
