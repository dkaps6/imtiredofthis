import numpy as np
from scripts.research.evaluate_wr_dual_reliability_trust_integration_v1 import corr

def test_corr_positive():
    x=np.arange(20,dtype=float)
    y=2*x+1
    assert corr(x,y)>0.999999
