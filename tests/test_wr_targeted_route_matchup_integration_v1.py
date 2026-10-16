import numpy as np
import pandas as pd

from scripts.research.persist_wr_targeted_route_matchup_integration_v1 import (
    _empirical_crps,
    _ols_slope_clipped,
)


def test_ols_slope_uses_development_relationship_and_clips():
    x=np.arange(1.0,201.0)
    y=0.4*x+3.0
    raw,beta,n=_ols_slope_clipped(x,y)
    assert n==200
    assert np.isclose(raw,0.4,atol=1e-10)
    assert np.isclose(beta,0.4,atol=1e-10)

    raw2,beta2,_=_ols_slope_clipped(x,2.0*x)
    assert raw2>1.0
    assert beta2==1.0


def test_empirical_crps_is_zero_for_perfect_degenerate_draws():
    draws=np.full(2000,12.5)
    assert np.isclose(_empirical_crps(draws,12.5),0.0,atol=1e-12)


def test_empirical_crps_positive_for_miss():
    draws=np.full(2000,10.0)
    assert np.isclose(_empirical_crps(draws,12.0),2.0,atol=1e-12)
