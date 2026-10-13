import numpy as np

from scripts.research.evaluate_distribution_q90_upper_tail_hinge_v1 import (
    empirical_crps,
    q90_hinge,
)


def test_q90_hinge_preserves_mean_and_nonnegative():
    x=np.linspace(0.0,100.0,2000)
    mean=float(x.mean())
    y=q90_hinge(x,mean)
    assert len(y)==len(x)
    assert np.isfinite(y).all()
    assert (y>=0).all()
    assert abs(float(y.mean())-mean)<=1e-10
    assert float(np.quantile(y,.99))>float(np.quantile(x,.99))


def test_q90_hinge_changes_only_after_hinge_before_realign_semantics():
    x=np.arange(1.0,2001.0)
    y=q90_hinge(x,float(x.mean()))
    assert abs(float(y.mean())-float(x.mean()))<=1e-10
    assert float(y[-1]/y[-200])>float(x[-1]/x[-200])


def test_empirical_crps_prefers_draws_near_actual():
    near=np.array([9.0,10.0,11.0])
    far=np.array([0.0,1.0,2.0])
    assert empirical_crps(near,10.0)<empirical_crps(far,10.0)
