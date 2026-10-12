from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.audit_player_opportunity_share_volatility_mechanism_v2 import (
    excess_spearman,
    partial_spearman,
)


def test_partial_spearman_detects_signal_beyond_controls():
    n=40
    sd=np.linspace(.01,.20,n)
    mean=np.linspace(.10,.30,n)
    sampling=np.linspace(.02,.04,n)
    surprise=.5*sd + .01*np.sin(np.arange(n))
    q=pd.DataFrame({
        "prior_share_sd":sd,
        "share_surprise":surprise,
        "prior_share_mean":mean,
        "expected_sampling_sd":sampling,
    })
    assert partial_spearman(q)>0


def test_excess_spearman_expected_direction():
    q=pd.DataFrame({
        "excess_share_sd":[0,.01,.03,.08,.12],
        "share_surprise":[.01,.02,.04,.10,.20],
    })
    assert excess_spearman(q)>0


def test_sampling_variance_subtraction_definition():
    mu=.20
    n=np.array([20.,20.,20.,20.])
    sampling_var=float(np.mean(mu*(1-mu)/n))
    raw_sd=np.sqrt(sampling_var)
    excess=max(0.0,raw_sd*raw_sd-sampling_var)
    assert abs(excess)<1e-12

    role_sd=raw_sd+.10
    excess2=max(0.0,role_sd*role_sd-sampling_var)
    assert excess2>0
