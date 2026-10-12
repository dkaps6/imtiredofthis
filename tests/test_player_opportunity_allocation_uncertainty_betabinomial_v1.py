from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from scripts.research.audit_player_opportunity_allocation_uncertainty_betabinomial_v1 import (
    beta_parameters,
    score_row,
)


def test_beta_parameters_preserve_frozen_mean_and_variance():
    mu=.20
    v=.01
    got=beta_parameters(mu,v)
    assert got is not None
    alpha,beta,kappa=got
    assert abs(kappa-15.0)<1e-12
    assert abs(alpha-3.0)<1e-12
    assert abs(beta-12.0)<1e-12
    dist=betabinom(40,alpha,beta)
    assert abs(dist.mean()-40*mu)<1e-12


def test_zero_excess_variance_is_exact_baseline():
    r=pd.Series({
        "N":40,"y":8,
        "prior_share_mean":.20,
        "excess_share_var":0.0,
    })
    got=score_row(r)
    assert got["candidate_changed"] is False
    assert got["candidate_ineligible_reason"]=="NO_POSITIVE_EXCESS_VAR"
    assert abs(got["candidate_crps"]-got["baseline_crps"])<1e-12
    assert got["paired_nll_improvement"]==0.0
    assert got["expected_count_gap"]==0.0


def test_positive_valid_excess_variance_changes_width_not_mean():
    r=pd.Series({
        "N":40,"y":12,
        "prior_share_mean":.20,
        "excess_share_var":.01,
    })
    got=score_row(r)
    assert got["candidate_changed"] is True
    assert np.isfinite(got["candidate_crps"])
    assert np.isfinite(got["candidate_nll"])
    assert np.isfinite(got["baseline_nll"])
    assert np.isfinite(got["paired_nll_improvement"])
    assert got["expected_count_gap"]==0.0
    assert got["candidate_90_low"] <= got["candidate_90_high"]


def test_impossible_beta_variance_fails_to_baseline_without_clipping():
    mu=.20
    r=pd.Series({
        "N":40,"y":8,
        "prior_share_mean":mu,
        "excess_share_var":mu*(1-mu),
    })
    got=score_row(r)
    assert got["candidate_changed"] is False
    assert got["candidate_ineligible_reason"]=="INVALID_METHOD_OF_MOMENTS_VARIANCE"
    assert got["candidate_crps"]==got["baseline_crps"]
    assert got["paired_nll_improvement"]==0.0
