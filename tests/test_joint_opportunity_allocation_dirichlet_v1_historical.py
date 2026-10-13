from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from scripts.research.audit_joint_opportunity_allocation_dirichlet_v1_historical import (
    build_room_record,
    dirichlet_multinomial_logpmf,
    multinomial_logpmf,
    normalize_player_probs,
)


def test_normalize_player_probs_preserves_residual_mass():
    p,res=normalize_player_probs(np.array([0.60,0.50,0.10]))
    assert abs(float(p.sum())+res-1.0)<1e-12
    assert float(p.sum())<=0.95+1e-12
    assert res>=0.05-1e-12


def test_multinomial_logpmf_is_finite_for_valid_room():
    got=multinomial_logpmf(np.array([2,3]),np.array([0.4,0.6]))
    assert np.isfinite(got)


def test_one_eligible_joint_reduces_to_beta_binomial():
    N=20
    q=.30
    rho=.10
    kappa=1/rho-1
    alpha=q*kappa
    beta=(1-q)*kappa
    y=8
    dm=dirichlet_multinomial_logpmf(
        np.array([y,N-y]),
        np.array([alpha,beta]),
    )
    bb=float(betabinom(N,alpha,beta).logpmf(y))
    assert abs(dm-bb)<1e-10


def test_room_record_preserves_expected_counts():
    canonical=pd.DataFrame([
        {
            "team":"IND","player_clean_key":"backa","canonical_q":0.30,
            "canonical_residual":0.40,
        },
        {
            "team":"IND","player_clean_key":"backb","canonical_q":0.30,
            "canonical_residual":0.40,
        },
    ])
    source=pd.DataFrame([
        {
            "season":2024,"week":8,"team":"IND","cell":"RB_CARRY_SHARE",
            "game_id":"g","player_identity_key":"gsis:a","player_clean_key":"backa",
            "rho_i":0.10,"N":20,"y":8,
        },
        {
            "season":2024,"week":8,"team":"IND","cell":"RB_CARRY_SHARE",
            "game_id":"g","player_identity_key":"gsis:b","player_clean_key":"backb",
            "rho_i":np.nan,"N":20,"y":5,
        },
    ])
    rec,marg=build_room_record(
        season=2024,week=8,cell="RB_CARRY_SHARE",team="IND",game_id="g",
        canonical=canonical,source=source,
    )
    assert rec is not None
    assert rec["eligible_players"]==1
    assert abs(rec["complement_probability"]-0.70)<1e-12
    assert rec["max_expected_count_gap"]<=1e-12
    assert len(marg)==1
    assert marg[0]["expected_count_gap"]<=1e-12
