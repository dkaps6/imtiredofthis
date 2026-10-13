from __future__ import annotations

import numpy as np
import pytest

from scripts.research.joint_opportunity_allocation_dirichlet_v1 import (
    allocate_baseline_multinomial,
    allocate_joint_dirichlet_group_complement,
    beta_intraclass_rho,
    expected_counts,
    room_rho,
    theoretical_candidate_variance,
)


def test_beta_intraclass_rho_is_exact_dimensionless_transport():
    mu=0.25
    v=0.01875
    assert abs(beta_intraclass_rho(mu,v)-0.10)<1e-12
    assert beta_intraclass_rho(0.0,v) is None
    assert beta_intraclass_rho(mu,0.0) is None


def test_no_eligible_players_is_exact_baseline_path():
    totals=np.full(2000,7,dtype=int)
    probs=np.array([0.30,0.20,0.50])
    eligible=np.array([False,False,False])
    rhos=np.array([np.nan,np.nan,np.nan])

    base=allocate_baseline_multinomial(
        np.random.default_rng(1234),totals,probs
    )
    cand,meta=allocate_joint_dirichlet_group_complement(
        rng=np.random.default_rng(1234),
        totals=totals,
        baseline_probs=probs,
        eligible_mask=eligible,
        player_rhos=rhos,
        cell="RB_CARRY_SHARE",
    )
    assert meta.candidate_applied is False
    assert np.array_equal(cand,base)


def test_joint_allocator_conserves_every_team_total_and_empirical_means():
    draws=400_000
    N=4
    totals=np.full(draws,N,dtype=int)
    probs=np.array([0.25,0.15,0.20,0.40])
    eligible=np.array([True,True,False,False])
    rhos=np.array([0.12,0.30,np.nan,np.nan])

    alloc,meta=allocate_joint_dirichlet_group_complement(
        rng=np.random.default_rng(20261013),
        totals=totals,
        baseline_probs=probs,
        eligible_mask=eligible,
        player_rhos=rhos,
        cell="WR_TARGET_SHARE",
    )
    assert meta.candidate_applied is True
    assert meta.eligible_count==2
    assert np.array_equal(alloc.sum(axis=1),totals)

    empirical=alloc.mean(axis=0)
    expected=expected_counts(N,probs)
    assert float(np.max(np.abs(empirical-expected))) <= 0.01


def test_one_eligible_player_matches_beta_binomial_marginal_variance():
    draws=300_000
    N=10
    q=0.30
    rho=0.20
    totals=np.full(draws,N,dtype=int)
    probs=np.array([q,0.20,0.50])
    eligible=np.array([True,False,False])
    rhos=np.array([rho,np.nan,np.nan])

    alloc,meta=allocate_joint_dirichlet_group_complement(
        rng=np.random.default_rng(77),
        totals=totals,
        baseline_probs=probs,
        eligible_mask=eligible,
        player_rhos=rhos,
        cell="RB_CARRY_SHARE",
    )
    assert meta.candidate_applied is True
    assert abs(meta.rho_room-rho)<1e-12

    empirical_var=float(np.var(alloc[:,0],ddof=0))
    expected_var=theoretical_candidate_variance(N,q,rho)
    assert abs(empirical_var-expected_var)/expected_var < 0.03


def test_complement_internal_expected_means_remain_baseline():
    draws=300_000
    N=10
    probs=np.array([0.30,0.20,0.50])
    eligible=np.array([True,False,False])
    rhos=np.array([0.18,np.nan,np.nan])

    alloc,_=allocate_joint_dirichlet_group_complement(
        rng=np.random.default_rng(88),
        totals=np.full(draws,N,dtype=int),
        baseline_probs=probs,
        eligible_mask=eligible,
        player_rhos=rhos,
        cell="WR_TARGET_SHARE",
    )
    empirical=alloc.mean(axis=0)
    expected=expected_counts(N,probs)
    assert np.all(np.abs(empirical-expected)<0.03)


def test_room_rho_is_variance_weighted_average():
    probs=np.array([0.30,0.20,0.50])
    eligible=np.array([True,True,False])
    rhos=np.array([0.10,0.30,np.nan])
    got=room_rho(probs,eligible,rhos)
    assert got is not None
    rho,kappa,complement=got
    weights=probs[:2]*(1-probs[:2])
    expected=float(np.sum(weights*np.array([0.10,0.30]))/np.sum(weights))
    assert abs(rho-expected)<1e-12
    assert abs(kappa-(1/rho-1))<1e-12
    assert abs(complement-0.50)<1e-12


def test_unsupported_cell_is_rejected():
    with pytest.raises(ValueError,match="unsupported candidate cell"):
        allocate_joint_dirichlet_group_complement(
            rng=np.random.default_rng(1),
            totals=np.array([10]),
            baseline_probs=np.array([0.3,0.7]),
            eligible_mask=np.array([True,False]),
            player_rhos=np.array([0.2,np.nan]),
            cell="TE_TARGET_SHARE",
        )
