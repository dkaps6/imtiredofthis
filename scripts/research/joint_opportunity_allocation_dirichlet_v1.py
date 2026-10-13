#!/usr/bin/env python3
"""Research-only joint team-conserving opportunity allocator V1.

Implements the frozen Dirichlet eligible-group + complement law from:
docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PLAN.md

No production integration lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

VERSION="PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1"
SUPPORTED_CELLS={"RB_CARRY_SHARE","WR_TARGET_SHARE"}


@dataclass(frozen=True)
class JointAllocationMeta:
    candidate_applied: bool
    eligible_count: int
    complement_probability: float
    rho_room: float
    kappa: float
    reason: str


def beta_intraclass_rho(mu:float, excess_var:float)->float | None:
    if not (np.isfinite(mu) and np.isfinite(excess_var)):
        return None
    if not (0.0<mu<1.0):
        return None
    upper=mu*(1.0-mu)
    if not (0.0<excess_var<upper):
        return None
    rho=excess_var/upper
    if not (0.0<rho<1.0):
        return None
    return float(rho)


def _normalize_probs(probs:np.ndarray)->np.ndarray:
    p=np.asarray(probs,dtype=float).copy()
    if p.ndim!=1 or len(p)<2:
        raise ValueError("baseline probabilities must be a 1D vector with >=2 categories")
    if not np.isfinite(p).all() or (p<0).any():
        raise ValueError("baseline probabilities must be finite and nonnegative")
    s=float(p.sum())
    if s<=0:
        raise ValueError("baseline probabilities sum to zero")
    p/=s
    return p


def room_rho(
    baseline_probs:np.ndarray,
    eligible_mask:np.ndarray,
    player_rhos:np.ndarray,
)->tuple[float,float,float] | None:
    p=_normalize_probs(baseline_probs)
    e=np.asarray(eligible_mask,dtype=bool)
    r=np.asarray(player_rhos,dtype=float)
    if len(e)!=len(p) or len(r)!=len(p):
        raise ValueError("eligible mask / rho length mismatch")
    valid=e&np.isfinite(r)&(r>0)&(r<1)&(p>0)&(p<1)
    if not valid.any():
        return None
    complement=float(p[~valid].sum())
    if not (0.0<complement<1.0):
        return None
    weights=p[valid]*(1.0-p[valid])
    den=float(weights.sum())
    if den<=0:
        return None
    rho=float(np.sum(weights*r[valid])/den)
    if not (0.0<rho<1.0):
        return None
    kappa=1.0/rho-1.0
    if not (np.isfinite(kappa) and kappa>0):
        return None
    return rho,kappa,complement


def _rowwise_multinomial(
    rng:np.random.Generator,
    totals:np.ndarray,
    probs:np.ndarray,
)->np.ndarray:
    """Exact multinomial draws for row-varying probabilities via sequential binomials."""
    n=np.asarray(totals,dtype=int)
    if n.ndim!=1:
        raise ValueError("totals must be 1D")
    p=np.asarray(probs,dtype=float)
    if p.ndim==1:
        p=np.broadcast_to(p,(len(n),len(p))).copy()
    if p.ndim!=2 or p.shape[0]!=len(n):
        raise ValueError("probability matrix shape mismatch")
    if not np.isfinite(p).all() or (p<0).any():
        raise ValueError("probabilities must be finite/nonnegative")
    row_sums=p.sum(axis=1)
    if (row_sums<=0).any():
        raise ValueError("probability row sums must be positive")
    p=p/row_sums[:,None]

    k=p.shape[1]
    out=np.zeros((len(n),k),dtype=int)
    remaining_n=n.copy()
    remaining_prob=np.ones(len(n),dtype=float)
    for j in range(k-1):
        cond=np.divide(
            p[:,j],
            remaining_prob,
            out=np.zeros(len(n),dtype=float),
            where=remaining_prob>0,
        )
        cond=np.clip(cond,0.0,1.0)
        draw=rng.binomial(remaining_n,cond)
        out[:,j]=draw
        remaining_n=remaining_n-draw
        remaining_prob=remaining_prob-p[:,j]
        remaining_prob=np.clip(remaining_prob,0.0,1.0)
    out[:,-1]=remaining_n
    return out


def allocate_baseline_multinomial(
    rng:np.random.Generator,
    totals:np.ndarray,
    baseline_probs:np.ndarray,
)->np.ndarray:
    p=_normalize_probs(baseline_probs)
    return _rowwise_multinomial(rng,np.asarray(totals,dtype=int),p)


def allocate_joint_dirichlet_group_complement(
    *,
    rng:np.random.Generator,
    totals:np.ndarray,
    baseline_probs:np.ndarray,
    eligible_mask:np.ndarray,
    player_rhos:np.ndarray,
    cell:str,
)->tuple[np.ndarray,JointAllocationMeta]:
    if cell not in SUPPORTED_CELLS:
        raise ValueError(f"unsupported candidate cell: {cell}")

    p=_normalize_probs(baseline_probs)
    e=np.asarray(eligible_mask,dtype=bool)
    r=np.asarray(player_rhos,dtype=float)
    if len(e)!=len(p) or len(r)!=len(p):
        raise ValueError("eligible mask / rho length mismatch")

    valid=e&np.isfinite(r)&(r>0)&(r<1)&(p>0)&(p<1)
    rr=room_rho(p,valid,r)
    if rr is None:
        allocations=allocate_baseline_multinomial(rng,totals,p)
        return allocations,JointAllocationMeta(
            candidate_applied=False,
            eligible_count=int(valid.sum()),
            complement_probability=float(p[~valid].sum()) if len(p) else np.nan,
            rho_room=np.nan,
            kappa=np.nan,
            reason="NO_VALID_JOINT_OVERDISPERSION_STATE",
        )

    rho,kappa,complement=rr
    eligible_idx=np.flatnonzero(valid)
    complement_idx=np.flatnonzero(~valid)

    first_probs=np.append(p[eligible_idx],complement)
    alpha=first_probs*kappa
    if not np.isfinite(alpha).all() or (alpha<=0).any():
        allocations=allocate_baseline_multinomial(rng,totals,p)
        return allocations,JointAllocationMeta(
            candidate_applied=False,
            eligible_count=int(len(eligible_idx)),
            complement_probability=complement,
            rho_room=np.nan,
            kappa=np.nan,
            reason="INVALID_DIRICHLET_ALPHA",
        )

    n=np.asarray(totals,dtype=int)
    latent=rng.dirichlet(alpha,size=len(n))
    first_counts=_rowwise_multinomial(rng,n,latent)

    out=np.zeros((len(n),len(p)),dtype=int)
    out[:,eligible_idx]=first_counts[:,:len(eligible_idx)]

    comp_counts=first_counts[:,-1]
    comp_probs=p[complement_idx]/complement
    comp_alloc=_rowwise_multinomial(rng,comp_counts,comp_probs)
    out[:,complement_idx]=comp_alloc

    if not np.array_equal(out.sum(axis=1),n):
        raise RuntimeError("joint allocator violated finite team total")

    return out,JointAllocationMeta(
        candidate_applied=True,
        eligible_count=int(len(eligible_idx)),
        complement_probability=float(complement),
        rho_room=float(rho),
        kappa=float(kappa),
        reason="DIRICHLET_GROUP_COMPLEMENT_APPLIED",
    )


def theoretical_candidate_variance(N:int,q:float,rho_room:float)->float:
    """Marginal count variance under the Dirichlet-Multinomial first stage."""
    return float(N*q*(1.0-q)*(1.0+(N-1)*rho_room))


def expected_counts(N:float,baseline_probs:np.ndarray)->np.ndarray:
    return float(N)*_normalize_probs(baseline_probs)


__all__=[
    "VERSION","SUPPORTED_CELLS","JointAllocationMeta",
    "beta_intraclass_rho","room_rho",
    "allocate_baseline_multinomial",
    "allocate_joint_dirichlet_group_complement",
    "theoretical_candidate_variance","expected_counts",
]
