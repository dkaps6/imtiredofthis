"""Consume explicit target entitlement through the canonical joint simulator.

This adapter is deliberately narrow.  When ``entitlement_tgt_share`` is present,
it replaces the raw rule target share and temporarily disables the simulator's
internal M38 sharpening because M38 has already been applied by
``target_entitlement_v1``.  The canonical multinomial allocator, efficiency
shocks, rushing logic, passing logic, and TD logic are otherwise untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.simulation_v2 as canonical


def simulate(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    if metrics is None or metrics.empty or "entitlement_tgt_share" not in metrics.columns:
        return canonical.simulate(metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace)
    frame = metrics.copy()
    ent = pd.to_numeric(frame["entitlement_tgt_share"], errors="coerce")
    if ent.isna().any() or not np.isfinite(ent.to_numpy(float)).all() or ent.lt(0).any():
        raise RuntimeError("explicit entitlement simulation received invalid entitlement_tgt_share")
    frame["rules_tgt_share"] = ent.astype(float)

    original = canonical._sharpen_wr_target_shares
    canonical._sharpen_wr_target_shares = lambda team_df, shares: np.asarray(shares, dtype=float)
    try:
        return canonical.simulate(
            frame,
            iterations=iterations,
            seed=seed,
            allocation_trace=allocation_trace,
        )
    finally:
        canonical._sharpen_wr_target_shares = original
