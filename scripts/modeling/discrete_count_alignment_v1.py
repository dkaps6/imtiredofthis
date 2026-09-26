"""Production candidate: preserve discrete support for count-market mean alignment.

Qualified research authority:
- DISCRETE_COUNT_MEAN_ALIGNMENT_V1
- run 36276366140
- artifact 10916528395

This module changes only the representation of already-frozen final means for
`receptions` and `rush_att`. It does not construct a mean, weight, feature,
or sportsbook input.
"""
from __future__ import annotations

import numpy as np

VERSION = "DISCRETE_COUNT_MEAN_ALIGNMENT_V1"
COUNT_MARKETS = frozenset({"receptions", "rush_att"})
ATOL = 1e-12


def _largest_remainder(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 1:
        raise RuntimeError(f"{VERSION} requires a one-dimensional outcome array")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{VERSION} received non-finite aligned count outcomes")
    if (arr < -ATOL).any():
        raise RuntimeError(f"{VERSION} received negative aligned count outcomes")
    arr = np.clip(arr, 0.0, None)

    floors = np.floor(arr).astype(np.int64)
    desired_total = int(np.rint(float(arr.sum())))
    need = desired_total - int(floors.sum())
    if need < 0 or need > len(arr):
        raise RuntimeError(
            f"{VERSION} largest-remainder residual out of bounds "
            f"need={need} draws={len(arr)}"
        )

    out = floors.copy()
    if need:
        frac = arr - floors
        order = np.argsort(-frac, kind="mergesort")
        out[order[:need]] += 1
    return out.astype(float)


def align_outcomes(
    base_outcomes: np.ndarray,
    *,
    market: str,
    mc_proj: float,
    target_mean: float,
) -> tuple[np.ndarray, dict]:
    """Apply exact production mean alignment with V1 discrete count support.

    Non-count markets retain the pre-V1 continuous multiplicative semantics.
    Zero/nonfinite-MC rows retain the existing production no-op behavior.
    """
    base = np.asarray(base_outcomes, dtype=float)
    if base.ndim != 1 or len(base) == 0:
        raise RuntimeError(f"{VERSION} requires a non-empty 1-D outcome array")

    canonical_market = str(market or "").lower().strip()
    mc = float(mc_proj) if np.isfinite(mc_proj) else np.nan
    target = float(target_mean) if np.isfinite(target_mean) else np.nan

    eligible = bool(np.isfinite(mc) and mc > 0 and np.isfinite(target))
    if eligible:
        continuous = base * max(0.0, target / mc)
    else:
        continuous = base.copy()

    applied = int(eligible and canonical_market in COUNT_MARKETS)
    if applied:
        adjusted = _largest_remainder(continuous)
        integer_gap = float(np.max(np.abs(adjusted - np.rint(adjusted))))
        if integer_gap > ATOL or (adjusted < -ATOL).any():
            raise RuntimeError(f"{VERSION} failed integer-support invariant")
        target_gap = abs(float(np.mean(adjusted)) - target)
        max_gap = 0.5 / float(len(adjusted)) + ATOL
        if target_gap > max_gap:
            raise RuntimeError(
                f"{VERSION} target-mean gap {target_gap} exceeds {max_gap}"
            )
        frac_rate = float(np.mean(np.abs(continuous - np.rint(continuous)) > ATOL))
        meta = {
            "discrete_count_alignment_applied": 1,
            "discrete_count_alignment_version": VERSION,
            "discrete_count_alignment_pre_fractional_rate": frac_rate,
            "discrete_count_alignment_post_integer_max_gap": integer_gap,
            "discrete_count_alignment_target_mean_gap": target_gap,
        }
        return adjusted, meta

    # Non-count markets and production-ineligible zero/nonfinite-MC rows preserve
    # exact pre-V1 semantics.
    meta = {
        "discrete_count_alignment_applied": 0,
        "discrete_count_alignment_version": "",
        "discrete_count_alignment_pre_fractional_rate": np.nan,
        "discrete_count_alignment_post_integer_max_gap": np.nan,
        "discrete_count_alignment_target_mean_gap": np.nan,
    }
    return continuous, meta
