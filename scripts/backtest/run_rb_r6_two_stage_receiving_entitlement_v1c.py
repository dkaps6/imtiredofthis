#!/usr/bin/env python3
"""Second mechanical execution wrapper for frozen RB-R6 science.

R6B proved that removing the duplicate ``_row_index`` column was insufficient:
the shared participation helper then recreated the RB-subset index rather than the
full baseline row label.  This wrapper preserves the already-stamped baseline row
key by value, invokes the unchanged shared helper, and restores that key afterward.

No scientific feature, fit, fold, threshold, seed, target, or allocation rule is
changed.  This is row-identity plumbing only.
"""
from __future__ import annotations

import numpy as np

from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6

_original_strict_prior = r6._strict_prior_snap_features


def _strict_prior_preserve_baseline_row(frame, snaps):
    if "_row_index" not in frame.columns:
        return _original_strict_prior(frame, snaps)
    baseline_row = frame["_row_index"].to_numpy(copy=True)
    clean = frame.drop(columns=["_row_index"]).copy()
    out, future = _original_strict_prior(clean, snaps)
    if len(out) != len(baseline_row):
        raise RuntimeError(
            f"RB-R6C participation helper changed row count: {len(baseline_row)} -> {len(out)}"
        )
    out["_row_index"] = baseline_row
    if not np.isfinite(out["_row_index"].astype(float)).all():
        raise RuntimeError("RB-R6C restored baseline row keys are not finite")
    return out, future


r6._strict_prior_snap_features = _strict_prior_preserve_baseline_row


if __name__ == "__main__":
    raise SystemExit(r6.main())
