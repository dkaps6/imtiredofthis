#!/usr/bin/env python3
"""Mechanical execution wrapper for RB-R6.

RB-R6 V1 stamps ``_row_index`` before calling the shared strict-prior participation
helper. That helper also stamps ``_row_index`` from the DataFrame index, producing
duplicate columns and a pandas DataFrame-vs-Series failure during confirmation.

This wrapper changes no scientific inputs, parameters, folds, features, models,
seeds, or gates. It removes only the already-materialized ``_row_index`` column
immediately before the shared helper. The RB frame's index is still the original
baseline row position, so the helper recreates the same intended source-row key.
"""
from __future__ import annotations

from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6

_original_strict_prior = r6._strict_prior_snap_features


def _strict_prior_without_duplicate_row_index(frame, snaps):
    clean = frame.drop(columns=["_row_index"], errors="ignore").copy()
    return _original_strict_prior(clean, snaps)


r6._strict_prior_snap_features = _strict_prior_without_duplicate_row_index


if __name__ == "__main__":
    raise SystemExit(r6.main())
