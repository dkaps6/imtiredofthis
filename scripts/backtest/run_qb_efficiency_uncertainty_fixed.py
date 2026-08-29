#!/usr/bin/env python3
"""Authoritative M71 entrypoint with two mechanical evaluation fixes.

The frozen M71 scientific design is unchanged.

1. Baseline risk-score alignment:
   The initial run passed the simple `qb_ypa_sd8` comparator as a pandas Series
   into an evaluator that resets the test index. Converting risk scores to a
   positional NumPy array prevents label reindexing from turning that one
   comparator into NaNs. Learned Ridge/HGB predictions were already arrays.

2. Structural-control alias deduplication:
   In the immutable canonical snapshot, `pred_pass_yards` is an alias of
   `m64_pass_raw_reference`, and `pred_attempts` is an alias of `attempts_raw`.
   Counting both gives Raw double weight in standard-deviation disagreement.
   The authoritative wrapper recomputes structural dispersion from distinct
   model outputs only. Ranges are recomputed for the same explicit set even
   though duplicate aliases do not mathematically change a range.

No feature family, target, model, threshold, gate, interpretation rule, or
production behavior changes.
"""
from __future__ import annotations

import numpy as np

import scripts.backtest.audit_qb_efficiency_uncertainty as m

_original_quartile_metrics = m.quartile_metrics
_original_add_canonical = m.add_canonical_pregame_features


def quartile_metrics_positional(test, risk, family, model_name):
    risk_array = np.asarray(risk, dtype=float)
    return _original_quartile_metrics(test, risk_array, family, model_name)


def add_canonical_pregame_features_distinct(base):
    x = _original_add_canonical(base)

    # Frozen canonical aliases:
    #   pred_pass_yards == m64_pass_raw_reference
    #   pred_attempts   == attempts_raw
    # Use each underlying forecast once when measuring model disagreement.
    pass_cols = [
        "m64_pass_raw_reference",
        "m64_pass_generative_neutral",
        "m64_pass_generative_gamescript",
        "m65_pass_state_ridge",
    ]
    att_cols = [
        "attempts_raw",
        "m64_attempts_generative_neutral",
        "m64_attempts_generative_gamescript",
        "m65_attempts_state_ridge",
    ]

    x["model_pass_prediction_sd"] = [m.row_sd(r, pass_cols) for _, r in x.iterrows()]
    x["model_pass_prediction_range"] = [m.row_range(r, pass_cols) for _, r in x.iterrows()]
    x["model_attempt_prediction_sd"] = [m.row_sd(r, att_cols) for _, r in x.iterrows()]
    x["model_attempt_prediction_range"] = [m.row_range(r, att_cols) for _, r in x.iterrows()]
    return x


m.quartile_metrics = quartile_metrics_positional
m.add_canonical_pregame_features = add_canonical_pregame_features_distinct


if __name__ == "__main__":
    raise SystemExit(m.main())
