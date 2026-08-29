#!/usr/bin/env python3
"""Authoritative M71 entrypoint with baseline risk-score alignment fix.

The core M71 design is unchanged. The initial run passed the simple
`qb_ypa_sd8` comparator as a pandas Series into a function that resets the test
index, causing label alignment to turn that one comparator into NaNs. Learned
Ridge/HGB candidates were unaffected because they pass NumPy arrays.

This wrapper converts every risk score to a positional NumPy array before the
frozen evaluator sees it. No feature family, target, model, threshold, gate, or
interpretation rule changes.
"""
from __future__ import annotations

import numpy as np

import scripts.backtest.audit_qb_efficiency_uncertainty as m

_original_quartile_metrics = m.quartile_metrics


def quartile_metrics_positional(test, risk, family, model_name):
    risk_array = np.asarray(risk, dtype=float)
    return _original_quartile_metrics(test, risk_array, family, model_name)


m.quartile_metrics = quartile_metrics_positional


if __name__ == "__main__":
    raise SystemExit(m.main())
