#!/usr/bin/env python3
"""Synthetic/parity tests for WR Phase 4B mechanics only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.research.evaluate_wr_phase4b_authority_exact_opportunity_attribution_v1 import (
    _identity_status_frame,
    cluster_bootstrap_mean_ci,
    r15_disposition,
    symmetric_product_decomposition,
    symmetric_yard_decomposition,
)


def _assert_close(a, b, tol=1e-10):
    if not math.isclose(float(a), float(b), abs_tol=tol, rel_tol=tol):
        raise AssertionError(f"{a} != {b}")


def test_symmetric_normal():
    opp, eff = symmetric_yard_decomposition(5, 6, 50, 72)
    _assert_close(opp, 11.0)
    _assert_close(eff, 11.0)
    _assert_close(opp + eff, 22.0)


def test_zero_pred_targets():
    opp, eff = symmetric_yard_decomposition(0, 4, 0, 40)
    _assert_close(opp, 40.0)
    _assert_close(eff, 0.0)


def test_zero_actual_targets():
    opp, eff = symmetric_yard_decomposition(4, 0, 40, 0)
    _assert_close(opp, -40.0)
    _assert_close(eff, 0.0)


def test_both_zero_targets():
    opp, eff = symmetric_yard_decomposition(0, 0, 0, 0)
    _assert_close(opp, 0.0)
    _assert_close(eff, 0.0)


def test_invalid_zero_target_nonzero_yards():
    try:
        symmetric_yard_decomposition(0, 2, 1, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("expected predicted-zero consistency failure")
    try:
        symmetric_yard_decomposition(2, 0, 20, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected actual-zero consistency failure")


def test_product_identity():
    mass, share = symmetric_product_decomposition(30, 0.60, 35, 0.50)
    _assert_close(mass + share, 35 * 0.50 - 30 * 0.60)


def test_dispositions():
    assert r15_disposition(-0.06, -0.07, -0.05) == "R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED"
    assert r15_disposition(0.06, 0.02, 0.03) == "R15_WR2PLUS_ALLOCATION_STRUCTURED_ERROR"
    assert r15_disposition(0.06, 0.03, -0.01) == "R15_WR2PLUS_ALLOCATION_MIXED_OR_SMALL"
    assert r15_disposition(-0.06, 0.05, -0.20) == "R15_WR2PLUS_ALLOCATION_MIXED_OR_SMALL"


def test_cluster_bootstrap_deterministic():
    df = pd.DataFrame({
        "season": [2023, 2023, 2023, 2024, 2024, 2024],
        "week": [1, 1, 2, 1, 1, 2],
        "team": ["A", "A", "B", "C", "C", "D"],
        "d": [-0.1, -0.2, 0.0, 0.1, 0.2, 0.0],
    })
    a = cluster_bootstrap_mean_ci(df, "d", reps=500, seed=1234, stratify_season=True)
    b = cluster_bootstrap_mean_ci(df, "d", reps=500, seed=1234, stratify_season=True)
    _assert_close(a["mean"], 0.0)
    _assert_close(a["ci_low"], b["ci_low"])
    _assert_close(a["ci_high"], b["ci_high"])
    assert a["clusters"] == 4


def test_missing_weekly_is_unresolved_not_zero():
    identities = pd.DataFrame({
        "season": [2023, 2023],
        "week": [1, 1],
        "team": ["A", "A"],
        "player_clean_key": ["one", "two"],
    })
    weekly = pd.DataFrame({
        "season": [2023],
        "week": [1],
        "team": ["A"],
        "player_clean_key": ["one"],
        "targets": [0.0],
    })
    out = _identity_status_frame(identities, weekly)
    statuses = dict(zip(out["player_clean_key"], out["weekly_identity_status"]))
    assert statuses["one"] == "EXACT_WEEKLY_MATCH"
    assert statuses["two"] == "UNRESOLVED_WEEKLY_IDENTITY"
    one = out.loc[out["player_clean_key"].eq("one"), "actual_targets_weekly"].iloc[0]
    two = out.loc[out["player_clean_key"].eq("two"), "actual_targets_weekly"].iloc[0]
    _assert_close(one, 0.0)
    assert np.isnan(two), "unresolved identity must not be silently zero-imputed"


def main() -> int:
    tests = [
        test_symmetric_normal,
        test_zero_pred_targets,
        test_zero_actual_targets,
        test_both_zero_targets,
        test_invalid_zero_target_nonzero_yards,
        test_product_identity,
        test_dispositions,
        test_cluster_bootstrap_deterministic,
        test_missing_weekly_is_unresolved_not_zero,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"WR Phase 4B synthetic mechanics: PASS ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
