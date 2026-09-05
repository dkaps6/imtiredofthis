#!/usr/bin/env python3
"""Validate production RB P3 primitives against authoritative frozen artifacts.

This is a parity test, not a model-selection backtest.  It verifies that the
promoted production constants and deterministic synthesis reproduce the exact
research quantities they are supposed to implement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.rb_rush_synthesis_v1 import apply_p3

EXPECTED_ROWS = 1393
EXPECTED_WEEK1_ROWS = 85
EXPECTED_MAE = 19.949523978340356
EXPECTED_RMSE = 28.866519286368135
EXPECTED_BIAS = -2.496943515423683
EXPECTED_CORR = 0.6312657687946912
ATOL = 1e-10

RB_WEIGHTS = {
    "rush_att": (0.3164919683016017, 0.6528957474344519, 0.030612284263946517),
    "rush_yards": (0.5569542426070742, 0.4430457573929258, 0.0),
}


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _metric(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce").astype(float)
    p = pd.to_numeric(pred, errors="coerce").astype(float)
    if a.isna().any() or p.isna().any():
        raise RuntimeError("parity metric inputs contain missing values")
    e = p - a
    return {
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.square(e).mean())),
        "bias": float(e.mean()),
        "corr": float(np.corrcoef(p, a)[0, 1]),
    }


def _assert_close(name: str, got: float, want: float, atol: float = ATOL) -> None:
    if not np.isclose(float(got), float(want), rtol=0.0, atol=atol):
        raise RuntimeError(f"{name} parity failed: got={got:.15f} want={want:.15f} atol={atol}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack1-root", type=Path, required=True)
    ap.add_argument("--stack3-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/rb_p3_production_parity.csv"))
    args = ap.parse_args()

    stack1 = pd.read_csv(_one(args.stack1_root, "stack1_2025_rb_trace.csv"), low_memory=False)
    stack3 = pd.read_csv(_one(args.stack3_root, "stack3_2025_casebook.csv"), low_memory=False)
    weights = pd.read_csv(args.weights)

    if len(stack3) != EXPECTED_ROWS:
        raise RuntimeError(f"STACK3 row count changed: {len(stack3)} != {EXPECTED_ROWS}")
    if int(pd.to_numeric(stack3["week"], errors="coerce").eq(1).sum()) != EXPECTED_WEEK1_ROWS:
        raise RuntimeError("STACK3 Week-1 row count changed")

    rows: list[dict] = []
    # 1) Verify promoted CSV weights are exactly the frozen STACK1 weights and
    # reproduce the historical full-stack parent row-by-row.
    for market, frozen in RB_WEIGHTS.items():
        wrow = weights.loc[weights["market"].astype(str).eq(market)]
        if len(wrow) != 1:
            raise RuntimeError(f"production ensemble weights must contain exactly one {market} row")
        wrow = wrow.iloc[0]
        promoted = tuple(float(wrow[c]) for c in ("mc_weight", "ml_weight", "state_weight"))
        for i, (got, want) in enumerate(zip(promoted, frozen)):
            _assert_close(f"{market} weight[{i}]", got, want, atol=1e-15)
        _assert_close(f"{market} weight sum", sum(promoted), 1.0, atol=1e-12)

        q = stack1.loc[stack1["market"].astype(str).eq(market)].copy()
        if len(q) != EXPECTED_ROWS:
            raise RuntimeError(f"STACK1 {market} row count changed: {len(q)}")
        calc = (
            promoted[0] * pd.to_numeric(q["mc_proj"], errors="coerce")
            + promoted[1] * pd.to_numeric(q["ml_proj"], errors="coerce")
            + promoted[2] * pd.to_numeric(q["state_proj"], errors="coerce")
        )
        ref = pd.to_numeric(q["ensemble_2024_frozen"], errors="coerce")
        max_diff = float((calc - ref).abs().max())
        if max_diff > ATOL:
            raise RuntimeError(f"{market} STACK1 row-level ensemble parity failed max_diff={max_diff}")
        rows.append({"gate": f"STACK1_{market.upper()}_ENSEMBLE", "n": len(q), "max_abs_diff": max_diff, "passed": 1})

    # 2) Verify exact frozen P3 composition across all 1,393 2025 RB rows.
    context = stack3[["week", "stack_att", "stack_yards", "enriched_att", "m94c_implied_ypc"]].copy()
    out = apply_p3(context)
    p3 = pd.to_numeric(out["rb_synthesis_proj"], errors="coerce")
    ref = pd.to_numeric(stack3["arm_week1_stack"], errors="coerce")
    max_p3_diff = float((p3 - ref).abs().max())
    if max_p3_diff > ATOL:
        raise RuntimeError(f"P3 row-level composition parity failed max_diff={max_p3_diff}")

    week1 = pd.to_numeric(stack3["week"], errors="coerce").eq(1)
    week1_diff = float((p3.loc[week1] - pd.to_numeric(stack3.loc[week1, "stack_yards"], errors="coerce")).abs().max())
    if week1_diff > ATOL:
        raise RuntimeError(f"Week-1 stack override parity failed max_diff={week1_diff}")

    metrics = _metric(stack3["actual_rush_yards"], p3)
    _assert_close("P3 MAE", metrics["mae"], EXPECTED_MAE)
    _assert_close("P3 RMSE", metrics["rmse"], EXPECTED_RMSE)
    _assert_close("P3 bias", metrics["bias"], EXPECTED_BIAS)
    _assert_close("P3 corr", metrics["corr"], EXPECTED_CORR)
    rows.append({"gate": "P3_ALL_2025_ROW_LEVEL", "n": len(stack3), "max_abs_diff": max_p3_diff, "passed": 1})
    rows.append({"gate": "P3_WEEK1_STACK_OVERRIDE", "n": int(week1.sum()), "max_abs_diff": week1_diff, "passed": 1})
    rows.append({"gate": "P3_FROZEN_METRICS", "n": len(stack3), "max_abs_diff": 0.0, "passed": 1, **metrics})

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print("RB P3 PRODUCTION PARITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
