#!/usr/bin/env python3
"""Diagnose why the STRONG/LEAN decision gate over-fires on graded detail rows.

GPT-5.6's checkpoint 22 found the STRONG tier fires on ~87-93% of rows even
under the corrected empirical translator, and that this rate does not vary
meaningfully across component_sd quartiles -- ruling out "some rows are more
miscalibrated than others" as the explanation. This script quantifies the
actual mechanism: grade_full_stack_vegas_benchmark_v1.py's gate feeds a
Normal(proj, component_sd) into the no-vig/EV formulas, where component_sd is
the cross-component disagreement (std of mc_proj/ml_proj/state_proj) -- a
"how much do my three correlated internal estimators disagree" signal, not a
measure of true outcome variance. If component_sd is systematically smaller
than the real empirical residual spread (std of proj - actual), the resulting
Normal is too narrow, p_over/p_under get pushed toward 0/1 for almost any
proj-vs-line gap, and the gate fires on nearly every row -- independent of
whether the football model itself is any good.

This takes an already-graded detail CSV (must carry component_sd, model_error,
and signal columns, e.g. docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv)
and reports, per market and per component_sd quartile, the ratio of median
component_sd to the empirical residual SD (std of model_error). A ratio well
below 1.0 across every quartile -- not just the low end -- is the fingerprint
this script exists to surface.

Research only. Reads already-committed detail data; runs no new simulation,
touches no production/model/weight/threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose(detail: pd.DataFrame, *, quartiles: int = 4) -> pd.DataFrame:
    required = {"market", "component_sd", "model_error", "signal"}
    missing = required - set(detail.columns)
    if missing:
        raise RuntimeError(f"detail frame missing required columns: {sorted(missing)}")

    rows = []
    for market, g in detail.groupby("market"):
        g = g.copy()
        csd = pd.to_numeric(g["component_sd"], errors="coerce")
        me = pd.to_numeric(g["model_error"], errors="coerce")
        valid = csd.notna() & me.notna()
        g = g.loc[valid]
        csd = csd.loc[valid]
        me = me.loc[valid]
        if g.empty:
            continue

        overall_resid_sd = float(me.std())
        overall_strong_rate = float(g["signal"].eq("STRONG_EDGE").mean())
        rows.append({
            "market": market, "quartile": "ALL",
            "n": int(len(g)),
            "component_sd_median": float(csd.median()),
            "empirical_resid_sd": overall_resid_sd,
            "undersize_ratio": float(csd.median() / overall_resid_sd) if overall_resid_sd else np.nan,
            "strong_rate": overall_strong_rate,
        })

        try:
            q = pd.qcut(csd, quartiles, labels=[f"Q{i + 1}" for i in range(quartiles)])
        except ValueError:
            continue
        for label, gg in g.groupby(q, observed=True):
            qcsd = pd.to_numeric(gg["component_sd"], errors="coerce")
            qme = pd.to_numeric(gg["model_error"], errors="coerce")
            resid_sd = float(qme.std())
            rows.append({
                "market": market, "quartile": str(label),
                "n": int(len(gg)),
                "component_sd_median": float(qcsd.median()),
                "empirical_resid_sd": resid_sd,
                "undersize_ratio": float(qcsd.median() / resid_sd) if resid_sd else np.nan,
                "strong_rate": float(gg["signal"].eq("STRONG_EDGE").mean()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no scoreable rows after dropping missing component_sd/model_error")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail", type=Path, required=True, help="graded detail CSV (e.g. *_full_stack_vegas_benchmark_detail.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quartiles", type=int, default=4)
    a = ap.parse_args()

    if not a.detail.exists() or not a.detail.stat().st_size:
        raise RuntimeError(f"missing detail file: {a.detail}")
    detail = pd.read_csv(a.detail, low_memory=False)
    detail.columns = [str(c).strip().lower() for c in detail.columns]

    out = diagnose(detail, quartiles=a.quartiles)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print("=== STRONG-GATE component_sd UNDERSIZE DIAGNOSIS ===")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
