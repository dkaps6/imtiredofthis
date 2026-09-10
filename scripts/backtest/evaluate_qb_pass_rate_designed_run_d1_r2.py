#!/usr/bin/env python3
"""Mechanical Run-2 wrapper for frozen QB designed-run D1.

The immutable parent casebook stores ``pred_C`` and ``pred_S`` with uppercase
factor letters. Run 1 requested lowercase names inside pandas ``usecols``
before the frozen evaluator's normal column normalization. This wrapper changes
only that source-reader seam and delegates every scientific calculation, gate,
formula, threshold, cohort, bootstrap setting, and stopping rule to the frozen
D1 evaluator unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtest import evaluate_qb_pass_rate_designed_run_d1 as base


def repaired_load_chain(root: Path) -> pd.DataFrame:
    source_cols = base.KEYS + [
        "pred_C", "pred_S", "pred_attempts", "actual_attempts", "football_synthesis"
    ]
    x = pd.read_csv(
        base.one(root, "qb_opportunity_chain_casebook.csv"),
        usecols=source_cols,
        low_memory=False,
    )
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = base.canon_keys(x)
    x = x.loc[x["season"].eq(2024)].copy()
    if len(x) != 444 or x.duplicated(base.KEYS).any():
        raise RuntimeError(f"2024 opportunity-chain cohort drift rows={len(x)}")
    for c in ["pred_c", "pred_s", "pred_attempts", "actual_attempts", "football_synthesis"]:
        x[c] = base.num(x[c])
    return x


def main() -> int:
    base.load_chain = repaired_load_chain
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
