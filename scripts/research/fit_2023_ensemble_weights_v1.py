#!/usr/bin/env python3
"""Fit genuine, held-out ensemble weights for rec_yards/receptions/
rush_rec_yards -- the 3 markets checkpoint 14 found have NO fitted weight
entry in data/model_ensemble_weights.csv, silently falling back to 100%
MC even where ml_proj is measurably more accurate.

Fits ONLY on 2023 (never touching 2024-2025, the seasons those weights will
later be graded against Vegas on) -- the same train/eval separation used
for the QB M89 recipe (train 2023, eval 2024/2025). Fitting and grading on
the same rows would be circular, exactly the leakage trap GPT-5.6 flagged
in Issue #535 checkpoint 8. This is intentionally NOT the in-sample oracle
computed in component_level_diagnostic_v1.py -- that was an upper-bound
diagnostic on the same rows being evaluated; this is a real, blind-tested
weight fit.

Writes a RESEARCH-only weights file -- does not touch
data/model_ensemble_weights.csv. No production/model/weight change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.modeling.ensemble_v2 import fit_market_weights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, required=True, help="component_predictions_2023.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cal = pd.read_csv(a.calibration, low_memory=False)
    cal.columns = [str(c).strip().lower() for c in cal.columns]

    weights = fit_market_weights(cal)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    weights.to_csv(a.out, index=False)
    print(f"[fit_2023_weights] wrote {len(weights)} market weight rows -> {a.out}")
    print(weights.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
