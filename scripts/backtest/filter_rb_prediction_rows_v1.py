#!/usr/bin/env python3
"""Deterministically restrict a prediction artifact to RB/FB rows.

Mechanical utility only. It changes no model outputs, features, labels, thresholds,
or science logic. It exists to prevent cross-position rows from entering RB-only
forensics/candidates when consuming a full prediction artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ALLOWED = {"RB", "FB", "HB", "TB"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    a = p.parse_args()

    x = pd.read_csv(a.input, low_memory=False)
    if "position_family" not in x.columns:
        raise RuntimeError("prediction artifact missing position_family; refusing implicit position inference")
    pos = x.position_family.fillna("").astype(str).str.upper().str.strip()
    keep = pos.isin(ALLOWED)
    y = x.loc[keep].copy()
    if y.empty:
        raise RuntimeError("RB/FB filter produced zero rows")

    # Preserve byte-level column schema/order and row order within the selected set.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    y.to_csv(a.output, index=False)

    audit = {
        "input_rows": int(len(x)),
        "output_rows": int(len(y)),
        "removed_non_rb_rows": int((~keep).sum()),
        "input_variants": sorted(x.variant.dropna().astype(str).unique().tolist()) if "variant" in x.columns else [],
        "output_variants": sorted(y.variant.dropna().astype(str).unique().tolist()) if "variant" in y.columns else [],
        "output_position_family_counts": y.position_family.fillna("<NA>").astype(str).value_counts().to_dict(),
        "seasons": sorted(pd.to_numeric(y.get("season"), errors="coerce").dropna().astype(int).unique().tolist()),
        "mechanical_filter_only": True,
        "model_values_modified": 0,
        "science_thresholds_modified": 0,
    }
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
