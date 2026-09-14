#!/usr/bin/env python3
"""Merge the WR-R15/TE-R5P-adjusted component predictions with the M90 QB
football-only synthesis correction into one unified component-predictions
file, so a single downstream ensemble/grade pass reflects every currently
promoted position overlay at once instead of any one in isolation.

Inputs:
- ``--component-file``: the WR-R15/TE-R5P-adjusted component predictions for
  one season (output of persist_wr_te_production_order_historical_v1.py's
  ``--component-out``). Already has mc_proj overridden on authorized WR/TE
  receiving rows; everything else is byte-identical to the plain base
  ensemble component trace.
- ``--qb-trace``: the M90 rotated confirmation trace for the same test
  season (output of run_m90_qb_synthesis_confirmation.py). Has the
  football-only-synthesis-corrected pass_yards projection
  (``football_synthesis``) per player/week, reconstructed from the exact
  frozen M89/M90 feature list and Ridge fit -- not a re-derivation.

For every pass_yards row in the component file whose identity matches a row
in the QB trace, mc_proj is overridden with the M90 football_synthesis value
(mirroring how run_pricing_v2.py treats promoted QB synthesis as replacing
the target mean production prices from). All non-pass_yards rows, and any
pass_yards row without a QB trace match, are left untouched -- RB rows stay
on the generic ensemble mean (P3 is legitimately out of scope for 2024-2025;
see docs/production/RB_P3_WEEK1_PROMOTION_2026_09_05.md), and any unmatched
QB row fails closed to the existing ensemble mean rather than silently
guessing.

This script performs no modeling of its own -- every value it writes was
produced by an already-frozen, already-validated production or research
component. It only decides which already-computed number wins per row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["season", "week", "team", "opponent", "player_clean_key"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"{label} has 0 rows: {path}")
    return out


def merge_qb_synthesis(component: pd.DataFrame, qb_trace: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = component.copy()
    for c in KEYS:
        if c not in out.columns:
            raise RuntimeError(f"component predictions missing required key column: {c}")
        if c not in qb_trace.columns:
            raise RuntimeError(f"QB M90 trace missing required key column: {c}")

    qb = qb_trace.copy()
    qb["football_synthesis"] = pd.to_numeric(qb["football_synthesis"], errors="coerce")
    qb = qb.dropna(subset=["football_synthesis"])
    dup = qb.duplicated(KEYS, keep=False)
    if dup.any():
        raise RuntimeError(f"QB M90 trace has duplicate identities:\n{qb.loc[dup, KEYS].to_string(index=False)}")
    qb_lookup = qb.set_index(KEYS)["football_synthesis"]

    is_pass = out["market"].astype(str).str.lower().eq("pass_yards")
    key_tuples = list(out[KEYS].itertuples(index=False, name=None))
    matched_mask = [bool(p) and t in qb_lookup.index for p, t in zip(is_pass, key_tuples)]
    matched = pd.Series(matched_mask, index=out.index)

    before = pd.to_numeric(out.loc[matched, "mc_proj"], errors="coerce").copy()
    new_values = [qb_lookup.loc[t] for t, m in zip(key_tuples, matched_mask) if m]
    out.loc[matched, "mc_proj"] = new_values
    out.loc[matched, "qb_m90_synthesis_applied"] = 1
    out["qb_m90_synthesis_applied"] = pd.to_numeric(out.get("qb_m90_synthesis_applied"), errors="coerce").fillna(0).astype(int)

    stats = {
        "pass_yards_rows": int(is_pass.sum()),
        "pass_yards_rows_matched_to_qb_trace": int(matched.sum()),
        "pass_yards_rows_unmatched_stay_on_generic_ensemble": int(is_pass.sum() - matched.sum()),
        "mean_abs_mc_proj_change_on_matched_rows": float(
            np.mean(np.abs(pd.to_numeric(out.loc[matched, "mc_proj"], errors="coerce").to_numpy() - before.to_numpy()))
        ) if matched.any() else 0.0,
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component-file", type=Path, required=True, help="WR-R15/TE-R5P-adjusted component predictions (one season)")
    ap.add_argument("--qb-trace", type=Path, required=True, help="M90 rotated confirmation trace (matching test season)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    component = _read(a.component_file, "component predictions")
    qb_trace = _read(a.qb_trace, "QB M90 trace")

    merged, stats = merge_qb_synthesis(component, qb_trace)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print(f"[merge_promoted_stack] wrote {len(merged)} rows -> {a.out}")
    print(f"[merge_promoted_stack] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
