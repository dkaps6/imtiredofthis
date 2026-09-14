#!/usr/bin/env python3
"""Merge the M89 football-only QB synthesis correction into a full-stack
projection trace, at the same point production applies it.

Production order (scripts/run_pricing_v2.py) is: MC/ML/State -> ensemble ->
(for pass_yards) promoted QB synthesis REPLACES the ensemble mean as the
final football target mean. It is not blended back into another ensemble
pass. So this script must run AFTER build_full_stack_vegas_projection_trace_v2.py
has already produced ``ensemble_proj``, and it overrides that column
directly on matched pass_yards rows -- it does not touch mc_proj/ml_proj/
state_proj, and it must not run before the ensemble step (doing so would
let the ensemble re-blend the QB synthesis mean back down with raw ml_proj/
state_proj, diluting the promoted correction instead of replacing the mean
with it, as production does).

Input:
- ``--projection-file``: the full-stack projection trace (output of
  build_full_stack_vegas_projection_trace_v2.py), already carrying
  ensemble_proj for every market/row, including any upstream WR-R15/TE-R5P
  adjustment already baked into mc_proj before that ensemble step ran.
- ``--qb-trace``: the M89 single-fit synthesis trace (output of
  run_m89_pregame_synthesis.py, trained once on 2023, evaluated on both
  2024 and 2025 -- not the M90 rotating-retrain confirmation, which is a
  different, separately-purposed robustness check with its own
  train=test-1 refit each year). Carries ``football_synthesis`` per
  player/week, reconstructed from the exact frozen M89 feature list and
  Ridge fit -- not a re-derivation.

For every pass_yards row in the projection file whose identity matches a
row in the QB trace, ensemble_proj is overridden with the M89
football_synthesis value. All non-pass_yards rows, and any pass_yards row
without a QB trace match (games the frozen 2023 fit's feature contract
could not cover), are left untouched -- RB rows stay on the generic
ensemble mean (P3 is legitimately out of scope for 2024-2025; see
docs/production/RB_P3_WEEK1_PROMOTION_2026_09_05.md), and any unmatched QB
row fails closed to the existing ensemble mean rather than silently
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


def merge_qb_synthesis(projection: pd.DataFrame, qb_trace: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = projection.copy()
    for c in KEYS:
        if c not in out.columns:
            raise RuntimeError(f"projection trace missing required key column: {c}")
        if c not in qb_trace.columns:
            raise RuntimeError(f"QB M89 trace missing required key column: {c}")
    if "ensemble_proj" not in out.columns:
        raise RuntimeError("projection trace missing ensemble_proj -- run the ensemble step before this merge")

    qb = qb_trace.copy()
    qb["football_synthesis"] = pd.to_numeric(qb["football_synthesis"], errors="coerce")
    qb = qb.dropna(subset=["football_synthesis"])
    dup = qb.duplicated(KEYS, keep=False)
    if dup.any():
        raise RuntimeError(f"QB M89 trace has duplicate identities:\n{qb.loc[dup, KEYS].to_string(index=False)}")
    qb_lookup = qb.set_index(KEYS)["football_synthesis"]

    is_pass = out["market"].astype(str).str.lower().eq("pass_yards")
    key_tuples = list(out[KEYS].itertuples(index=False, name=None))
    matched_mask = [bool(p) and t in qb_lookup.index for p, t in zip(is_pass, key_tuples)]
    matched = pd.Series(matched_mask, index=out.index)

    before = pd.to_numeric(out.loc[matched, "ensemble_proj"], errors="coerce").copy()
    new_values = [qb_lookup.loc[t] for t, m in zip(key_tuples, matched_mask) if m]
    out.loc[matched, "ensemble_proj"] = new_values
    out.loc[matched, "qb_m89_synthesis_applied"] = 1
    out["qb_m89_synthesis_applied"] = pd.to_numeric(out.get("qb_m89_synthesis_applied"), errors="coerce").fillna(0).astype(int)

    stats = {
        "pass_yards_rows": int(is_pass.sum()),
        "pass_yards_rows_matched_to_qb_trace": int(matched.sum()),
        "pass_yards_rows_unmatched_stay_on_generic_ensemble": int(is_pass.sum() - matched.sum()),
        "mean_abs_ensemble_proj_change_on_matched_rows": float(
            np.mean(np.abs(pd.to_numeric(out.loc[matched, "ensemble_proj"], errors="coerce").to_numpy() - before.to_numpy()))
        ) if matched.any() else 0.0,
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", type=Path, required=True, help="full-stack projection trace with ensemble_proj already computed")
    ap.add_argument("--qb-trace", type=Path, required=True, help="M89 single-fit (2023->2024+2025) synthesis trace")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    projection = _read(a.projection_file, "projection trace")
    qb_trace = _read(a.qb_trace, "QB M89 trace")

    merged, stats = merge_qb_synthesis(projection, qb_trace)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out, index=False)
    print(f"[merge_promoted_stack] wrote {len(merged)} rows -> {a.out}")
    print(f"[merge_promoted_stack] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
