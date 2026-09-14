#!/usr/bin/env python3
"""Merge the M89 football-only QB synthesis correction into a full-stack
projection trace, at the same point production applies it.

Production order (scripts/run_pricing_v2.py) is: MC/ML/State -> ensemble ->
(for pass_yards) promoted QB synthesis REPLACES the ensemble mean as the
final football target mean. It is not blended back into another ensemble
pass. So this script must run AFTER build_full_stack_vegas_projection_trace_v2.py
has already produced ``ensemble_proj``, and it overrides that column
directly on matched pass_yards rows -- it does not touch mc_proj/ml_proj/
state_proj.

For this historical *promoted-stack* benchmark, pass-yards rows without
historical M89 authority are excluded rather than mixed into the QB grade as
generic-ensemble controls. Every finite M89 authority row must be present in
the projection universe; missing authority rows fail closed. Non-QB markets
are preserved unchanged.

This script performs no modeling of its own -- every value it writes was
produced by an already-frozen, already-validated production or research
component. It only decides which already-computed number wins per row and
which QB rows belong to the historically authorized benchmark cohort.
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
    if "market" not in out.columns:
        raise RuntimeError("projection trace missing market")

    qb = qb_trace.copy()
    qb["football_synthesis"] = pd.to_numeric(qb["football_synthesis"], errors="coerce")
    qb = qb.dropna(subset=["football_synthesis"])
    dup = qb.duplicated(KEYS, keep=False)
    if dup.any():
        raise RuntimeError(f"QB M89 trace has duplicate identities:\n{qb.loc[dup, KEYS].to_string(index=False)}")
    qb_lookup = qb.set_index(KEYS)["football_synthesis"]

    is_pass = out["market"].astype(str).str.lower().eq("pass_yards")
    pass_rows = out.loc[is_pass].copy()
    proj_dup = pass_rows.duplicated(KEYS, keep=False)
    if proj_dup.any():
        raise RuntimeError(
            "projection trace has duplicate pass_yards identities:\n"
            + pass_rows.loc[proj_dup, KEYS].head(20).to_string(index=False)
        )

    pass_key_set = set(pass_rows[KEYS].itertuples(index=False, name=None))
    qb_key_set = set(qb_lookup.index.tolist())
    missing_authority = sorted(qb_key_set - pass_key_set)
    if missing_authority:
        raise RuntimeError(
            f"QB M89 authority rows missing from pass_yards projection universe: "
            f"count={len(missing_authority)} sample={missing_authority[:10]}"
        )

    key_tuples = list(out[KEYS].itertuples(index=False, name=None))
    matched_mask = [bool(p) and t in qb_lookup.index for p, t in zip(is_pass, key_tuples)]
    matched = pd.Series(matched_mask, index=out.index)
    unmatched_pass = is_pass & ~matched

    before = pd.to_numeric(out.loc[matched, "ensemble_proj"], errors="coerce").copy()
    new_values = [qb_lookup.loc[t] for t, m in zip(key_tuples, matched_mask) if m]
    out.loc[matched, "ensemble_proj"] = new_values
    out.loc[matched, "qb_m89_synthesis_applied"] = 1
    out["qb_m89_synthesis_applied"] = pd.to_numeric(
        out.get("qb_m89_synthesis_applied"), errors="coerce"
    ).fillna(0).astype(int)

    input_rows = int(len(out))
    excluded = int(unmatched_pass.sum())
    changed_mean = float(
        np.mean(
            np.abs(
                pd.to_numeric(out.loc[matched, "ensemble_proj"], errors="coerce").to_numpy()
                - before.to_numpy()
            )
        )
    ) if matched.any() else 0.0

    # The unified QB benchmark is the historically authorized M89 cohort only.
    # Base-ensemble QB rows outside that cohort are controls, not promoted-stack
    # observations, so exclude them before any sportsbook archive join/grade.
    out = out.loc[~unmatched_pass].copy().reset_index(drop=True)

    stats = {
        "input_projection_rows": input_rows,
        "output_projection_rows": int(len(out)),
        "qb_trace_rows_with_finite_synthesis": int(len(qb_lookup)),
        "qb_trace_rows_missing_from_projection": 0,
        "pass_yards_rows_input": int(is_pass.sum()),
        "pass_yards_rows_m89_authorized": int(matched.sum()),
        "pass_yards_rows_excluded_no_m89_authority": excluded,
        "mean_abs_ensemble_proj_change_on_m89_rows": changed_mean,
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
