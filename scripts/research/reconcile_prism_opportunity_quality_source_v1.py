#!/usr/bin/env python3
"""Reconcile independent PRISM opportunity-quality source authorities V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VERSION = "PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_V1"
FREEZE_COMMIT = "b3d67fbb1840942dd80e9776f93c33fd230c732b"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
EXPECTED_ROWS = 832

FIELD_MAP = {
    "team_rushes": "team_rush_attempts",
    "team_inside10_rushes": "inside10_rush",
    "team_targets": "team_targets",
    "team_redzone_targets": "red_zone_targets",
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).lower().strip() for c in x.columns]
    return x


def read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile(parent: pd.DataFrame, raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keys = ["season", "week", "team"]
    for x, label in ((parent, "parent"), (raw, "raw")):
        need = set(keys)
        if label == "parent":
            need.update(FIELD_MAP.keys())
        else:
            need.update(FIELD_MAP.values())
        if need - set(x.columns):
            raise RuntimeError(f"{label} missing {sorted(need-set(x.columns))}")
        x["season"] = num(x["season"]).astype("Int64")
        x["week"] = num(x["week"]).astype("Int64")

    p = parent.loc[
        parent["season"].isin(TARGET_SEASONS)
        & parent["week"].isin(TARGET_WEEKS)
    ].copy()
    r = raw.loc[
        raw["season"].isin(TARGET_SEASONS)
        & raw["week"].isin(TARGET_WEEKS)
    ].copy()

    if p.duplicated(keys).any():
        raise RuntimeError("duplicate parent team-game key")
    if r.duplicated(keys).any():
        raise RuntimeError("duplicate raw team-game key")

    pkeep = p[keys + list(FIELD_MAP.keys())].copy()
    rkeep = r[keys + list(FIELD_MAP.values())].copy()
    m = pkeep.merge(
        rkeep,
        on=keys,
        how="outer",
        indicator=True,
        suffixes=("_parent", "_raw"),
        validate="one_to_one",
    )

    left_only = int(m["_merge"].eq("left_only").sum())
    right_only = int(m["_merge"].eq("right_only").sum())
    both = int(m["_merge"].eq("both").sum())

    mismatch = pd.Series(False, index=m.index)
    field_metrics = {}
    for pcol, rcol in FIELD_MAP.items():
        pc = pcol if pcol != rcol else f"{pcol}_parent"
        rc = rcol if pcol != rcol else f"{rcol}_raw"
        pv = num(m[pc])
        rv = num(m[rc])
        delta = pv - rv
        bad = m["_merge"].eq("both") & (
            pv.isna() | rv.isna() | delta.ne(0)
        )
        mismatch |= bad
        field_metrics[pcol] = {
            "raw_field": rcol,
            "max_abs_delta": float(delta.loc[m["_merge"].eq("both")].abs().max())
            if both
            else None,
            "mismatch_rows": int(bad.sum()),
            "parent_sum": float(pv.loc[m["_merge"].eq("both")].sum()) if both else 0.0,
            "raw_sum": float(rv.loc[m["_merge"].eq("both")].sum()) if both else 0.0,
        }
        m[f"delta__{pcol}"] = delta

    key_pass = (
        len(m) == EXPECTED_ROWS
        and both == EXPECTED_ROWS
        and left_only == 0
        and right_only == 0
    )
    field_pass = not bool(mismatch.any())
    result = {
        "rows_union": int(len(m)),
        "overlap_rows": both,
        "left_only_rows": left_only,
        "right_only_rows": right_only,
        "expected_rows": EXPECTED_ROWS,
        "exact_key_parity": bool(key_pass),
        "exact_field_parity": bool(field_pass),
        "mismatch_rows": int(mismatch.sum()),
        "field_metrics": field_metrics,
    }
    diff = m.loc[mismatch | m["_merge"].ne("both")].copy()
    return result, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-team-state", type=Path, required=True)
    ap.add_argument("--raw-teamgames", type=Path, required=True)
    ap.add_argument("--parent-result", type=Path, required=True)
    ap.add_argument("--raw-result", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parent_result = read_json(args.parent_result, "parent source result")
        raw_result = read_json(args.raw_result, "raw source result")
        if parent_result.get("disposition") != "PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL":
            raise RuntimeError("Stage-2 compatibility parent is not source-ready-all")
        if raw_result.get("disposition") != "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY":
            raise RuntimeError("raw finite-pool parent is not source-ready")

        parent = read_csv(args.parent_team_state, "parent team state")
        raw = read_csv(args.raw_teamgames, "raw PBP teamgames")
        metrics, diff = reconcile(parent, raw)

        ready = bool(metrics["exact_key_parity"] and metrics["exact_field_parity"])
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": (
                "PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL"
                if ready
                else "PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_BLOCKED"
            ),
            "parent_stage2_compatibility": {
                "run": 38991429245,
                "artifact": 12085262613,
                "digest": "sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d",
            },
            "parent_raw_finite_pool": {
                "run": 38991429809,
                "artifact": 12084992535,
                "digest": "sha256:753dcc94db548b3be7e4da185d3c72cbc2c766bcf56520d6c99a213dfeff1ee8",
            },
            "metrics": metrics,
            "predictive_candidates_scored": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }
        diff.to_csv(
            args.out_dir / "prism_opportunity_quality_source_reconciliation_differences.csv",
            index=False,
        )
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "predictive_candidates_scored": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }
        (args.out_dir / "prism_opportunity_quality_source_reconciliation_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "prism_opportunity_quality_source_reconciliation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
