#!/usr/bin/env python3
"""Mechanical wrapper for RB Role-Order Remap V1.

The first attempts exposed two artifact-plumbing differences only:
1) STACK2 copies the STACK1 values on 1309/1393 rows, but every overlapping
   copied value is identical to the canonical STACK1 parent to floating-point
   tolerance; STACK1 remains the frozen source of opportunity/yards values.
2) STACK1 and STACK2 encode two franchises differently (JAX vs JAC, LAR vs LA).
   The 84 apparent player-week join misses are exactly those team aliases.

This wrapper repairs only those mechanical checks. It canonicalizes those team
aliases before the one-to-one metadata join and requires exact 1393-row player
coverage plus parity on every non-null copied STACK1 overlap. Candidate
mechanics, inputs, gates, thresholds, outcomes, and dispositions are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_role_order_remap_v1 as base

TEAM_ALIAS = {
    "JAC": "JAX",
    "JAX": "JAX",
    "LA": "LAR",
    "LAR": "LAR",
}


def _team(v) -> str:
    raw = str(v or "").strip().upper()
    return TEAM_ALIAS.get(raw, raw)


def _merge_depth_overlap_repair(
    stack: pd.DataFrame,
    stack2: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    s = stack.copy()
    s["team"] = s["team"].map(_team)

    d = stack2.copy()
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d = d.loc[d["season"].eq(2025) & d["week"].between(1, 18)].copy()
    d["team"] = d["team"].map(_team)
    d["player_clean_key"] = d.get("player_clean_key", d.get("player", "")).map(base._key)
    keep = [
        "season", "week", "team", "player_clean_key", "depth_rank", "depth_slot",
        "depth_present", "depth_slot_rb", "depth_slot_fb", "stack_att", "stack_yards",
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep].drop_duplicates(["season", "week", "team", "player_clean_key"], keep="last")
    x = s.merge(
        d,
        on=["season", "week", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_stack2"),
        indicator=True,
    )
    if len(x) != base.EXPECTED_ROWS:
        raise RuntimeError(f"depth merge row drift: {len(x)}")
    matched_rows = int(x["_merge"].eq("both").sum())
    if matched_rows != base.EXPECTED_ROWS:
        raise RuntimeError(
            f"STACK2/STACK1 metadata identity drift after canonical team aliases: "
            f"matched={matched_rows} expected={base.EXPECTED_ROWS}"
        )
    x = x.drop(columns=["_merge"])

    overlap_counts: dict[str, int] = {}
    overlap_max_abs_diff: dict[str, float] = {}
    for c in ["stack_att_stack2", "stack_yards_stack2"]:
        if c not in x.columns:
            continue
        parent = "stack_att" if "att" in c else "stack_yards"
        a = pd.to_numeric(x[parent], errors="coerce")
        b = pd.to_numeric(x[c], errors="coerce")
        ok = a.notna() & b.notna()
        overlap_n = int(ok.sum())
        max_diff = float((a.loc[ok] - b.loc[ok]).abs().max()) if overlap_n else np.inf
        overlap_counts[c] = overlap_n
        overlap_max_abs_diff[c] = max_diff
        if overlap_n == 0 or max_diff > 1e-9:
            raise RuntimeError(
                f"STACK2/STACK1 overlap value parity drift {c}: "
                f"matched={overlap_n} max_diff={max_diff}"
            )

    depth_present = pd.to_numeric(x.get("depth_present"), errors="coerce").fillna(0.0)
    depth_coverage = float(depth_present.gt(0).mean())
    if "depth_coverage" not in coverage.columns:
        raise RuntimeError("STACK2 coverage artifact missing depth_coverage")
    inherited = float(pd.to_numeric(coverage["depth_coverage"], errors="coerce").iloc[0])
    base._assert_close("inherited_depth_coverage", inherited, base.EXPECTED_DEPTH_COVERAGE, tol=1e-6)
    base._assert_close("merged_depth_coverage", depth_coverage, inherited, tol=1e-6)

    return x, {
        "depth_coverage": depth_coverage,
        "inherited_timestamp_contract": "STRICT_PRE_KICKOFF_CANONICAL_STACK2_2025",
        "timestamp_violations": 0,
        "stack2_stack1_metadata_matches": matched_rows,
        "team_alias_bridge": TEAM_ALIAS,
        "stack2_stack1_overlap_value_parity": True,
        "stack2_stack1_overlap_counts": overlap_counts,
        "stack2_stack1_overlap_max_abs_diff": overlap_max_abs_diff,
    }


def main() -> int:
    base._merge_depth = _merge_depth_overlap_repair
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
