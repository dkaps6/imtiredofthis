#!/usr/bin/env python3
"""Grade the M89/M90 QB synthesis variants (base/football_synthesis/
market_assisted) against real historical Vegas pass_yards lines.

Closes the missing-adapter gap traced in Issue #535: no committed script ever
bridged run_m89_pregame_synthesis.py's output into
grade_full_stack_vegas_benchmark_v1.py::grade(). This reuses that same,
unmodified grading function three times (once per proj_col), the identical
PLAY/LEAN/STRONG gate used everywhere else in this research thread -- no new
grading rule invented.

Research only. No production, model, weight, or threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade

VARIANTS = {
    "base_proj": "qb_base_summary",
    "football_synthesis": "qb_synthesis_summary",
    "market_assisted": "qb_market_assisted_summary",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True, help="identity-attached QB synthesis trace")
    ap.add_argument("--props", type=Path, required=True, help="m60b_historical_qb_pass_props.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    trace = _read(a.trace, "identity-attached QB synthesis trace")
    props = _read(a.props, "historical QB pass_yards props")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    all_details = []
    for proj_col, out_stem in VARIANTS.items():
        detail, summary = grade(trace, props, proj_col=proj_col)
        detail["variant"] = proj_col
        all_details.append(detail)
        summary.to_csv(a.out_dir / f"{out_stem}.csv", index=False)
        print(f"[qb_synthesis_vegas_grade] {proj_col} -> {out_stem}.csv ({len(detail)} matched rows)")
        if not summary.empty:
            print(summary.to_string(index=False))
        print()

    pd.concat(all_details, ignore_index=True).to_csv(a.out_dir / "qb_synthesis_vegas_detail_all_variants.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
