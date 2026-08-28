#!/usr/bin/env python3
"""Finalize Migration 60B market outputs with source-specific provenance.

Migration 60's numerical grader is reused unchanged. This postprocessor fixes
claim support at the season level and renames/copies outputs to M60B-specific
files while explicitly labeling the free archive's non-timestamped market
snapshot definition.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

SOURCE = "gcampb41/nfl_data- Action Network-derived archive"
LINE_DEFINITION = "archived_latest_per_book_closing_like_not_exact_30min"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    args = ap.parse_args()
    d = args.summary_dir

    coverage_path = d / "m60_market_coverage.csv"
    if not coverage_path.exists():
        raise RuntimeError("M60 grader did not produce market coverage")
    c = pd.read_csv(coverage_path)
    c["min_coverage_required"] = float(args.min_coverage)
    c["claim_status"] = np.where(
        pd.to_numeric(c.coverage, errors="coerce").ge(float(args.min_coverage)),
        "benchmark_claims_supported",
        "exploratory_only_low_coverage",
    )
    c["market_source"] = SOURCE
    c["source_line_definition"] = LINE_DEFINITION
    c["exact_30min_snapshot"] = False
    c.to_csv(d / "m60b_market_coverage.csv", index=False)

    mapping = {
        "m60_football_and_market_metrics.csv": "m60b_football_and_market_metrics.csv",
        "m60_market_edge_buckets.csv": "m60b_market_edge_buckets.csv",
        "m60_market_edge_thresholds.csv": "m60b_market_edge_thresholds.csv",
        "m60_catastrophic_market_diagnostics.csv": "m60b_catastrophic_market_diagnostics.csv",
        "m60_market_game_detail.csv": "m60b_market_game_detail.csv",
    }
    for src_name, dst_name in mapping.items():
        src = d / src_name
        if not src.exists():
            continue
        x = pd.read_csv(src)
        x["market_source"] = SOURCE
        x["source_line_definition"] = LINE_DEFINITION
        x["exact_30min_snapshot"] = False
        x.to_csv(d / dst_name, index=False)

    print("=== M60B CLAIM SUPPORT ===")
    print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
