#!/usr/bin/env python3
"""Finalize Migration 60B market outputs with source-specific provenance.

Migration 60's numerical grader is reused unchanged. This postprocessor enforces
M60B's stricter claim rule: 2024, 2025, and the combined sample must each meet
the precommitted coverage threshold before combined benchmark claims are
supported. It also removes the grader's unlabeled ``m60_*`` market files after
creating provenance-labeled ``m60b_*`` replacements so closing-like archive
results cannot be mistaken for the original exact-30-minute M60 definition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "gcampb41/nfl_data- Action Network-derived archive"
LINE_DEFINITION = "archived_latest_per_book_closing_like_not_exact_30min"


def season_value(v) -> str:
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return s


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
    cov = pd.to_numeric(c.get("coverage"), errors="coerce")
    c["min_coverage_required"] = float(args.min_coverage)

    # The PR contract requires both development seasons individually to clear
    # the threshold before any combined claim is supported.
    all_seasons_supported = True
    if "season" in c.columns:
        labels = c["season"].map(season_value)
        season_rows = labels.isin({"2024", "2025"})
        present = set(labels.loc[season_rows])
        all_seasons_supported = present == {"2024", "2025"} and bool(
            cov.loc[season_rows].ge(float(args.min_coverage)).all()
        )
        combined_rows = ~season_rows
    else:
        # Without explicit season rows we cannot prove the precommitted
        # per-season coverage rule, so claims remain exploratory.
        all_seasons_supported = False
        combined_rows = pd.Series(True, index=c.index)

    local_supported = cov.ge(float(args.min_coverage))
    supported = local_supported.copy()
    supported.loc[combined_rows] = supported.loc[combined_rows] & all_seasons_supported
    c["all_required_seasons_supported"] = bool(all_seasons_supported)
    c["claim_status"] = np.where(
        supported,
        "benchmark_claims_supported",
        "exploratory_only_low_or_incomplete_season_coverage",
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

    # Prevent closing-like archive outputs from being published under M60's
    # unlabeled exact-snapshot filenames.
    for src_name in ["m60_market_coverage.csv", *mapping.keys()]:
        src = d / src_name
        if src.exists():
            src.unlink()

    print("=== M60B CLAIM SUPPORT ===")
    print(c.to_string(index=False))
    print(f"[m60b finalize] all_required_seasons_supported={all_seasons_supported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
