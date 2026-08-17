#!/usr/bin/env python3
"""Compatibility wrapper for the season-aware QB run feature builder."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build.pbp_features import build_qb_run_metrics
from scripts.utils.pbp import get_pbp


def main(season: int | None = None) -> None:
    if season is None:
        import os
        season = int(os.getenv("SEASON", "2026"))
    pbp = get_pbp(int(season), min_rows=1)
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")]
        if not reg.empty:
            pbp = reg.copy()
    a, b, combined = build_qb_run_metrics(pbp)
    data = Path("data")
    data.mkdir(parents=True, exist_ok=True)
    a.to_csv(data / "qb_scramble_rates.csv", index=False)
    b.to_csv(data / "qb_designed_runs.csv", index=False)
    combined.to_csv(data / "qb_run_metrics.csv", index=False)
    print(f"[qb_run_metrics] season={season} scramble={len(a)} designed={len(b)} combined={len(combined)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    main(args.season)
