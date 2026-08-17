#!/usr/bin/env python3
"""Run the deterministic metrics v2 builder under shared runtime context."""
from __future__ import annotations

import argparse

from scripts.metrics_v2 import build
from scripts.runtime_context import log_runtime_context, resolve_season, resolve_slate_date, resolve_week


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--mode", default="full")  # compatibility only
    args = parser.parse_args()

    season = int(args.season if args.season is not None else resolve_season())
    slate = (args.date if args.date is not None else resolve_slate_date()) or ""
    week = int(args.week if args.week is not None else resolve_week(season=season, slate_date=slate))
    log_runtime_context()
    metrics = build(season, week)
    from pathlib import Path
    out = Path("data/metrics_ready.csv")
    export = Path("data/make_metrics_output.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    metrics.to_csv(export, index=False)
    print(f"[run_metrics_context] metrics_v2 rows={len(metrics)} season={season} week={week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
