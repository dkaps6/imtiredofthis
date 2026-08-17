#!/usr/bin/env python3
"""Run make_metrics with season/week resolved from shared runtime context."""

from __future__ import annotations

import sys

from scripts.runtime_context import log_runtime_context, resolve_slate_date, resolve_week, resolve_season
import scripts.make_metrics as make_metrics


def main() -> int:
    season = resolve_season()
    slate_date = resolve_slate_date()
    week = resolve_week(season=season, slate_date=slate_date)

    # Legacy make_metrics.py currently exposes WEEK as a module-level constant.
    # Override it before its CLI executes so all downstream writes/filters use
    # the authoritative schedule-derived week rather than a stale hardcode.
    make_metrics.WEEK = week

    log_runtime_context()
    print(f"[run_metrics_context] overriding make_metrics.WEEK -> {week}")
    return make_metrics.cli()


if __name__ == "__main__":
    sys.exit(main())
