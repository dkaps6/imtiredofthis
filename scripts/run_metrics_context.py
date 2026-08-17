#!/usr/bin/env python3
"""Run make_metrics with season/week resolved from shared runtime context.

The wrapper also injects the authoritative NFL week into props before legacy
make_metrics can fall back to ISO/calendar week inference.
"""
from __future__ import annotations

import sys

import pandas as pd

from scripts.runtime_context import log_runtime_context, resolve_slate_date, resolve_week, resolve_season
import scripts.make_metrics as make_metrics


def _install_props_week_guard(season: int, week: int) -> None:
    original = make_metrics.load_props

    def guarded_load_props():
        props = original()
        if props is None or props.empty:
            return props
        props = props.copy()
        if "season" not in props.columns:
            props["season"] = int(season)
        else:
            props["season"] = pd.to_numeric(props["season"], errors="coerce").fillna(int(season)).astype("Int64")
        # The pipeline is a single-slate build.  Week comes from team_week_map,
        # never from datetime.isocalendar().week.
        props["week"] = int(week)
        return props

    make_metrics.load_props = guarded_load_props


def main() -> int:
    season = resolve_season()
    slate_date = resolve_slate_date()
    week = resolve_week(season=season, slate_date=slate_date)

    make_metrics.WEEK = week
    _install_props_week_guard(season, week)

    log_runtime_context()
    print(f"[run_metrics_context] overriding make_metrics.WEEK -> {week}")
    print("[run_metrics_context] props week guard installed; ISO week inference disabled")
    return make_metrics.cli()


if __name__ == "__main__":
    sys.exit(main())
