#!/usr/bin/env python3
"""Build full-stack (MC+ML+State+ensemble) historical projections, all markets.

Companion to build_qb_synthesis_confirmation_inputs_v1.py for the markets
that have no position-specific frozen overlay applicable outside their
qualified scope (RB P3/R26 are qualified for the 2026 Week-1 route only, so
they are correctly absent here; QB pass_yards should instead go through the
QB script so the M89/M90 synthesis can be layered on afterward).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.component_predictions import predict_week
from scripts.backtest.walk_forward import _parse_weeks
from scripts.modeling.ensemble_v2 import apply_ensemble


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--prior-season", type=int, required=True)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, required=True)
    p.add_argument("--team-weekly", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--universe-dir", type=Path, required=True)
    p.add_argument("--exclude-markets", default="pass_yards", help="comma-separated markets to skip (QB pass_yards has its own script)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    logs, tw, sched = read(a.player_logs), read(a.team_weekly), read(a.schedule)
    exclude = {m.strip() for m in a.exclude_markets.split(",") if m.strip()}

    traces = []
    for week in _parse_weeks(a.weeks):
        universe_path = a.universe_dir / f"{a.season}_week_{week:02d}.csv"
        if not universe_path.exists():
            print(f"[full_stack_market] {a.season} W{week:02d}: no pregame universe, skipping")
            continue
        universe = read(universe_path)
        try:
            out = predict_week(
                player_logs=logs, team_weekly=tw, pregame_universe=universe, schedule=sched,
                season=a.season, week=week, prior_season=a.prior_season, iterations=a.iterations, seed=53 + week,
            )
        except Exception as exc:
            print(f"[full_stack_market] {a.season} W{week:02d}: failed ({exc}); skipping")
            continue
        rows = out.loc[~out.market.isin(exclude)].copy()
        if rows.empty:
            continue
        rows = apply_ensemble(rows)
        traces.append(rows)
        print(f"[full_stack_market] {a.season} W{week:02d}: {len(rows)} rows")

    if not traces:
        raise RuntimeError(f"no rows produced for season {a.season}")

    long = pd.concat(traces, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.out, index=False)
    print(f"[full_stack_market] wrote {len(long)} rows -> {a.out}")
    print(long.groupby("market").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
