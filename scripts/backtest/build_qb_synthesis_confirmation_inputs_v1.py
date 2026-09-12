#!/usr/bin/env python3
"""Build QB pass_yards trace inputs for run_m89_pregame_synthesis.py.

Reuses predict_week() (MC + ML + State + actual join, unchanged) and attaches
the calibrated ensemble projection with the same frozen production weights
(data/model_ensemble_weights.csv) used everywhere else, so the base_proj the
synthesis corrects on top of matches what production actually starts from --
not a bespoke re-derivation.
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


def opt(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
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
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    logs, tw, sched = read(a.player_logs), read(a.team_weekly), read(a.schedule)

    traces = []
    for week in _parse_weeks(a.weeks):
        universe_path = a.universe_dir / f"{a.season}_week_{week:02d}.csv"
        if not universe_path.exists():
            print(f"[qb_synthesis_inputs] {a.season} W{week:02d}: no pregame universe, skipping")
            continue
        universe = read(universe_path)
        try:
            out = predict_week(
                player_logs=logs, team_weekly=tw, pregame_universe=universe, schedule=sched,
                season=a.season, week=week, prior_season=a.prior_season, iterations=a.iterations, seed=53 + week,
            )
        except Exception as exc:
            print(f"[qb_synthesis_inputs] {a.season} W{week:02d}: failed ({exc}); skipping")
            continue
        qb = out.loc[out.market.eq("pass_yards")].copy()
        if qb.empty:
            continue
        qb = apply_ensemble(qb)
        traces.append(qb)
        print(f"[qb_synthesis_inputs] {a.season} W{week:02d}: {len(qb)} QB rows")

    if not traces:
        raise RuntimeError(f"no QB pass_yards rows produced for season {a.season}")

    long = pd.concat(traces, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.out, index=False)
    print(f"[qb_synthesis_inputs] wrote {len(long)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
