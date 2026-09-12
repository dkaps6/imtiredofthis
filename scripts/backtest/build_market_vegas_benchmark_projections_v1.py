#!/usr/bin/env python3
"""Build current-production historical projections for the market Vegas benchmark.

This generalizes Migration 60's QB-only benchmark-projection builder
(``build_qb_vegas_benchmark_mc.py``) to every market the canonical component
stack already produces per player-week: pass_yards, rush_yards, rec_yards,
receptions, rush_att, rush_rec_yards. It runs the single current production
candidate (no diagnostic raw/joint-cap-shrink variants) so the output is a
fair comparison of the actual deployed model, not a research artifact.

Measurement only: builds independent football projections first, before any
sportsbook line is loaded. Output is safe to join to historical prop lines
afterward without the market ever influencing the projection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity
from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks


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
    p.add_argument("--injuries", type=Path, default=None)
    p.add_argument("--weather", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    logs, tw, sched = read(a.player_logs), read(a.team_weekly), read(a.schedule)
    inj, weather = opt(a.injuries), opt(a.weather)

    traces = []
    for week in _parse_weeks(a.weeks):
        universe_path = a.universe_dir / f"{a.season}_week_{week:02d}.csv"
        if not universe_path.exists():
            print(f"[market_vegas_benchmark] {a.season} W{week:02d}: no pregame universe, skipping")
            continue
        universe = read(universe_path)
        try:
            bundle = build_historical_context_bundle(
                player_logs=logs,
                team_weekly=tw,
                pregame_universe=universe,
                schedule=sched,
                season=a.season,
                week=week,
                prior_season=a.prior_season,
                injuries=_exact_week(inj, a.season, week),
                weather=_exact_week(weather, a.season, week),
            )
        except Exception as exc:
            print(f"[market_vegas_benchmark] {a.season} W{week:02d}: bundle failed ({exc}); skipping")
            continue

        mc = build_mc_predictions(bundle, iterations=a.iterations, seed=53 + week)
        actual = build_actual_rows(logs, a.season, week)
        if actual.empty:
            print(f"[market_vegas_benchmark] {a.season} W{week:02d}: no actual results yet, skipping")
            continue

        z = mc.merge(actual, on=["team", "player_clean_key", "market"], how="inner")
        z["season"] = a.season
        z["week"] = week
        keep = [c for c in ["season", "week", "team", "opponent", "player_clean_key", "market", "mc_proj", "actual"] if c in z.columns]
        traces.append(z[keep])
        print(f"[market_vegas_benchmark] {a.season} W{week:02d}: {len(z)} projected rows")

    if not traces:
        raise RuntimeError(f"no market Vegas benchmark rows produced for season {a.season}")

    long = pd.concat(traces, ignore_index=True)

    sk = sched.copy()
    sk.columns = [str(c).strip().lower() for c in sk.columns]
    sk = sk.loc[pd.to_numeric(sk.season, errors="coerce").eq(a.season)] if "season" in sk.columns else sk
    keep_sk = [c for c in ["week", "team", "opponent", "game_id"] if c in sk.columns]
    sk = sk[keep_sk].drop_duplicates(["week", "team"])
    if "opponent" in long.columns and "opponent" in sk.columns:
        sk = sk.rename(columns={"opponent": "schedule_opponent"})
    long = long.merge(sk, on=["week", "team"], how="left", validate="many_to_one")
    if "schedule_opponent" in long.columns:
        mismatch = long["opponent"].astype(str).ne(long["schedule_opponent"].astype(str))
        if mismatch.any():
            sample = long.loc[mismatch, ["season", "week", "team", "opponent", "schedule_opponent", "game_id"]].head(20).to_dict(orient="records")
            raise RuntimeError(f"benchmark component/schedule opponent mismatch: {sample}")
        long = long.drop(columns=["schedule_opponent"])

    assert_benchmark_identity(
        long,
        label=f"market Vegas benchmark projections {a.season}",
        require_team=True,
        require_opponent=("opponent" in long.columns),
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(a.out, index=False)
    print(f"[market_vegas_benchmark] wrote {len(long)} rows -> {a.out}")
    print(long.groupby("market").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())