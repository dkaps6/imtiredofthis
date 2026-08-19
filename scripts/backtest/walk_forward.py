"""CLI runner for leakage-safe historical component predictions.

The runner expects one explicit pregame-universe CSV per week. Those snapshots
must be created from historical roster/depth information, not target-week box
scores. This keeps participation knowledge out of the feature set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.backtest.component_predictions import append_component_predictions, predict_week


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def run_walk_forward(
    *,
    player_logs_path: Path,
    team_weekly_path: Path,
    schedule_path: Path,
    universe_dir: Path,
    season: int,
    prior_season: int,
    weeks: list[int],
    out_path: Path,
    iterations: int = 5000,
) -> pd.DataFrame:
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "historical team-week features")
    schedule = _read(schedule_path, "historical schedule")
    all_rows = []
    for week in weeks:
        universe_path = universe_dir / f"{season}_week_{week:02d}.csv"
        universe = _read(universe_path, f"pregame universe for {season} week {week}")
        print(f"[backtest] predicting {season} W{week:02d} players={len(universe)}")
        pred = predict_week(
            player_logs=player_logs,
            team_weekly=team_weekly,
            pregame_universe=universe,
            schedule=schedule,
            season=season,
            week=week,
            prior_season=prior_season,
            iterations=iterations,
            seed=42 + week,
        )
        if pred.empty:
            print(f"[backtest] WARN {season} W{week:02d} produced no observed rows")
            continue
        append_component_predictions(pred, out_path)
        all_rows.append(pred)
        print(
            f"[backtest] {season} W{week:02d} rows={len(pred)} "
            f"mc={int(pred['mc_proj'].notna().sum())} "
            f"ml={int(pred['ml_proj'].notna().sum())} "
            f"state={int(pred['state_proj'].notna().sum())}"
        )
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def _parse_weeks(value: str) -> list[int]:
    value = value.strip()
    if not value:
        return list(range(1, 19))
    out = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--player-logs", type=Path, default=Path("data/player_game_logs.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--out", type=Path, default=Path("data/backtests/component_predictions.csv"))
    p.add_argument("--iterations", type=int, default=5000)
    args = p.parse_args()
    run_walk_forward(
        player_logs_path=args.player_logs,
        team_weekly_path=args.team_weekly,
        schedule_path=args.schedule,
        universe_dir=args.universe_dir,
        season=args.season,
        prior_season=args.prior_season,
        weeks=_parse_weeks(args.weeks),
        out_path=args.out,
        iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
