#!/usr/bin/env python3
"""Migration 28: resolve the canonical rushing-output contradiction with a keyed trace.

Migration 27 used an external wrapper around `_allocate_counts`. Because the
canonical `mc_proj` should mathematically equal the mean of the exact carry array
stored in SimulationResult, a full-slate mismatch indicates the external trace
may have attributed allocation calls to the wrong player/team context.

This diagnostic uses the opt-in allocation trace emitted from *inside* the
canonical simulator at the exact point the carry array is created and keyed.
Production behavior is unchanged when tracing is disabled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
import scripts.simulation_v2 as simv2


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _stage_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "canonical_mc_proj",
        "keyed_lookup_mean",
        "keyed_realized_mean",
        "keyed_expected_from_probability",
    ]:
        g = frame.loc[
            pd.to_numeric(frame[col], errors="coerce").notna()
            & pd.to_numeric(frame["actual"], errors="coerce").notna()
        ].copy()
        if g.empty:
            continue
        pred = pd.to_numeric(g[col], errors="coerce")
        actual = pd.to_numeric(g["actual"], errors="coerce")
        err = pred - actual
        rows.append({
            "stage": col,
            "n": len(g),
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "bias": float(err.mean()),
            "correlation": float(pred.corr(actual)) if pred.nunique() > 1 and actual.nunique() > 1 else np.nan,
            "pred_mean": float(pred.mean()),
            "actual_mean": float(actual.mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/keyed_rushing_trace"))
    a = p.parse_args()

    logs = _read(a.player_logs, "player logs")
    team = _read(a.team_weekly, "team weekly")
    sched = _read(a.schedule, "schedule")
    injuries = _optional(a.injuries)
    weather = _optional(a.weather)
    all_rows = []

    for week in _parse_weeks(a.weeks):
        universe = _read(a.universe_dir / f"{a.season}_week_{week:02d}.csv", f"universe W{week}")
        bundle = build_historical_context_bundle(
            player_logs=logs,
            team_weekly=team,
            pregame_universe=universe,
            schedule=sched,
            season=a.season,
            week=week,
            prior_season=a.prior_season,
            injuries=_exact_week(injuries, a.season, week),
            weather=_exact_week(weather, a.season, week),
        )
        seed = 42 + week

        # First run: exact canonical build, producing the mc_proj currently evaluated.
        prepared = cp.build_mc_predictions(bundle, iterations=a.iterations, seed=seed)
        rush = prepared.loc[prepared.market.eq("rush_att")].copy()
        rush = rush[["event_id", "team", "player_clean_key", "player", "position", "mc_proj"]].drop_duplicates(
            ["event_id", "team", "player_clean_key"]
        )
        rush = rush.rename(columns={"mc_proj": "canonical_mc_proj"})

        # Second run: same prepared metrics, same seed, with a keyed trace emitted
        # from inside simulation_v2 immediately after carries are allocated.
        allocation_trace: list[dict] = []
        keyed_result = simv2.simulate(
            prepared,
            iterations=a.iterations,
            seed=seed,
            allocation_trace=allocation_trace,
        )
        keyed = pd.DataFrame(allocation_trace)
        if keyed.empty:
            raise RuntimeError(f"W{week} keyed allocation trace is empty")

        keyed_lookup = []
        for _, row in rush.iterrows():
            arr = simv2.lookup(keyed_result, row, "rush_att")
            keyed_lookup.append(float(np.mean(arr)) if arr is not None and len(arr) else np.nan)
        rush["keyed_lookup_mean"] = keyed_lookup

        keyed = keyed.rename(columns={
            "realized_multinomial_mean_carries": "keyed_realized_mean",
            "expected_carries_from_final_probability": "keyed_expected_from_probability",
        })
        keep = [
            "event_id", "team", "player_clean_key", "sim_selected_market",
            "raw_player_rush_share", "raw_team_rush_share_sum",
            "final_player_probability", "residual_probability",
            "team_rush_total_mean", "keyed_expected_from_probability",
            "keyed_realized_mean",
        ]
        trace = rush.merge(keyed[keep], on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")

        actual = cp.build_actual_rows(logs, a.season, week)
        actual = actual.loc[actual.market.eq("rush_att"), ["team", "player_clean_key", "actual"]].drop_duplicates(
            ["team", "player_clean_key"]
        )
        trace = trace.merge(actual, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        trace.insert(0, "week", week)

        trace["canonical_vs_keyed_lookup_abs"] = (
            pd.to_numeric(trace.canonical_mc_proj, errors="coerce")
            - pd.to_numeric(trace.keyed_lookup_mean, errors="coerce")
        ).abs()
        trace["keyed_lookup_vs_realized_abs"] = (
            pd.to_numeric(trace.keyed_lookup_mean, errors="coerce")
            - pd.to_numeric(trace.keyed_realized_mean, errors="coerce")
        ).abs()
        trace["realized_vs_expected_abs"] = (
            pd.to_numeric(trace.keyed_realized_mean, errors="coerce")
            - pd.to_numeric(trace.keyed_expected_from_probability, errors="coerce")
        ).abs()
        all_rows.append(trace)

        print(
            f"[rush28] W{week:02d} rows={len(trace)} "
            f"canonical/keyed={trace.canonical_vs_keyed_lookup_abs.max():.10f} "
            f"lookup/realized={trace.keyed_lookup_vs_realized_abs.max():.10f} "
            f"mean sampling={trace.realized_vs_expected_abs.mean():.6f}"
        )

    out = pd.concat(all_rows, ignore_index=True)
    stages = _stage_metrics(out)
    divergence = pd.DataFrame([{
        "rows": len(out),
        "canonical_vs_keyed_lookup_mismatch_rows": int((out.canonical_vs_keyed_lookup_abs > 1e-9).sum()),
        "canonical_vs_keyed_lookup_max_abs": float(out.canonical_vs_keyed_lookup_abs.max()),
        "keyed_lookup_vs_realized_mismatch_rows": int((out.keyed_lookup_vs_realized_abs > 1e-9).sum()),
        "keyed_lookup_vs_realized_max_abs": float(out.keyed_lookup_vs_realized_abs.max()),
        "mean_abs_realized_vs_expected": float(out.realized_vs_expected_abs.mean()),
        "max_abs_realized_vs_expected": float(out.realized_vs_expected_abs.max()),
        "mean_raw_team_share_sum": float(out.drop_duplicates(["week", "event_id", "team"]).raw_team_rush_share_sum.mean()),
        "mean_residual_probability": float(out.drop_duplicates(["week", "event_id", "team"]).residual_probability.mean()),
    }])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / "keyed_rushing_player_trace.csv", index=False)
    stages.to_csv(a.out_dir / "keyed_rushing_stage_summary.csv", index=False)
    divergence.to_csv(a.out_dir / "keyed_rushing_divergence_summary.csv", index=False)
    print("\n[rush28] stage summary\n", stages.to_string(index=False))
    print("\n[rush28] divergence summary\n", divergence.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
