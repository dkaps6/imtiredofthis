#!/usr/bin/env python3
"""Leakage-safe historical feature ablation for the Monte Carlo projection path.

Each variant removes one contextual feature family while keeping the same
pregame universe, historical cutoff, simulation seed, and outcome sample.
Positive delta_mae_vs_full means the removed feature helped the full model;
negative delta means the full model was worse with that feature enabled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks

VARIANT_COLUMNS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no_coverage": ("coverage_man_rate", "coverage_zone_rate", "man_rate", "zone_rate", "middle_open_rate"),
    "no_box": ("light_box_rate", "heavy_box_rate", "avg_defenders_in_box", "box_snap_count"),
    "no_pressure": ("pressure_rate_generated", "pressure_rate_allowed", "pressure_rate", "dl_pressure_rate", "def_pressure_rate", "off_pressure_rate_allowed"),
    "no_pace_volume": ("neutral_pace", "neutral_pace_last5", "sec_per_play_last5", "plays_est", "plays_per_game"),
    "no_script_tendency": ("success_rate_off", "success_rate_def", "proe", "pass_rate_over_expected"),
    "no_injuries": (),
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def mask_team_features(team_weekly: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant not in VARIANT_COLUMNS:
        raise ValueError(f"unknown ablation variant: {variant}")
    out = team_weekly.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for col in VARIANT_COLUMNS[variant]:
        if col in out.columns:
            out[col] = np.nan
    return out


def _project_variant_week(*, variant: str, player_logs: pd.DataFrame, team_weekly: pd.DataFrame, schedule: pd.DataFrame, universe: pd.DataFrame, injuries: pd.DataFrame, weather: pd.DataFrame, season: int, week: int, prior_season: int, iterations: int, seed: int) -> pd.DataFrame:
    masked_team = mask_team_features(team_weekly, variant)
    variant_injuries = pd.DataFrame() if variant == "no_injuries" else injuries
    bundle = build_historical_context_bundle(player_logs=player_logs, team_weekly=masked_team, pregame_universe=universe, schedule=schedule, season=season, week=week, prior_season=prior_season, injuries=variant_injuries, weather=weather)
    mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    actual = build_actual_rows(player_logs, season, week)
    keep = [c for c in ["player", "player_clean_key", "team", "opponent", "position", "market", "mc_proj"] if c in mc.columns]
    out = mc[keep].merge(actual, on=["team", "player_clean_key", "market"], how="inner", validate="one_to_one")
    out.insert(0, "variant", variant)
    out.insert(1, "season", int(season))
    out.insert(2, "week", int(week))
    return out


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, market), g in predictions.groupby(["variant", "market"], sort=True):
        actual = pd.to_numeric(g["actual"], errors="coerce")
        proj = pd.to_numeric(g["mc_proj"], errors="coerce")
        ok = actual.notna() & proj.notna()
        a, p = actual[ok], proj[ok]
        if len(a) == 0:
            continue
        err = p - a
        corr = float(a.corr(p)) if len(a) > 1 and a.nunique() > 1 and p.nunique() > 1 else np.nan
        rows.append({"variant": variant, "market": market, "n": int(len(a)), "mae": float(err.abs().mean()), "rmse": float(np.sqrt(np.mean(np.square(err)))), "bias": float(err.mean()), "correlation": corr})
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    base = summary.loc[summary["variant"].eq("full"), ["market", "mae", "rmse"]].rename(columns={"mae": "full_mae", "rmse": "full_rmse"})
    summary = summary.merge(base, on="market", how="left", validate="many_to_one")
    summary["delta_mae_vs_full"] = summary["mae"] - summary["full_mae"]
    summary["delta_rmse_vs_full"] = summary["rmse"] - summary["full_rmse"]
    summary["feature_effect"] = np.select([summary["variant"].eq("full"), summary["delta_mae_vs_full"].gt(0.05), summary["delta_mae_vs_full"].lt(-0.05)], ["baseline", "helps_full_model", "hurts_full_model"], default="near_neutral")
    return summary.sort_values(["market", "delta_mae_vs_full"], ascending=[True, False]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18")
    p.add_argument("--iterations", type=int, default=750)
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/ablation"))
    args = p.parse_args()

    player_logs = _read(args.player_logs, "player logs")
    team_weekly = _read(args.team_weekly, "team weekly history")
    schedule = _read(args.schedule, "schedule")
    injuries_history = pd.read_csv(args.injuries) if args.injuries.exists() and args.injuries.stat().st_size else pd.DataFrame()
    weather_history = pd.read_csv(args.weather) if args.weather.exists() and args.weather.stat().st_size else pd.DataFrame()
    all_rows = []
    for week in _parse_weeks(args.weeks):
        universe = _read(args.universe_dir / f"{args.season}_week_{week:02d}.csv", f"pregame universe W{week}")
        injuries = _exact_week(injuries_history, args.season, week)
        weather = _exact_week(weather_history, args.season, week)
        for variant in VARIANT_COLUMNS:
            pred = _project_variant_week(variant=variant, player_logs=player_logs, team_weekly=team_weekly, schedule=schedule, universe=universe, injuries=injuries, weather=weather, season=args.season, week=week, prior_season=args.prior_season, iterations=args.iterations, seed=9000 + week)
            all_rows.append(pred)
            print(f"[ablation] W{week:02d} variant={variant} rows={len(pred)}")
    predictions = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary = summarize(predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_dir / "feature_ablation_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "feature_ablation_summary.csv", index=False)
    print("\n[ablation] summary (positive delta_mae means feature helps full model)")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
