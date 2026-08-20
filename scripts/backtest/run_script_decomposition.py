#!/usr/bin/env python3
"""Decompose the game-script/pass-tendency rule without changing production logic."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules
from scripts.modeling.rules_v2 import GameScriptProjection, estimate_plays, offensive_pressure_mismatch, script_distribution, success_diff

VARIANTS = {
    "full": {"use_proe": True, "use_success": True, "game_state_coeff": 0.08},
    "no_proe": {"use_proe": False, "use_success": True, "game_state_coeff": 0.08},
    "no_success_diff": {"use_proe": True, "use_success": False, "game_state_coeff": 0.08},
    "no_lead_trail_adjustment": {"use_proe": True, "use_success": True, "game_state_coeff": 0.00},
    "fixed_pass_share": {"use_proe": False, "use_success": False, "game_state_coeff": 0.00},
    "game_state_0.02": {"use_proe": True, "use_success": True, "game_state_coeff": 0.02},
    "game_state_0.04": {"use_proe": True, "use_success": True, "game_state_coeff": 0.04},
    "game_state_0.06": {"use_proe": True, "use_success": True, "game_state_coeff": 0.06},
    "game_state_0.08": {"use_proe": True, "use_success": True, "game_state_coeff": 0.08},
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _num(value, default=0.0):
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def project_script_variant(offense, defense, *, use_proe: bool, use_success: bool, game_state_coeff: float):
    diff = success_diff(offense, defense) if use_success else 0.0
    lead, neutral, trail = script_distribution(diff)
    plays = estimate_plays(offense)
    pass_share = 0.55
    if use_proe:
        pass_share += float(np.clip(_num(getattr(offense, "proe", 0.0)), -0.10, 0.10))
    pass_share += float(game_state_coeff) * (trail - lead)
    pass_share = float(np.clip(pass_share, 0.42, 0.70))
    pressure = offensive_pressure_mismatch(offense, defense)
    return GameScriptProjection(
        projected_plays=plays,
        projected_pass_attempts=plays * pass_share,
        projected_rush_attempts=plays * (1.0 - pass_share),
        lead_prob=lead, neutral_prob=neutral, trail_prob=trail,
        pressure_mismatch=abs(pressure) >= 0.05,
        blowout_risk=abs(diff) >= 0.06,
        shootout_risk=abs(diff) < 0.03 and plays >= 68.0,
    )


def _project_week(variant, cfg, *, player_logs, team_weekly, schedule, universe, injuries, weather, season, week, prior_season, iterations, seed):
    bundle = build_historical_context_bundle(
        player_logs=player_logs, team_weekly=team_weekly, pregame_universe=universe,
        schedule=schedule, season=season, week=week, prior_season=prior_season,
        injuries=injuries, weather=weather,
    )
    def patched(offense, defense):
        return project_script_variant(offense, defense, **cfg)
    with patch.object(simulation_rules, "project_game_script", side_effect=patched):
        mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    actual = build_actual_rows(player_logs, season, week)
    keep = [c for c in ["player", "player_clean_key", "team", "opponent", "position", "market", "mc_proj", "mc_projected_plays", "mc_dropback_rate", "mc_expected_pass_attempts"] if c in mc.columns]
    out = mc[keep].merge(actual, on=["team", "player_clean_key", "market"], how="inner", validate="one_to_one")
    out.insert(0, "variant", variant); out.insert(1, "season", season); out.insert(2, "week", week)
    return out


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, market), g in predictions.groupby(["variant", "market"]):
        a = pd.to_numeric(g["actual"], errors="coerce"); p = pd.to_numeric(g["mc_proj"], errors="coerce")
        ok = a.notna() & p.notna(); a, p = a[ok], p[ok]
        if not len(a): continue
        err = p - a
        rows.append({"variant": variant, "market": market, "n": len(a), "mae": err.abs().mean(), "rmse": np.sqrt(np.mean(err**2)), "bias": err.mean(), "correlation": a.corr(p) if len(a) > 1 else np.nan})
    s = pd.DataFrame(rows)
    base = s[s.variant.eq("full")][["market","mae","rmse"]].rename(columns={"mae":"full_mae","rmse":"full_rmse"})
    s = s.merge(base, on="market", how="left")
    s["delta_mae_vs_full"] = s.mae - s.full_mae
    s["delta_rmse_vs_full"] = s.rmse - s.full_rmse
    return s.sort_values(["market","mae"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025); p.add_argument("--prior-season", type=int, default=2024)
    p.add_argument("--weeks", default="1-18"); p.add_argument("--iterations", type=int, default=750)
    p.add_argument("--player-logs", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    p.add_argument("--team-weekly", type=Path, default=Path("data/backtests/team_weekly_history.csv"))
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--universe-dir", type=Path, default=Path("data/backtests/pregame_universe"))
    p.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/script_decomposition"))
    args = p.parse_args()
    logs = _read(args.player_logs,"player logs"); team = _read(args.team_weekly,"team weekly"); sched = _read(args.schedule,"schedule")
    inj_hist = pd.read_csv(args.injuries) if args.injuries.exists() else pd.DataFrame(); wx_hist = pd.read_csv(args.weather) if args.weather.exists() else pd.DataFrame()
    rows=[]
    for week in _parse_weeks(args.weeks):
        universe=_read(args.universe_dir/f"{args.season}_week_{week:02d}.csv",f"universe W{week}")
        injuries=_exact_week(inj_hist,args.season,week); weather=_exact_week(wx_hist,args.season,week)
        for variant,cfg in VARIANTS.items():
            pred=_project_week(variant,cfg,player_logs=logs,team_weekly=team,schedule=sched,universe=universe,injuries=injuries,weather=weather,season=args.season,week=week,prior_season=args.prior_season,iterations=args.iterations,seed=12000+week)
            rows.append(pred); print(f"[script-decomp] W{week:02d} {variant} rows={len(pred)}")
    predictions=pd.concat(rows,ignore_index=True); summary=summarize(predictions)
    args.out_dir.mkdir(parents=True,exist_ok=True); predictions.to_csv(args.out_dir/"script_decomposition_predictions.csv",index=False); summary.to_csv(args.out_dir/"script_decomposition_summary.csv",index=False)
    print(summary.to_string(index=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
