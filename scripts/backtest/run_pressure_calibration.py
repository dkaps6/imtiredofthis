#!/usr/bin/env python3
"""Migration 22: decompose and calibrate pressure-matchup rules leakage-safely."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling import simulation_rules
from scripts.modeling.rules_v2 import MatchupMultipliers, offensive_pressure_mismatch

# threshold, pass-eff penalty, clean-pocket bonus, RB target boost, risk/volatility boost
VARIANTS = {
    "current_full": (0.05, 0.94, 1.03, 1.25, 1.10),
    "no_pressure_effect": (0.05, 1.00, 1.00, 1.00, 1.00),
    "pass_eff_97": (0.05, 0.97, 1.03, 1.25, 1.10),
    "pass_eff_98": (0.05, 0.98, 1.03, 1.25, 1.10),
    "no_clean_bonus": (0.05, 0.94, 1.00, 1.25, 1.10),
    "rb_checkdown_110": (0.05, 0.94, 1.03, 1.10, 1.10),
    "rb_checkdown_115": (0.05, 0.94, 1.03, 1.15, 1.10),
    "no_risk_volatility": (0.05, 0.94, 1.03, 1.25, 1.00),
    "threshold_075": (0.075, 0.94, 1.03, 1.25, 1.10),
    "threshold_100": (0.10, 0.94, 1.03, 1.25, 1.10),
    "gentle_combo": (0.075, 0.98, 1.01, 1.10, 1.05),
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _pressure_rule_factory(params):
    threshold, pass_penalty, clean_bonus, rb_boost, risk_boost = params
    base_fn = simulation_rules.matchup_multipliers

    def calibrated(offense, defense):
        # Start from canonical rules, then remove only the canonical pressure effects.
        base = base_fn(offense, defense)
        pressure = offensive_pressure_mismatch(offense, defense)
        values = base.__dict__.copy()
        if pressure > 0.05:
            values["pass_eff_mult"] /= 0.94
            values["rb_rec_target_mult"] /= 1.25
            values["sack_mult"] /= 1.10
            values["int_mult"] /= 1.10
            values["volatility_mult"] /= 1.10
        elif pressure < -0.05:
            values["pass_eff_mult"] /= 1.03
        if pressure > threshold:
            values["pass_eff_mult"] *= pass_penalty
            values["rb_rec_target_mult"] *= rb_boost
            values["sack_mult"] *= risk_boost
            values["int_mult"] *= risk_boost
            values["volatility_mult"] *= risk_boost
        elif pressure < -threshold:
            values["pass_eff_mult"] *= clean_bonus
        return MatchupMultipliers(**{k: max(0.50, min(1.80, float(v))) for k, v in values.items()})
    return calibrated


def _project_week(*, variant, params, player_logs, team_weekly, schedule, universe, injuries, weather, season, week, prior_season, iterations, seed):
    bundle = build_historical_context_bundle(player_logs=player_logs, team_weekly=team_weekly, pregame_universe=universe, schedule=schedule, season=season, week=week, prior_season=prior_season, injuries=injuries, weather=weather)
    original = simulation_rules.matchup_multipliers
    try:
        simulation_rules.matchup_multipliers = _pressure_rule_factory(params)
        mc = build_mc_predictions(bundle, iterations=iterations, seed=seed)
    finally:
        simulation_rules.matchup_multipliers = original
    actual = build_actual_rows(player_logs, season, week)
    keep = [c for c in ["player", "player_clean_key", "team", "opponent", "position", "market", "mc_proj"] if c in mc.columns]
    out = mc[keep].merge(actual, on=["team", "player_clean_key", "market"], how="inner", validate="one_to_one")
    out.insert(0, "variant", variant); out.insert(1, "season", season); out.insert(2, "week", week)
    return out


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (variant, market), g in predictions.groupby(["variant","market"], sort=True):
        a=pd.to_numeric(g.actual,errors="coerce"); p=pd.to_numeric(g.mc_proj,errors="coerce"); ok=a.notna()&p.notna(); a=a[ok]; p=p[ok]
        if not len(a): continue
        e=p-a; corr=float(a.corr(p)) if len(a)>1 and a.nunique()>1 and p.nunique()>1 else np.nan
        rows.append({"variant":variant,"market":market,"n":len(a),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":corr})
    s=pd.DataFrame(rows)
    base=s[s.variant.eq("current_full")][["market","mae","rmse"]].rename(columns={"mae":"full_mae","rmse":"full_rmse"})
    s=s.merge(base,on="market",how="left"); s["delta_mae_vs_full"]=s.mae-s.full_mae; s["delta_rmse_vs_full"]=s.rmse-s.full_rmse
    return s.sort_values(["market","mae"]).reset_index(drop=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=1000)
    p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/pressure_calibration")); args=p.parse_args()
    logs=_read(args.player_logs,"player logs"); team=_read(args.team_weekly,"team weekly"); schedule=_read(args.schedule,"schedule"); ih=pd.read_csv(args.injuries) if args.injuries.exists() else pd.DataFrame(); wh=pd.read_csv(args.weather) if args.weather.exists() else pd.DataFrame(); all_rows=[]
    for week in _parse_weeks(args.weeks):
        universe=_read(args.universe_dir/f"{args.season}_week_{week:02d}.csv",f"universe W{week}"); injuries=_exact_week(ih,args.season,week); weather=_exact_week(wh,args.season,week)
        for name,params in VARIANTS.items():
            r=_project_week(variant=name,params=params,player_logs=logs,team_weekly=team,schedule=schedule,universe=universe,injuries=injuries,weather=weather,season=args.season,week=week,prior_season=args.prior_season,iterations=args.iterations,seed=12000+week); all_rows.append(r); print(f"[pressure-cal] W{week:02d} {name} rows={len(r)}")
    pred=pd.concat(all_rows,ignore_index=True); summary=summarize(pred); args.out_dir.mkdir(parents=True,exist_ok=True); pred.to_csv(args.out_dir/"pressure_calibration_predictions.csv",index=False); summary.to_csv(args.out_dir/"pressure_calibration_summary.csv",index=False)
    passing=summary[summary.market.eq("passing_yards")].sort_values("mae"); passing.to_csv(args.out_dir/"pressure_calibration_passing_rank.csv",index=False); print("\n[pressure-cal] passing ranking\n",passing.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
