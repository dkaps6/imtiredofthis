#!/usr/bin/env python3
"""Leakage-safe candidate test for pass-tendency architecture.

Compares the current script with rolling historical team dropback tendency,
league shrinkage, partial PROE, and small game-state corrections. Diagnostic
only: production project_game_script() is not changed.
"""
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

VARIANTS={
 "current_full":{"mode":"current","shrink":0.0,"proe_weight":1.0,"state":0.08},
 "fixed_55":{"mode":"fixed","shrink":0.0,"proe_weight":0.0,"state":0.00},
 "hist_team_25":{"mode":"historical","shrink":0.25,"proe_weight":0.0,"state":0.00},
 "hist_team_50":{"mode":"historical","shrink":0.50,"proe_weight":0.0,"state":0.00},
 "hist_team_75":{"mode":"historical","shrink":0.75,"proe_weight":0.0,"state":0.00},
 "hist50_state02":{"mode":"historical","shrink":0.50,"proe_weight":0.0,"state":0.02},
 "hist50_proe25":{"mode":"historical","shrink":0.50,"proe_weight":0.25,"state":0.00},
 "hist50_proe50":{"mode":"historical","shrink":0.50,"proe_weight":0.50,"state":0.00},
 "hist50_proe25_state02":{"mode":"historical","shrink":0.50,"proe_weight":0.25,"state":0.02},
}

def _read(path,label):
 if not path.exists() or path.stat().st_size==0: raise RuntimeError(f"missing {label}: {path}")
 return pd.read_csv(path)

def _num(v,d=0.0):
 try:
  x=float(v); return x if np.isfinite(x) else float(d)
 except Exception: return float(d)

def rolling_dropback_baselines(team_weekly,season,week,prior_season):
 x=team_weekly.copy(); x.columns=[str(c).strip().lower() for c in x.columns]
 if "dropback_rate" not in x.columns: raise RuntimeError("team weekly history missing dropback_rate")
 s=pd.to_numeric(x.season,errors="coerce"); w=pd.to_numeric(x.week,errors="coerce")
 h=x.loc[s.eq(prior_season)|(s.eq(season)&w.lt(week))].copy(); h["dropback_rate"]=pd.to_numeric(h.dropback_rate,errors="coerce"); h=h.dropna(subset=["dropback_rate"])
 league=float(h.dropback_rate.mean()) if len(h) else 0.55
 return h.groupby("team").dropback_rate.mean().astype(float).to_dict(),league

def candidate_pass_share(offense,defense,cfg,team_rate,league_rate):
 diff=success_diff(offense,defense); lead,neutral,trail=script_distribution(diff)
 if cfg["mode"]=="current": share=0.55+float(np.clip(_num(getattr(offense,"proe",0.0)),-0.10,0.10))
 elif cfg["mode"]=="fixed": share=0.55
 else:
  hist=team_rate if np.isfinite(team_rate) else league_rate; shrink=float(cfg["shrink"])
  share=(1-shrink)*league_rate+shrink*hist
  share+=float(cfg["proe_weight"])*float(np.clip(_num(getattr(offense,"proe",0.0)),-0.10,0.10))
 share+=float(cfg["state"])*(trail-lead)
 return float(np.clip(share,0.42,0.70)),lead,neutral,trail

def _project_week(variant,cfg,*,player_logs,team_weekly,schedule,universe,injuries,weather,season,week,prior_season,iterations,seed):
 team_rates,league=rolling_dropback_baselines(team_weekly,season,week,prior_season)
 bundle=build_historical_context_bundle(player_logs=player_logs,team_weekly=team_weekly,pregame_universe=universe,schedule=schedule,season=season,week=week,prior_season=prior_season,injuries=injuries,weather=weather)
 def patched(offense,defense):
  share,lead,neutral,trail=candidate_pass_share(offense,defense,cfg,team_rates.get(str(offense.team),league),league)
  plays=estimate_plays(offense); pressure=offensive_pressure_mismatch(offense,defense); diff=success_diff(offense,defense)
  return GameScriptProjection(plays,plays*share,plays*(1-share),lead,neutral,trail,abs(pressure)>=0.05,abs(diff)>=0.06,abs(diff)<0.03 and plays>=68.0)
 with patch.object(simulation_rules,"project_game_script",side_effect=patched): mc=build_mc_predictions(bundle,iterations=iterations,seed=seed)
 actual=build_actual_rows(player_logs,season,week); keep=[c for c in ["player","player_clean_key","team","opponent","position","market","mc_proj","mc_expected_pass_attempts"] if c in mc.columns]
 out=mc[keep].merge(actual,on=["team","player_clean_key","market"],how="inner",validate="one_to_one"); out.insert(0,"variant",variant); out.insert(1,"season",season); out.insert(2,"week",week); return out

def summarize(pred):
 rows=[]
 for (variant,market),g in pred.groupby(["variant","market"]):
  a=pd.to_numeric(g.actual,errors="coerce"); p=pd.to_numeric(g.mc_proj,errors="coerce"); ok=a.notna()&p.notna(); a,p=a[ok],p[ok]
  if not len(a): continue
  e=p-a; rows.append({"variant":variant,"market":market,"n":len(a),"mae":e.abs().mean(),"rmse":np.sqrt(np.mean(e**2)),"bias":e.mean(),"correlation":a.corr(p) if len(a)>1 else np.nan})
 s=pd.DataFrame(rows); base=s[s.variant.eq("current_full")][["market","mae","rmse"]].rename(columns={"mae":"full_mae","rmse":"full_rmse"}); s=s.merge(base,on="market",how="left"); s["delta_mae_vs_current"]=s.mae-s.full_mae; s["delta_rmse_vs_current"]=s.rmse-s.full_rmse; return s.sort_values(["market","mae"]).reset_index(drop=True)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=750); p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/pass_tendency_recalibration")); a=p.parse_args()
 logs=_read(a.player_logs,"player logs"); team=_read(a.team_weekly,"team weekly"); sched=_read(a.schedule,"schedule"); inj=pd.read_csv(a.injuries) if a.injuries.exists() else pd.DataFrame(); wx=pd.read_csv(a.weather) if a.weather.exists() else pd.DataFrame(); rows=[]
 for week in _parse_weeks(a.weeks):
  universe=_read(a.universe_dir/f"{a.season}_week_{week:02d}.csv",f"universe W{week}"); injuries=_exact_week(inj,a.season,week); weather=_exact_week(wx,a.season,week)
  for variant,cfg in VARIANTS.items(): rows.append(_project_week(variant,cfg,player_logs=logs,team_weekly=team,schedule=sched,universe=universe,injuries=injuries,weather=weather,season=a.season,week=week,prior_season=a.prior_season,iterations=a.iterations,seed=15000+week)); print(f"[pass-recal] W{week:02d} {variant}")
 pred=pd.concat(rows,ignore_index=True); summary=summarize(pred); a.out_dir.mkdir(parents=True,exist_ok=True); pred.to_csv(a.out_dir/"pass_tendency_predictions.csv",index=False); summary.to_csv(a.out_dir/"pass_tendency_summary.csv",index=False); print(summary.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
