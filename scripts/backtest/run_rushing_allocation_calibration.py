#!/usr/bin/env python3
"""Migration 25: trace and calibrate the MC rushing-allocation pool.

The production simulator is not changed here. This harness rebuilds the same
leakage-safe pregame metrics, then isolates carry allocation so we can measure
whether low-usage roster players with Bayesian nonzero rush-share priors dilute
the modeled rushing pool.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_actual_rows, build_market_frame
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules

VARIANTS = ("current", "min_02", "min_05", "top4", "top5", "top6")


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _finite(v, default=0.0):
    try:
        x=float(v); return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _prepare_metrics(bundle):
    metrics = build_market_frame(bundle)
    metrics = apply_bayesian_to_metrics(metrics, build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules, "load_model_contexts", return_value=(bundle.teams, bundle.players)):
        metrics = simulation_rules.apply_rules_to_metrics(metrics)
    # one row per player; all rule/Bayes inputs are player-level and repeated by market
    cols=[c for c in ["event_id","team","opponent","player","player_clean_key","position","rules_plays_est","rules_pass_rate","rules_rush_share","bayes_rush_share","rush_share"] if c in metrics.columns]
    return metrics[cols].drop_duplicates(["event_id","team","player_clean_key"], keep="last").copy()


def _transform(shares: np.ndarray, variant: str) -> np.ndarray:
    s=np.nan_to_num(shares.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    s=np.clip(s,0.0,0.95)
    if variant=="min_02": s=np.where(s>=0.02,s,0.0)
    elif variant=="min_05": s=np.where(s>=0.05,s,0.0)
    elif variant.startswith("top"):
        n=int(variant[3:])
        if len(s)>n:
            keep=np.argpartition(s,-n)[-n:]
            mask=np.zeros(len(s),dtype=bool); mask[keep]=True; s=np.where(mask,s,0.0)
    return s


def _simulate_week(metrics: pd.DataFrame, actual: pd.DataFrame, *, variant: str, iterations: int, seed: int):
    rng=np.random.default_rng(seed); rows=[]; teams=[]
    actual=actual.loc[actual.market.eq("rush_att"),["team","player_clean_key","actual"]].copy()
    for (game,team), g in metrics.groupby(["event_id","team"],dropna=False):
        base=np.array([_finite(v,0.0) for v in g["rules_rush_share"]],dtype=float)
        use=_transform(base,variant)
        raw_sum=float(base.sum()); used_sum=float(use.sum())
        if used_sum>0.95: use*=0.95/used_sum
        residual=max(0.0,1.0-float(use.sum())); probs=np.append(use,residual); probs=probs/probs.sum()
        plays=np.mean([_finite(v,64.0) for v in g["rules_plays_est"]]); pass_rate=np.mean([_finite(v,0.57) for v in g["rules_pass_rate"]])
        sim_plays=np.rint(np.clip(rng.normal(plays,3.5,iterations),45,85)).astype(int)
        sim_pass=rng.binomial(sim_plays,np.clip(pass_rate,0.25,0.82)); sim_rush=sim_plays-sim_pass
        alloc=np.empty((iterations,len(g)),dtype=int)
        for i,total in enumerate(sim_rush): alloc[i]=rng.multinomial(int(total),probs)[:-1]
        means=alloc.mean(axis=0)
        for j,(_,p) in enumerate(g.iterrows()):
            rows.append({"variant":variant,"team":team,"player_clean_key":p.player_clean_key,"position":p.get("position",""),"mc_rush_att":float(means[j]),"raw_rush_share":float(base[j]),"used_rush_share":float(use[j])})
        teams.append({"variant":variant,"team":team,"game":game,"players_in_pool":len(g),"raw_share_sum":raw_sum,"post_gate_share_sum":used_sum,"normalized_player_share_sum":float(use.sum()),"residual_share":residual})
    p=pd.DataFrame(rows).merge(actual,on=["team","player_clean_key"],how="inner")
    return p,pd.DataFrame(teams)


def _summary(pred):
    out=[]
    for v,g in pred.groupby("variant"):
        e=g.mc_rush_att-g.actual
        corr=float(g.mc_rush_att.corr(g.actual)) if g.actual.nunique()>1 and g.mc_rush_att.nunique()>1 else np.nan
        out.append({"variant":v,"n":len(g),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":corr})
    return pd.DataFrame(out).sort_values("mae").reset_index(drop=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=1000)
    p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/rushing_allocation_calibration")); a=p.parse_args()
    logs=_read(a.player_logs,"player logs"); team=_read(a.team_weekly,"team weekly"); sched=_read(a.schedule,"schedule"); ih=pd.read_csv(a.injuries) if a.injuries.exists() else pd.DataFrame(); wh=pd.read_csv(a.weather) if a.weather.exists() else pd.DataFrame(); preds=[]; team_rows=[]
    for w in _parse_weeks(a.weeks):
        u=_read(a.universe_dir/f"{a.season}_week_{w:02d}.csv",f"universe W{w}")
        b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=w,prior_season=a.prior_season,injuries=_exact_week(ih,a.season,w),weather=_exact_week(wh,a.season,w))
        m=_prepare_metrics(b); actual=build_actual_rows(logs,a.season,w)
        for v in VARIANTS:
            pp,tt=_simulate_week(m,actual,variant=v,iterations=a.iterations,seed=15000+w); pp.insert(1,"week",w); tt.insert(1,"week",w); preds.append(pp); team_rows.append(tt)
            print(f"[rush25] W{w:02d} {v} matched={len(pp)} full_pool_players_mean={tt.players_in_pool.mean():.2f}")
    pred=pd.concat(preds,ignore_index=True); teams=pd.concat(team_rows,ignore_index=True); summ=_summary(pred)
    a.out_dir.mkdir(parents=True,exist_ok=True); pred.to_csv(a.out_dir/"rushing_allocation_predictions.csv",index=False); teams.to_csv(a.out_dir/"rushing_allocation_pool_trace.csv",index=False); summ.to_csv(a.out_dir/"rushing_allocation_summary.csv",index=False)
    print("\n[rush25] candidate ranking\n",summ.to_string(index=False)); print("\n[rush25] pool trace\n",teams.groupby("variant")[["players_in_pool","raw_share_sum","post_gate_share_sum","residual_share"]].mean().to_string())
    return 0

if __name__=="__main__": raise SystemExit(main())
