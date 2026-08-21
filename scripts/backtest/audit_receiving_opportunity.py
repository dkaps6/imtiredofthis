#!/usr/bin/env python3
"""Migration 31: diagnose receiving opportunity / target allocation end-to-end.

Diagnostic only. Rebuilds leakage-safe pregame metrics, derives the exact target
probabilities used by the canonical allocator, and compares staged deterministic
expectations plus canonical MC output with actual receptions/receiving yards.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from scripts.modeling import simulation_rules


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _finite(v, default=0.0):
    try:
        x=float(v); return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _allocator_probabilities(shares: np.ndarray) -> tuple[np.ndarray,float,float]:
    clean=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    clean=np.clip(clean,0.0,0.95); raw_sum=float(clean.sum()); used=clean.copy()
    if raw_sum>0.95: used*=0.95/raw_sum
    residual=max(0.0,1.0-float(used.sum())); probs=np.append(used,residual); probs=probs/probs.sum()
    return probs[:-1],raw_sum,float(probs[-1])


def _prepared_metrics(bundle):
    m=cp.build_market_frame(bundle)
    m=apply_bayesian_to_metrics(m,build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m["player_clean_key"].fillna("").astype(str)
    key=["event_id","team","player_clean_key"]
    return m.sort_values(key).drop_duplicates(key,keep="last").copy()


def _stage_summary(x: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for actual_col,pred_cols in {
        "receptions_actual":["det_expected_targets","det_expected_receptions","mc_receptions"],
        "rec_yards_actual":["det_expected_rec_yards","mc_rec_yards"],
    }.items():
        for col in pred_cols:
            g=x.loc[pd.to_numeric(x[actual_col],errors="coerce").notna() & pd.to_numeric(x[col],errors="coerce").notna()].copy()
            if g.empty: continue
            a=pd.to_numeric(g[actual_col],errors="coerce"); p=pd.to_numeric(g[col],errors="coerce"); e=p-a
            rows.append({"actual_market":actual_col,"stage":col,"n":len(g),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(p.corr(a)) if p.nunique()>1 and a.nunique()>1 else np.nan,"pred_mean":float(p.mean()),"actual_mean":float(a.mean())})
    return pd.DataFrame(rows)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=2000)
    p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/receiving_opportunity_diagnostics")); a=p.parse_args()
    logs=_read(a.player_logs,"player logs"); team=_read(a.team_weekly,"team weekly"); sched=_read(a.schedule,"schedule"); injuries=_optional(a.injuries); weather=_optional(a.weather); all_rows=[]; team_rows=[]
    for week in _parse_weeks(a.weeks):
        u=_read(a.universe_dir/f"{a.season}_week_{week:02d}.csv",f"universe W{week}")
        b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=week,prior_season=a.prior_season,injuries=_exact_week(injuries,a.season,week),weather=_exact_week(weather,a.season,week))
        m=_prepared_metrics(b); mc=cp.build_mc_predictions(b,iterations=a.iterations,seed=42+week); actual=cp.build_actual_rows(logs,a.season,week)
        rec_actual=actual.loc[actual.market.eq("receptions"),["team","player_clean_key","actual"]].rename(columns={"actual":"receptions_actual"}); yards_actual=actual.loc[actual.market.eq("rec_yards"),["team","player_clean_key","actual"]].rename(columns={"actual":"rec_yards_actual"})
        mc_rec=mc.loc[mc.market.eq("receptions"),["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"mc_receptions"}); mc_y=mc.loc[mc.market.eq("rec_yards"),["event_id","team","player_clean_key","mc_proj"]].rename(columns={"mc_proj":"mc_rec_yards"})
        for (game,t),g in m.groupby(["event_id","team"],dropna=False):
            shares=np.array([_finite(r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("target_share",0.0))),0.0) for _,r in g.iterrows()]); probs,raw_sum,resid=_allocator_probabilities(shares)
            plays=float(np.mean([_finite(v,64.0) for v in g.get("rules_plays_est",pd.Series([64.0]*len(g)))])); pass_rate=float(np.mean([_finite(v,0.57) for v in g.get("rules_pass_rate",pd.Series([0.57]*len(g)))])); team_pass=plays*pass_rate
            team_rows.append({"week":week,"event_id":game,"team":t,"players_in_target_pool":len(g),"raw_target_share_sum":raw_sum,"residual_target_probability":resid,"det_team_pass_attempts":team_pass})
            for j,(_,r) in enumerate(g.iterrows()):
                catch=_finite(r.get("rules_catch_rate",r.get("bayes_receptions_per_target",r.get("receptions_per_target",0.64))),0.64); ypt=_finite(r.get("rules_ypt",r.get("bayes_ypt",r.get("ypt",7.5))),7.5)
                et=team_pass*float(probs[j]); er=et*catch; ey=et*ypt
                all_rows.append({"week":week,"event_id":game,"team":t,"player_clean_key":r.player_clean_key,"player":r.get("player",""),"position":r.get("position",""),"raw_target_share":float(shares[j]),"final_target_probability":float(probs[j]),"raw_team_target_share_sum":raw_sum,"residual_target_probability":resid,"det_team_pass_attempts":team_pass,"det_expected_targets":et,"catch_rate":catch,"det_expected_receptions":er,"ypt":ypt,"det_expected_rec_yards":ey})
        x=pd.DataFrame([r for r in all_rows if r["week"]==week]); x=x.merge(mc_rec,on=["event_id","team","player_clean_key"],how="left").merge(mc_y,on=["event_id","team","player_clean_key"],how="left").merge(rec_actual,on=["team","player_clean_key"],how="left").merge(yards_actual,on=["team","player_clean_key"],how="left")
        all_rows=[r for r in all_rows if r["week"]!=week] + x.to_dict("records")
        print(f"[recv31] W{week:02d} players={len(x)} raw_target_sum_mean={pd.DataFrame(team_rows).loc[lambda d:d.week.eq(week),'raw_target_share_sum'].mean():.3f}")
    player=pd.DataFrame(all_rows); teams=pd.DataFrame(team_rows); stages=_stage_summary(player)
    pool=teams[["players_in_target_pool","raw_target_share_sum","residual_target_probability","det_team_pass_attempts"]].describe().T.reset_index().rename(columns={"index":"metric"})
    a.out_dir.mkdir(parents=True,exist_ok=True); player.to_csv(a.out_dir/"receiving_opportunity_player_diagnostics.csv",index=False); teams.to_csv(a.out_dir/"receiving_opportunity_team_diagnostics.csv",index=False); stages.to_csv(a.out_dir/"receiving_opportunity_stage_summary.csv",index=False); pool.to_csv(a.out_dir/"receiving_opportunity_pool_summary.csv",index=False)
    print("\n[recv31] stage summary\n",stages.to_string(index=False)); print("\n[recv31] pool summary\n",pool.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
