#!/usr/bin/env python3
"""Migration 53B: canonical MC matrix for attempts-only, YPA-only, and joint."""
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


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size: raise RuntimeError(f"missing {path}")
    return pd.read_csv(path)


def opt(path: Path) -> pd.DataFrame: return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
def num(value): return pd.to_numeric(value, errors="coerce")
def met(a, p):
    z=pd.DataFrame({"a":num(a),"p":num(p)}).dropna();e=z.p-z.a
    return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>2 else np.nan,"catastrophic_100plus":int(e.abs().ge(100).sum()),"under_100plus":int(e.le(-100).sum()),"over_100plus":int(e.ge(100).sum())}


def wrapper(original, attempt_factors: dict[str,float], ypa_factors: dict[tuple[str,str],float], mode: str):
    def apply(metrics: pd.DataFrame) -> pd.DataFrame:
        out=original(metrics); team=out.team.astype(str).str.upper().str.strip(); key=out.player_clean_key.astype(str)
        if mode in {"attempts_only","joint"}: out["rules_pass_rate"]=(num(out.rules_pass_rate)*team.map(attempt_factors).fillna(1.0)).clip(.25,.85)
        if mode in {"ypa_only","joint"}:
            factor=pd.Series([ypa_factors.get((t,k),1.0) for t,k in zip(team,key)],index=out.index)
            out["rules_ypa"]=(num(out.rules_ypa)*factor).clip(4.5,10.5)
        return out
    return apply


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,default=2025);p.add_argument("--prior-season",type=int,default=2024);p.add_argument("--weeks",default="1-18");p.add_argument("--iterations",type=int,default=2000);p.add_argument("--candidate-trace",type=Path,default=Path("data/backtests/qb_joint_attempts_ypa/qb_joint_attempts_ypa_trace.csv"));p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv"));p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv"));p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv"));p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe"));p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv"));p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv"));p.add_argument("--out-dir",type=Path,default=Path("data/backtests/qb_joint_attempts_ypa_mc"));a=p.parse_args()
    cand=read(a.candidate_trace);logs=read(a.player_logs);team=read(a.team_weekly);sched=read(a.schedule);inj=opt(a.injuries);weather=opt(a.weather);original=simulation_rules.apply_rules_to_metrics;traces=[];candidate_weeks=set(num(cand.week).dropna().astype(int))
    for week in [w for w in _parse_weeks(a.weeks) if w in candidate_weeks]:
        cw=cand[num(cand.week).eq(week)].copy(); af=dict(zip(cw.team.astype(str).str.upper().str.strip(),(num(cw.attempts_gamescript)/num(cw.attempts_current).replace(0,np.nan)).clip(.75,1.25)));yf={(str(r.team).upper().strip(),str(r.player_clean_key)):float(np.clip(r.ypa_contextual/r.ypa_current,.75,1.25)) for _,r in cw.iterrows() if pd.notna(r.ypa_contextual) and pd.notna(r.ypa_current) and r.ypa_current!=0}
        universe=read(a.universe_dir/f"{a.season}_week_{week:02d}.csv");bundle=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=universe,schedule=sched,season=a.season,week=week,prior_season=a.prior_season,injuries=_exact_week(inj,a.season,week),weather=_exact_week(weather,a.season,week));actual=build_actual_rows(logs,a.season,week)
        for mode in ["current","attempts_only","ypa_only","joint"]:
            fn=original if mode=="current" else wrapper(original,af,yf,mode)
            with patch.object(simulation_rules,"apply_rules_to_metrics",side_effect=fn): mc=build_mc_predictions(bundle,iterations=a.iterations,seed=53+week)
            z=mc.merge(actual,on=["team","player_clean_key","market"],how="inner");z["candidate"]=mode;z["week"]=week;traces.append(z)
        print(f"[m53] W{week:02d} stable_qbs={len(cw)}")
    t=pd.concat(traces,ignore_index=True);keys=cand[["week","team","player_clean_key"]].drop_duplicates();stable=t[t.market.eq("pass_yards")].merge(keys,on=["week","team","player_clean_key"],how="inner");summary=[]
    for mode,g in stable.groupby("candidate"):summary.append({"candidate":mode,"slice":"stable_qb","market":"pass_yards",**met(g.actual,g.mc_proj)})
    for (mode,market),g in t.groupby(["candidate","market"]):summary.append({"candidate":mode,"slice":"all_available","market":market,**met(g.actual,g.mc_proj)})
    s=pd.DataFrame(summary);a.out_dir.mkdir(parents=True,exist_ok=True);t.to_csv(a.out_dir/"qb_joint_attempts_ypa_mc_trace.csv",index=False);stable.to_csv(a.out_dir/"qb_joint_attempts_ypa_mc_stable.csv",index=False);s.to_csv(a.out_dir/"qb_joint_attempts_ypa_mc_summary.csv",index=False);print("=== JOINT CANONICAL MC ===");print(s.to_string(index=False));return 0


if __name__=="__main__":raise SystemExit(main())
