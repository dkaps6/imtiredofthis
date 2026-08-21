#!/usr/bin/env python3
"""Migration 32: receiving target-pool calibration + canonical coverage audit.

Diagnostic-only. Rebuilds leakage-safe pregame receiving inputs, tests multiple
ways of pruning noisy fringe target-share priors, and scores simulated
receptions/receiving yards against actual outcomes. Production receiving logic
is unchanged.
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

VARIANTS = (
    "current",
    "min_01", "min_02", "min_03", "min_05",
    "top5", "top6", "top7", "top8", "top9",
    "cum_70", "cum_80", "cum_90", "cum_95",
)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return pd.read_csv(path)


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _finite(v, default=0.0) -> float:
    try:
        x=float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _prepare_metrics(bundle) -> pd.DataFrame:
    m=cp.build_market_frame(bundle)
    m=apply_bayesian_to_metrics(m,build_bayesian_baseline(bundle.player_consensus))
    with patch.object(simulation_rules,"load_model_contexts",return_value=(bundle.teams,bundle.players)):
        m=simulation_rules.apply_rules_to_metrics(m)
    m["player_clean_key"]=m["player_clean_key"].fillna("").astype(str)
    key=["event_id","team","player_clean_key"]
    return m.sort_values(key).drop_duplicates(key,keep="last").copy()


def _transform(shares: np.ndarray, variant: str) -> np.ndarray:
    s=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    s=np.clip(s,0.0,0.95)
    if variant.startswith("min_"):
        threshold=float(variant.split("_")[1])/100.0
        s=np.where(s>=threshold,s,0.0)
    elif variant.startswith("top"):
        n=int(variant[3:])
        if len(s)>n:
            order=np.argsort(s)[::-1]
            keep=order[:n]
            mask=np.zeros(len(s),dtype=bool); mask[keep]=True
            s=np.where(mask,s,0.0)
    elif variant.startswith("cum_"):
        threshold=float(variant.split("_")[1])/100.0
        order=np.argsort(s)[::-1]
        keep=np.zeros(len(s),dtype=bool)
        running=0.0
        for idx in order:
            if s[idx] <= 0:
                continue
            keep[idx]=True
            running += float(s[idx])
            if running >= threshold:
                break
        s=np.where(keep,s,0.0)
    elif variant != "current":
        raise ValueError(f"unknown variant: {variant}")
    return s


def _allocator_probabilities(shares: np.ndarray) -> tuple[np.ndarray,float,float,float]:
    clean=np.nan_to_num(np.asarray(shares,dtype=float),nan=0.0,posinf=0.0,neginf=0.0)
    clean=np.clip(clean,0.0,0.95)
    gated_sum=float(clean.sum())
    used=clean.copy()
    if gated_sum>0.95:
        used*=0.95/gated_sum
    residual=max(0.0,1.0-float(used.sum()))
    probs=np.append(used,residual); probs=probs/probs.sum()
    return probs[:-1],gated_sum,float(used.sum()),float(probs[-1])


def _simulate_variant(metrics: pd.DataFrame, actual: pd.DataFrame, *, variant: str, iterations: int, seed: int):
    rng=np.random.default_rng(seed); rows=[]; teams=[]
    rec_actual=actual.loc[actual.market.eq("receptions"),["team","player_clean_key","actual"]].rename(columns={"actual":"actual_receptions"})
    y_actual=actual.loc[actual.market.eq("rec_yards"),["team","player_clean_key","actual"]].rename(columns={"actual":"actual_rec_yards"})
    for (game,team),g in metrics.groupby(["event_id","team"],dropna=False):
        base=np.array([_finite(r.get("rules_tgt_share",r.get("bayes_tgt_share",r.get("target_share",0.0))),0.0) for _,r in g.iterrows()],dtype=float)
        gated=_transform(base,variant)
        probs,gated_sum,normalized_sum,residual=_allocator_probabilities(gated)
        plays=float(np.mean([_finite(v,64.0) for v in g.get("rules_plays_est",pd.Series([64.0]*len(g)))]))
        pass_rate=float(np.mean([_finite(v,0.57) for v in g.get("rules_pass_rate",pd.Series([0.57]*len(g)))]))
        sim_plays=np.rint(np.clip(rng.normal(plays,3.5,iterations),45,85)).astype(int)
        sim_pass_rate=np.clip(rng.normal(pass_rate,0.035,iterations),0.25,0.82)
        pass_att=rng.binomial(sim_plays,sim_pass_rate)
        alloc=np.empty((iterations,len(g)),dtype=int)
        full_probs=np.append(probs,residual); full_probs=full_probs/full_probs.sum()
        for i,total in enumerate(pass_att):
            alloc[i]=rng.multinomial(max(0,int(total)),full_probs)[:len(g)]
        pass_eff=np.clip(rng.normal(1.0,0.09,iterations),0.65,1.35)
        for j,(_,r) in enumerate(g.iterrows()):
            catch=float(np.clip(_finite(r.get("rules_catch_rate",r.get("bayes_receptions_per_target",r.get("receptions_per_target",0.64))),0.64),0.001,0.999))
            receptions=rng.binomial(alloc[:,j],catch)
            ypt=_finite(r.get("rules_ypt",r.get("bayes_ypt",r.get("ypt",7.5))),7.5)
            if not np.isfinite(ypt) or ypt<=0: ypt=7.5
            vol=float(np.clip(_finite(r.get("rules_volatility_mult",1.0),1.0),0.75,1.50))
            rec_mu=alloc[:,j]*ypt*pass_eff
            rec_sd=np.maximum(6.0,np.sqrt(np.maximum(alloc[:,j],1))*ypt*0.55)*vol
            rec_yards=np.clip(rng.normal(rec_mu,rec_sd),0.0,None)
            rows.append({"variant":variant,"event_id":game,"team":team,"player_clean_key":r.player_clean_key,"position":r.get("position",""),"raw_target_share":float(base[j]),"gated_target_share":float(gated[j]),"final_target_probability":float(probs[j]),"mc_receptions":float(receptions.mean()),"mc_rec_yards":float(rec_yards.mean())})
        teams.append({"variant":variant,"event_id":game,"team":team,"players_in_pool":len(g),"players_retained":int((gated>0).sum()),"raw_share_sum":float(base.sum()),"post_gate_share_sum":gated_sum,"normalized_player_share_sum":normalized_sum,"residual_share":residual,"mean_sim_pass_attempts":float(pass_att.mean())})
    p=pd.DataFrame(rows).merge(rec_actual,on=["team","player_clean_key"],how="left").merge(y_actual,on=["team","player_clean_key"],how="left")
    return p,pd.DataFrame(teams)


def _summary(pred: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for variant,g in pred.groupby("variant"):
        row={"variant":variant}
        for label,pcol,acol in [("receptions","mc_receptions","actual_receptions"),("rec_yards","mc_rec_yards","actual_rec_yards")]:
            x=g.loc[pd.to_numeric(g[pcol],errors="coerce").notna() & pd.to_numeric(g[acol],errors="coerce").notna()].copy()
            p=pd.to_numeric(x[pcol],errors="coerce"); a=pd.to_numeric(x[acol],errors="coerce"); e=p-a
            row[f"{label}_n"]=len(x); row[f"{label}_mae"]=float(e.abs().mean()); row[f"{label}_rmse"]=float(np.sqrt(np.mean(e*e))); row[f"{label}_bias"]=float(e.mean()); row[f"{label}_correlation"]=float(p.corr(a)) if p.nunique()>1 and a.nunique()>1 else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(["rec_yards_mae","receptions_mae"]).reset_index(drop=True)


def _coverage_audit(metrics: pd.DataFrame, mc: pd.DataFrame, actual: pd.DataFrame, week: int) -> pd.DataFrame:
    base=metrics[["event_id","team","player_clean_key"]].drop_duplicates().copy(); base["week"]=week
    for market in ["receptions","rec_yards"]:
        mc_keys=mc.loc[mc.market.eq(market),["event_id","team","player_clean_key"]].drop_duplicates(); mc_keys["in_mc"]=True
        act_keys=actual.loc[actual.market.eq(market),["team","player_clean_key"]].drop_duplicates(); act_keys["has_actual"]=True
        x=base.merge(mc_keys,on=["event_id","team","player_clean_key"],how="left").merge(act_keys,on=["team","player_clean_key"],how="left")
        x["in_mc"]=x["in_mc"].fillna(False); x["has_actual"]=x["has_actual"].fillna(False); x["market"]=market
        x["coverage_status"]=np.select([x.in_mc & x.has_actual,~x.in_mc & x.has_actual,x.in_mc & ~x.has_actual],["mc_and_actual","missing_mc_has_actual","mc_no_actual"],default="neither")
        yield x


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--season",type=int,default=2025); p.add_argument("--prior-season",type=int,default=2024); p.add_argument("--weeks",default="1-18"); p.add_argument("--iterations",type=int,default=2000)
    p.add_argument("--player-logs",type=Path,default=Path("data/backtests/player_game_logs_history.csv")); p.add_argument("--team-weekly",type=Path,default=Path("data/backtests/team_weekly_history.csv")); p.add_argument("--schedule",type=Path,default=Path("data/backtests/schedule_history.csv")); p.add_argument("--universe-dir",type=Path,default=Path("data/backtests/pregame_universe")); p.add_argument("--injuries",type=Path,default=Path("data/backtests/injuries_history.csv")); p.add_argument("--weather",type=Path,default=Path("data/backtests/weather_history.csv")); p.add_argument("--out-dir",type=Path,default=Path("data/backtests/receiving_target_pool_calibration")); a=p.parse_args()
    logs=_read(a.player_logs,"player logs"); team=_read(a.team_weekly,"team weekly"); sched=_read(a.schedule,"schedule"); injuries=_optional(a.injuries); weather=_optional(a.weather); preds=[]; pools=[]; coverage=[]
    for week in _parse_weeks(a.weeks):
        u=_read(a.universe_dir/f"{a.season}_week_{week:02d}.csv",f"universe W{week}")
        b=build_historical_context_bundle(player_logs=logs,team_weekly=team,pregame_universe=u,schedule=sched,season=a.season,week=week,prior_season=a.prior_season,injuries=_exact_week(injuries,a.season,week),weather=_exact_week(weather,a.season,week))
        m=_prepare_metrics(b); actual=cp.build_actual_rows(logs,a.season,week); canonical=cp.build_mc_predictions(b,iterations=a.iterations,seed=42+week)
        coverage.extend(list(_coverage_audit(m,canonical,actual,week)))
        for idx,v in enumerate(VARIANTS):
            pp,tt=_simulate_variant(m,actual,variant=v,iterations=a.iterations,seed=32000+week*100+idx); pp.insert(1,"week",week); tt.insert(1,"week",week); preds.append(pp); pools.append(tt)
            print(f"[recv32] W{week:02d} {v} players={len(pp)} retained_mean={tt.players_retained.mean():.2f} raw_sum={tt.raw_share_sum.mean():.3f}")
    pred=pd.concat(preds,ignore_index=True); pool=pd.concat(pools,ignore_index=True); cov=pd.concat(coverage,ignore_index=True); summ=_summary(pred)
    cov_summary=cov.groupby(["market","coverage_status"],as_index=False).size().rename(columns={"size":"rows"})
    pool_summary=pool.groupby("variant",as_index=False)[["players_in_pool","players_retained","raw_share_sum","post_gate_share_sum","normalized_player_share_sum","residual_share","mean_sim_pass_attempts"]].mean()
    a.out_dir.mkdir(parents=True,exist_ok=True); pred.to_csv(a.out_dir/"receiving_target_pool_predictions.csv",index=False); pool.to_csv(a.out_dir/"receiving_target_pool_trace.csv",index=False); summ.to_csv(a.out_dir/"receiving_target_pool_summary.csv",index=False); cov.to_csv(a.out_dir/"receiving_coverage_audit.csv",index=False); cov_summary.to_csv(a.out_dir/"receiving_coverage_summary.csv",index=False); pool_summary.to_csv(a.out_dir/"receiving_target_pool_variant_summary.csv",index=False)
    print("\n[recv32] candidate ranking\n",summ.to_string(index=False)); print("\n[recv32] coverage summary\n",cov_summary.to_string(index=False)); print("\n[recv32] pool summary\n",pool_summary.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
