#!/usr/bin/env python3
"""Migration 54: paired multi-season robustness summary for the M53 joint QB model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size: raise RuntimeError(f"missing {path}")
    return pd.read_csv(path, low_memory=False)


def num(value): return pd.to_numeric(value, errors="coerce")


def metrics(actual, pred) -> dict:
    z=pd.DataFrame({"a":num(actual),"p":num(pred)}).dropna();e=z.p-z.a
    return {"n":len(z),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"correlation":float(z.p.corr(z.a)) if len(z)>2 else np.nan,"catastrophic_100plus":int(e.abs().ge(100).sum()),"under_100plus":int(e.le(-100).sum()),"over_100plus":int(e.ge(100).sum())}


def recent_games(frame: pd.DataFrame, logs: pd.DataFrame, season: int) -> pd.Series:
    p=logs.copy();p.columns=[str(c).strip().lower() for c in p.columns];p["season"],p["week"]=num(p.season),num(p.week);p["pass_att_n"]=num(p.get("pass_att")).fillna(0)
    values=[]
    for r in frame.itertuples():
        g=p[p.player_clean_key.astype(str).eq(str(r.player_clean_key)) & ((p.season<season)|((p.season==season)&(p.week<int(r.week)))) & p.pass_att_n.gt(0)]
        values.append(min(8,int(len(g))))
    return pd.Series(values,index=frame.index)


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("data/backtests/qb_multi_season"));p.add_argument("--seasons",default="2024,2025");p.add_argument("--bootstrap",type=int,default=10000);p.add_argument("--seed",type=int,default=54);p.add_argument("--out-dir",type=Path,default=Path("data/backtests/qb_multi_season/summary"));a=p.parse_args();seasons=[int(v) for v in a.seasons.split(",") if v.strip()]
    stable_frames=[];all_frames=[];feature_frames=[]
    for season in seasons:
        base=a.root/str(season);stable=read(base/"qb_joint_attempts_ypa_mc/qb_joint_attempts_ypa_mc_stable.csv");stable["target_season"]=season;stable_frames.append(stable)
        all_trace=read(base/"qb_joint_attempts_ypa_mc/qb_joint_attempts_ypa_mc_trace.csv");all_trace["target_season"]=season;all_frames.append(all_trace)
        feat=read(base/"qb_joint_attempts_ypa/qb_joint_attempts_ypa_trace.csv");feat["target_season"]=season;feat["qb_recent_games"]=recent_games(feat,read(base/"player_game_logs_history.csv"),season);feature_frames.append(feat)
    stable=pd.concat(stable_frames,ignore_index=True);all_trace=pd.concat(all_frames,ignore_index=True);features=pd.concat(feature_frames,ignore_index=True)
    keys=["target_season","week","team","player_clean_key"]
    summary=[]
    for season_value,g0 in [("combined",stable),*[(str(s),stable[stable.target_season.eq(s)]) for s in seasons]]:
        for candidate,g in g0.groupby("candidate"):
            summary.append({"season":season_value,"candidate":candidate,"slice":"stable_qb","market":"pass_yards",**metrics(g.actual,g.mc_proj)})
    for (candidate,market),g in all_trace.groupby(["candidate","market"]):summary.append({"season":"combined","candidate":candidate,"slice":"all_available","market":market,**metrics(g.actual,g.mc_proj)})
    summary=pd.DataFrame(summary)

    current=stable[stable.candidate.eq("current")][keys+["actual","mc_proj"]].rename(columns={"actual":"actual_current","mc_proj":"pred_current"})
    joint=stable[stable.candidate.eq("joint")][keys+["actual","mc_proj"]].rename(columns={"actual":"actual_joint","mc_proj":"pred_joint"})
    pairs=current.merge(joint,on=keys,how="inner",validate="one_to_one");pairs["actual"]=num(pairs.actual_current);pairs["current_abs_error"]=(num(pairs.pred_current)-pairs.actual).abs();pairs["joint_abs_error"]=(num(pairs.pred_joint)-pairs.actual).abs();pairs["mae_improvement"]=pairs.current_abs_error-pairs.joint_abs_error
    feature_cols=["actual_pass_att","pred_ypa","market_is_underdog","market_total","controlled_environment","qb_recent_games"]
    pairs=pairs.merge(features[keys+feature_cols].drop_duplicates(keys),on=keys,how="left",validate="one_to_one")
    pairs["season_phase"]=pd.cut(num(pairs.week),[0,9,13,18],labels=["W5-9","W10-13","W14-18"])
    pairs["favorite_status"]=np.where(num(pairs.market_is_underdog).eq(1),"underdog","favorite_or_pickem")
    pairs["total_environment"]=np.where(num(pairs.market_total).ge(num(pairs.market_total).median()),"high_total","low_total")
    pairs["actual_volume"]=np.where(num(pairs.actual_pass_att).ge(40),"40plus_attempts","under40_attempts")
    pairs["pregame_qb_tier"]=np.where(num(pairs.pred_ypa).ge(num(pairs.pred_ypa).median()),"higher_pred_ypa","lower_pred_ypa")
    pairs["venue"]=np.where(num(pairs.controlled_environment).eq(1),"controlled","outdoor_unknown_weather")
    pairs["prior_history"]=pd.cut(num(pairs.qb_recent_games),[-1,2,5,8],labels=["0-2_games","3-5_games","6-8_games"])

    subgroup=[]
    for dimension in ["target_season","season_phase","favorite_status","total_environment","actual_volume","pregame_qb_tier","venue","prior_history"]:
        for bucket,g in pairs.groupby(dimension,dropna=False,observed=True):
            if len(g)<10: continue
            cur=metrics(g.actual,g.pred_current);jnt=metrics(g.actual,g.pred_joint)
            subgroup.append({"dimension":dimension,"bucket":str(bucket),"n":len(g),"current_mae":cur["mae"],"joint_mae":jnt["mae"],"mae_improvement":cur["mae"]-jnt["mae"],"current_rmse":cur["rmse"],"joint_rmse":jnt["rmse"],"current_corr":cur["correlation"],"joint_corr":jnt["correlation"],"current_catastrophic":cur["catastrophic_100plus"],"joint_catastrophic":jnt["catastrophic_100plus"]})
    subgroup=pd.DataFrame(subgroup)

    weekly=pairs.groupby(["target_season","week"],as_index=False).mae_improvement.mean();vals=weekly.mae_improvement.to_numpy(float);rng=np.random.default_rng(a.seed);boot=np.empty(a.bootstrap,float)
    for i in range(a.bootstrap):boot[i]=float(np.mean(rng.choice(vals,size=len(vals),replace=True)))
    paired=pd.DataFrame([{"qb_games":len(pairs),"season_weeks":len(vals),"mean_game_improvement":float(pairs.mae_improvement.mean()),"mean_week_improvement":float(vals.mean()),"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),"probability_improvement_gt_zero":float(np.mean(boot>0)),"games_improved":int(pairs.mae_improvement.gt(0).sum()),"games_worsened":int(pairs.mae_improvement.lt(0).sum())}])
    season_consistency=subgroup[subgroup.dimension.eq("target_season")][["bucket","n","current_mae","joint_mae","mae_improvement","current_corr","joint_corr","current_catastrophic","joint_catastrophic"]]
    a.out_dir.mkdir(parents=True,exist_ok=True);summary.to_csv(a.out_dir/"qb_joint_multi_season_summary.csv",index=False);pairs.to_csv(a.out_dir/"qb_joint_multi_season_pairs.csv",index=False);subgroup.to_csv(a.out_dir/"qb_joint_multi_season_subgroups.csv",index=False);weekly.to_csv(a.out_dir/"qb_joint_multi_season_weekly.csv",index=False);paired.to_csv(a.out_dir/"qb_joint_multi_season_bootstrap.csv",index=False)
    print("=== MULTI-SEASON SUMMARY ===");print(summary[(summary.slice.eq("stable_qb"))].to_string(index=False));print("\n=== PAIRED BOOTSTRAP ===");print(paired.to_string(index=False));print("\n=== SEASON CONSISTENCY ===");print(season_consistency.to_string(index=False));print("\n=== SUBGROUPS ===");print(subgroup.to_string(index=False));return 0


if __name__=="__main__":raise SystemExit(main())
