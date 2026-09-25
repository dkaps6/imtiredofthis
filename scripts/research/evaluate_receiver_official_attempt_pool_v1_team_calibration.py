#!/usr/bin/env python3
"""Historical team-level calibration for receiver official-attempt pool V1.

This evaluator scores exactly one frozen opportunity candidate:
    candidate = projected_dropbacks * strict_prior_pass_attempts_per_dropback

No player projection is modified. No sportsbook data is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _parse_weeks

VERSION="RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION"

def read(path,label):
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    if x.empty:
        raise RuntimeError(f"empty {label}: {path}")
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def optional(path):
    if not path.exists() or path.stat().st_size<=0:
        return pd.DataFrame()
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def score(df,pred_col,actual_col):
    x=df[[pred_col,actual_col]].apply(pd.to_numeric,errors="coerce").dropna()
    if x.empty:
        raise RuntimeError(f"empty score {pred_col} vs {actual_col}")
    e=x[pred_col].to_numpy(float)-x[actual_col].to_numpy(float)
    ae=np.abs(e)
    return {
        "n":int(len(x)),
        "mae":float(ae.mean()),
        "rmse":float(np.sqrt(np.mean(e*e))),
        "bias":float(e.mean()),
        "abs_bias":float(abs(e.mean())),
        "corr":float(np.corrcoef(x[pred_col],x[actual_col])[0,1]) if len(x)>1 and x[pred_col].std()>0 and x[actual_col].std()>0 else None,
        "median_ae":float(np.quantile(ae,.50)),
        "p75_ae":float(np.quantile(ae,.75)),
        "p90_ae":float(np.quantile(ae,.90)),
        "miss5_rate":float(np.mean(ae>=5)),
        "miss8_rate":float(np.mean(ae>=8)),
        "miss10_rate":float(np.mean(ae>=10)),
    }

def pair_score(df,actual_col):
    b=score(df,"baseline_dropbacks",actual_col)
    c=score(df,"candidate_official_attempts",actual_col)
    actual=pd.to_numeric(df[actual_col],errors="coerce").to_numpy(float)
    bp=pd.to_numeric(df["baseline_dropbacks"],errors="coerce").to_numpy(float)
    cp=pd.to_numeric(df["candidate_official_attempts"],errors="coerce").to_numpy(float)
    changed=np.abs(bp-cp)>1e-12
    ba=np.abs(bp-actual); ca=np.abs(cp-actual)
    cand=changed&(ca<ba-1e-12); base=changed&(ba<ca-1e-12); decided=cand|base
    return {
        "baseline":b,
        "candidate":c,
        "changed_rows":int(changed.sum()),
        "candidate_closer":int(cand.sum()),
        "baseline_closer":int(base.sum()),
        "candidate_closer_rate":float(cand.sum()/decided.sum()) if int(decided.sum()) else None,
    }

def actual_team_outcomes(player_logs,team_weekly,season,week):
    pl=player_logs.copy()
    pl["season"]=pd.to_numeric(pl["season"],errors="coerce")
    pl["week"]=pd.to_numeric(pl["week"],errors="coerce")
    pl["team"]=pl["team"].map(canon_team)
    pl=pl.loc[pl["season"].eq(int(season))&pl["week"].eq(int(week))].copy()
    if pl.empty:
        raise RuntimeError(f"no player actuals {season} W{week}")
    for c in ("pass_att","targets"):
        if c not in pl.columns:
            raise RuntimeError(f"player logs missing {c}")
        pl[c]=pd.to_numeric(pl[c],errors="coerce").fillna(0.0)
    p=pl.groupby("team",as_index=False).agg(
        actual_official_pass_attempts=("pass_att","sum"),
        actual_team_targets=("targets","sum"),
    )

    tw=team_weekly.copy()
    tw["season"]=pd.to_numeric(tw["season"],errors="coerce")
    tw["week"]=pd.to_numeric(tw["week"],errors="coerce")
    tw["team"]=tw["team"].map(canon_team)
    tw=tw.loc[tw["season"].eq(int(season))&tw["week"].eq(int(week))].copy()
    if tw.empty:
        raise RuntimeError(f"no team actuals {season} W{week}")
    for c in ("plays_est","dropback_rate"):
        if c not in tw.columns:
            raise RuntimeError(f"team weekly missing {c}")
        tw[c]=pd.to_numeric(tw[c],errors="coerce")
    tw=tw.drop_duplicates("team",keep="last")
    tw["actual_dropbacks"]=tw["plays_est"]*tw["dropback_rate"]
    out=p.merge(tw[["team","actual_dropbacks"]],on="team",how="inner",validate="one_to_one")
    if out.empty:
        raise RuntimeError(f"no merged actuals {season} W{week}")
    if (out["actual_team_targets"]-out["actual_official_pass_attempts"]>1e-9).any():
        sample=out.loc[out["actual_team_targets"]>out["actual_official_pass_attempts"]].head().to_dict("records")
        raise RuntimeError(f"team targets exceed official attempts: {sample}")
    return out

def forecast_week(player_logs,team_weekly,schedule,universe,injuries,weather,season,week,prior_season):
    bundle=build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=int(season),
        week=int(week),
        prior_season=int(prior_season),
        injuries=injuries,
        weather=weather,
    )
    metrics=build_mc_predictions(bundle,iterations=20,seed=42+int(week))
    need={"team","mc_projected_plays","mc_dropback_rate","mc_pass_attempts_per_dropback","mc_pass_attempt_rate_source"}
    missing=need-set(metrics.columns)
    if missing:
        raise RuntimeError(f"MC trace missing columns {sorted(missing)}")
    x=metrics[list(need)].copy()
    x["team"]=x["team"].map(canon_team)
    for c in ("mc_projected_plays","mc_dropback_rate","mc_pass_attempts_per_dropback"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    # All player-market rows for one team must carry one team opportunity state.
    rows=[]
    for team,g in x.groupby("team"):
        vals={}
        for c in ("mc_projected_plays","mc_dropback_rate","mc_pass_attempts_per_dropback"):
            u=g[c].dropna().unique()
            if len(u)!=1:
                raise RuntimeError(f"{season} W{week} team={team} nonunique {c}: {u[:5]}")
            vals[c]=float(u[0])
        srcs=g["mc_pass_attempt_rate_source"].astype(str).unique()
        if len(srcs)!=1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique conversion source")
        baseline=vals["mc_projected_plays"]*vals["mc_dropback_rate"]
        changed=str(srcs[0])=="historical_pregame_pbp"
        candidate=baseline*vals["mc_pass_attempts_per_dropback"] if changed else baseline
        rows.append({
            "season":int(season),"week":int(week),"team":team,
            "projected_plays":vals["mc_projected_plays"],
            "projected_dropback_rate":vals["mc_dropback_rate"],
            "baseline_dropbacks":float(baseline),
            "pass_attempts_per_dropback":vals["mc_pass_attempts_per_dropback"],
            "conversion_source":str(srcs[0]),
            "candidate_official_attempts":float(candidate),
            "candidate_changed":int(changed and abs(candidate-baseline)>1e-12),
        })
    return pd.DataFrame(rows)

def evaluate_season(season,prior_season,weeks,player_logs,team_weekly,schedule,universe_dir,injuries,weather):
    rows=[]
    for week in weeks:
        u=read(universe_dir/f"{season}_week_{int(week):02d}.csv",f"{season} W{week} universe")
        inj=injuries.copy()
        wx=weather.copy()
        if not inj.empty and {"season","week"}.issubset(inj.columns):
            inj=inj.loc[
                pd.to_numeric(inj["season"],errors="coerce").eq(int(season))
                & pd.to_numeric(inj["week"],errors="coerce").lt(int(week))
            ].copy()
        if not wx.empty and {"season","week"}.issubset(wx.columns):
            wx=wx.loc[
                pd.to_numeric(wx["season"],errors="coerce").eq(int(season))
                & pd.to_numeric(wx["week"],errors="coerce").lt(int(week))
            ].copy()
        pred=forecast_week(player_logs,team_weekly,schedule,u,inj,wx,season,week,prior_season)
        actual=actual_team_outcomes(player_logs,team_weekly,season,week)
        joined=pred.merge(actual,on="team",how="inner",validate="one_to_one")
        if joined.empty:
            raise RuntimeError(f"{season} W{week} no scored teams")
        rows.append(joined)
    return pd.concat(rows,ignore_index=True)

def build_result(detail):
    result={"by_season":{}}
    for season in (2024,2025):
        d=detail.loc[detail["season"].eq(season)].copy()
        result["by_season"][str(season)]={
            "official_attempts":pair_score(d,"actual_official_pass_attempts"),
            "team_targets":pair_score(d,"actual_team_targets"),
            "dropback_semantic":score(d,"baseline_dropbacks","actual_dropbacks"),
            "rows":int(len(d)),
            "strict_prior_changed_rows":int(d["candidate_changed"].sum()),
            "fallback_rows":int((d["conversion_source"]!="historical_pregame_pbp").sum()),
        }
    result["pooled"]={
        "official_attempts":pair_score(detail,"actual_official_pass_attempts"),
        "team_targets":pair_score(detail,"actual_team_targets"),
        "dropback_semantic":score(detail,"baseline_dropbacks","actual_dropbacks"),
        "rows":int(len(detail)),
        "strict_prior_changed_rows":int(detail["candidate_changed"].sum()),
        "fallback_rows":int((detail["conversion_source"]!="historical_pregame_pbp").sum()),
    }
    return result

def gates(score):
    a24=score["by_season"]["2024"]["official_attempts"]; a25=score["by_season"]["2025"]["official_attempts"]; ap=score["pooled"]["official_attempts"]
    t24=score["by_season"]["2024"]["team_targets"]; t25=score["by_season"]["2025"]["team_targets"]; tp=score["pooled"]["team_targets"]
    g={
        "attempt_mae_2024_improves":a24["candidate"]["mae"]<a24["baseline"]["mae"],
        "attempt_mae_2025_improves":a25["candidate"]["mae"]<a25["baseline"]["mae"],
        "attempt_mae_pooled_improves":ap["candidate"]["mae"]<ap["baseline"]["mae"],
        "attempt_p90_pooled_nonworse":ap["candidate"]["p90_ae"]<=ap["baseline"]["p90_ae"]+1e-12,
        "attempt_abs_bias_pooled_improves":ap["candidate"]["abs_bias"]<ap["baseline"]["abs_bias"],
        "attempt_candidate_closer_gt50":ap["candidate_closer_rate"] is not None and ap["candidate_closer_rate"]>.50,
        "targets_mae_2024_improves":t24["candidate"]["mae"]<t24["baseline"]["mae"],
        "targets_mae_2025_improves":t25["candidate"]["mae"]<t25["baseline"]["mae"],
        "targets_mae_pooled_improves":tp["candidate"]["mae"]<tp["baseline"]["mae"],
        "targets_p90_pooled_nonworse":tp["candidate"]["p90_ae"]<=tp["baseline"]["p90_ae"]+1e-12,
        "targets_abs_bias_pooled_improves":tp["candidate"]["abs_bias"]<tp["baseline"]["abs_bias"],
        "targets_candidate_closer_gt50":tp["candidate_closer_rate"] is not None and tp["candidate_closer_rate"]>.50,
        "target_game_outcomes_upstream_zero":True,
        "sportsbook_inputs_zero":True,
        "one_candidate_only":True,
        "parameters_fit_zero":True,
        "strict_prior_provenance_for_changed_rows":bool(
            (score["pooled"]["fallback_rows"]+score["pooled"]["strict_prior_changed_rows"])==score["pooled"]["rows"]
        ),
    }
    return g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-2024",type=Path,required=True)
    ap.add_argument("--universe-2025",type=Path,required=True)
    ap.add_argument("--injuries",type=Path,required=True)
    ap.add_argument("--weather",type=Path,required=True)
    ap.add_argument("--weeks",default="1-18")
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    player_logs=read(a.player_logs,"player logs")
    team_weekly=read(a.team_weekly,"team weekly")
    schedule=read(a.schedule,"schedule")
    injuries=optional(a.injuries)
    weather=optional(a.weather)
    weeks=_parse_weeks(a.weeks)

    d24=evaluate_season(2024,2023,weeks,player_logs,team_weekly,schedule,a.universe_2024,injuries,weather)
    d25=evaluate_season(2025,2024,weeks,player_logs,team_weekly,schedule,a.universe_2025,injuries,weather)
    detail=pd.concat([d24,d25],ignore_index=True)
    score=build_result(detail)
    g=gates(score)
    qualified=all(g.values())
    disposition="RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_SUPPORTED" if qualified else "RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_FAILED_CLOSED"
    payload={
        "version":VERSION,
        "disposition":disposition,
        "qualified":bool(qualified),
        "parameters_fit":0,
        "candidate_variants_scored":1,
        "sportsbook_inputs_used":0,
        "target_game_outcomes_used_upstream":0,
        "scorecard":score,
        "gates":g,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.out_dir/"team_detail_2024_2025.csv",index=False)
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=[
        "# Receiver Official-Attempt Pool V1 — Team Calibration Result","",
        f"Disposition: **{disposition}**","",
    ]
    for scope in ("2024","2025"):
        s=score["by_season"][scope]
        lines += [
            f"## {scope}","",
            f"- attempts MAE: {s['official_attempts']['baseline']['mae']:.6f} -> {s['official_attempts']['candidate']['mae']:.6f}",
            f"- targets MAE: {s['team_targets']['baseline']['mae']:.6f} -> {s['team_targets']['candidate']['mae']:.6f}",
            f"- dropback baseline MAE vs actual dropbacks: {s['dropback_semantic']['mae']:.6f}","",
        ]
    p=score["pooled"]
    lines += [
        "## Pooled","",
        f"- attempts MAE: {p['official_attempts']['baseline']['mae']:.6f} -> {p['official_attempts']['candidate']['mae']:.6f}",
        f"- attempts p90: {p['official_attempts']['baseline']['p90_ae']:.6f} -> {p['official_attempts']['candidate']['p90_ae']:.6f}",
        f"- targets MAE: {p['team_targets']['baseline']['mae']:.6f} -> {p['team_targets']['candidate']['mae']:.6f}",
        f"- targets p90: {p['team_targets']['baseline']['p90_ae']:.6f} -> {p['team_targets']['candidate']['p90_ae']:.6f}","",
        "## Frozen gates","",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in g.items()]
    (a.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
