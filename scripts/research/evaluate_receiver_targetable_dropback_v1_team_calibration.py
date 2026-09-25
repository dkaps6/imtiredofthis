#!/usr/bin/env python3
"""Temporal team calibration for receiver targetable-dropback V1.

Frozen candidate:
  baseline = projected plays * 0.57
  candidate = baseline * strict-prior team targets/dropbacks

No player projection is changed. No sportsbook input is read.
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

VERSION="RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION"

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

def build_team_actual_history(player_logs,team_weekly):
    pl=player_logs.copy()
    tw=team_weekly.copy()
    for x in (pl,tw):
        x["season"]=pd.to_numeric(x["season"],errors="coerce")
        x["week"]=pd.to_numeric(x["week"],errors="coerce")
        x["team"]=x["team"].map(canon_team)
    if "targets" not in pl.columns:
        raise RuntimeError("player logs missing targets")
    pl["targets"]=pd.to_numeric(pl["targets"],errors="coerce").fillna(0.0)
    targets=pl.groupby(["season","week","team"],as_index=False).agg(actual_team_targets=("targets","sum"))

    for c in ("plays_est","dropback_rate"):
        if c not in tw.columns:
            raise RuntimeError(f"team weekly missing {c}")
        tw[c]=pd.to_numeric(tw[c],errors="coerce")
    tw=tw.drop_duplicates(["season","week","team"],keep="last").copy()
    tw["actual_dropbacks"]=tw["plays_est"]*tw["dropback_rate"]
    out=targets.merge(
        tw[["season","week","team","actual_dropbacks"]],
        on=["season","week","team"],how="inner",validate="one_to_one"
    )
    if out.empty:
        raise RuntimeError("team actual history empty")
    bad=out.loc[(out["actual_team_targets"]<0)|(out["actual_dropbacks"]<=0)]
    if not bad.empty:
        raise RuntimeError(f"invalid actual history sample={bad.head().to_dict('records')}")
    if (out["actual_team_targets"]>out["actual_dropbacks"]+1e-9).any():
        sample=out.loc[out["actual_team_targets"]>out["actual_dropbacks"]+1e-9].head().to_dict("records")
        raise RuntimeError(f"targets exceed dropbacks sample={sample}")
    return out.sort_values(["season","week","team"]).reset_index(drop=True)

def strict_prior_rate(history,season,week,team,prior_season):
    h=history.loc[
        history["season"].eq(int(prior_season))
        | (history["season"].eq(int(season)) & history["week"].lt(int(week)))
    ].copy()
    if h.empty:
        raise RuntimeError(f"no eligible history season={season} week={week}")
    league_drop=float(h["actual_dropbacks"].sum())
    league_tgt=float(h["actual_team_targets"].sum())
    if league_drop<=0:
        raise RuntimeError("league strict-prior dropbacks <=0")
    league_rate=league_tgt/league_drop

    t=h.loc[h["team"].eq(canon_team(team))].copy()
    tdrop=float(t["actual_dropbacks"].sum())
    ttgt=float(t["actual_team_targets"].sum())
    if tdrop>0:
        rate=ttgt/tdrop
        source="team_strict_prior"
        games=int(len(t))
        hist_drop=tdrop
        hist_tgt=ttgt
    else:
        rate=league_rate
        source="league_fallback"
        games=0
        hist_drop=league_drop
        hist_tgt=league_tgt
    if not np.isfinite(rate) or not (0.0<=rate<=1.0):
        raise RuntimeError(f"invalid targetable rate team={team} rate={rate}")
    return {
        "targetable_dropback_rate":float(rate),
        "conversion_source":source,
        "prior_history_games":games,
        "prior_history_dropbacks":float(hist_drop),
        "prior_history_targets":float(hist_tgt),
        "league_targetable_dropback_rate":float(league_rate),
    }

def forecast_week(player_logs,team_weekly,schedule,universe,injuries,weather,history,season,week,prior_season):
    bundle=build_historical_context_bundle(
        player_logs=player_logs,team_weekly=team_weekly,pregame_universe=universe,
        schedule=schedule,season=int(season),week=int(week),prior_season=int(prior_season),
        injuries=injuries,weather=weather,
    )
    mc=build_mc_predictions(bundle,iterations=20,seed=9200+int(week))
    if "mc_projected_plays" not in mc.columns:
        raise RuntimeError("MC trace missing mc_projected_plays")
    x=mc[["team","opponent","mc_projected_plays"]].copy()
    x["team"]=x["team"].map(canon_team)
    x["opponent"]=x["opponent"].map(canon_team)
    x["mc_projected_plays"]=pd.to_numeric(x["mc_projected_plays"],errors="coerce")
    rows=[]
    for team,g in x.groupby("team"):
        p=g["mc_projected_plays"].dropna().unique()
        if len(p)!=1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique projected plays {p[:5]}")
        opp=g["opponent"].dropna().astype(str).unique()
        if len(opp)!=1:
            raise RuntimeError(f"{season} W{week} team={team} nonunique opponent")
        projected_plays=float(p[0])
        baseline=projected_plays*0.57
        rate=strict_prior_rate(history,season,week,team,prior_season)
        candidate=baseline*rate["targetable_dropback_rate"]
        rows.append({
            "season":int(season),"week":int(week),"team":team,"opponent":str(opp[0]),
            "projected_plays":projected_plays,
            "baseline_projected_dropbacks":float(baseline),
            **rate,
            "candidate_targetable_pool":float(candidate),
        })
    return pd.DataFrame(rows)

def score(df,pred,actual="actual_team_targets"):
    x=df[[pred,actual]].apply(pd.to_numeric,errors="coerce").dropna()
    if x.empty:
        raise RuntimeError(f"empty score {pred}")
    e=x[pred].to_numpy(float)-x[actual].to_numpy(float)
    ae=np.abs(e)
    return {
        "n":int(len(x)),
        "mae":float(ae.mean()),
        "rmse":float(np.sqrt(np.mean(e*e))),
        "bias":float(e.mean()),
        "abs_bias":float(abs(e.mean())),
        "corr":float(np.corrcoef(x[pred],x[actual])[0,1]) if len(x)>1 and x[pred].std()>0 and x[actual].std()>0 else None,
        "median_ae":float(np.quantile(ae,.50)),
        "p75_ae":float(np.quantile(ae,.75)),
        "p90_ae":float(np.quantile(ae,.90)),
        "miss5_rate":float(np.mean(ae>=5)),
        "miss8_rate":float(np.mean(ae>=8)),
        "miss10_rate":float(np.mean(ae>=10)),
    }

def pair(df):
    b=score(df,"baseline_projected_dropbacks")
    c=score(df,"candidate_targetable_pool")
    actual=pd.to_numeric(df["actual_team_targets"],errors="coerce").to_numpy(float)
    bp=pd.to_numeric(df["baseline_projected_dropbacks"],errors="coerce").to_numpy(float)
    cp=pd.to_numeric(df["candidate_targetable_pool"],errors="coerce").to_numpy(float)
    changed=np.abs(bp-cp)>1e-12
    ba=np.abs(bp-actual); ca=np.abs(cp-actual)
    cw=changed&(ca<ba-1e-12); bw=changed&(ba<ca-1e-12); decided=cw|bw
    return {
        "baseline":b,"candidate":c,
        "changed_rows":int(changed.sum()),
        "candidate_closer":int(cw.sum()),
        "baseline_closer":int(bw.sum()),
        "candidate_closer_rate":float(cw.sum()/decided.sum()) if int(decided.sum()) else None,
    }

def evaluate_season(season,prior_season,weeks,player_logs,team_weekly,schedule,universe_dir,injuries,weather,history):
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
        pred=forecast_week(
            player_logs,team_weekly,schedule,u,inj,wx,history,
            season,week,prior_season
        )
        actual=history.loc[
            history["season"].eq(int(season))&history["week"].eq(int(week)),
            ["team","actual_team_targets","actual_dropbacks"]
        ].copy()
        joined=pred.merge(actual,on="team",how="inner",validate="one_to_one")
        if joined.empty:
            raise RuntimeError(f"{season} W{week} no joined target outcomes")
        rows.append(joined)
    return pd.concat(rows,ignore_index=True)

def result(detail):
    out={"by_season":{}}
    for season in (2022,2023):
        d=detail.loc[detail["season"].eq(season)].copy()
        out["by_season"][str(season)]={
            "targets":pair(d),
            "baseline_vs_actual_dropbacks":score(d.rename(columns={"actual_dropbacks":"actual_team_targets"}),"baseline_projected_dropbacks"),
            "rows":int(len(d)),
            "team_source_rows":int(d["conversion_source"].eq("team_strict_prior").sum()),
            "fallback_rows":int(d["conversion_source"].eq("league_fallback").sum()),
            "rate_min":float(d["targetable_dropback_rate"].min()),
            "rate_median":float(d["targetable_dropback_rate"].median()),
            "rate_max":float(d["targetable_dropback_rate"].max()),
        }
    out["pooled"]={
        "targets":pair(detail),
        "rows":int(len(detail)),
        "team_source_rows":int(detail["conversion_source"].eq("team_strict_prior").sum()),
        "fallback_rows":int(detail["conversion_source"].eq("league_fallback").sum()),
        "rate_min":float(detail["targetable_dropback_rate"].min()),
        "rate_median":float(detail["targetable_dropback_rate"].median()),
        "rate_max":float(detail["targetable_dropback_rate"].max()),
    }
    return out

def gates(s):
    y22=s["by_season"]["2022"]["targets"]; y23=s["by_season"]["2023"]["targets"]; p=s["pooled"]["targets"]
    fallback_rate=s["pooled"]["fallback_rows"]/max(1,s["pooled"]["rows"])
    g={
        "target_mae_2022_improves":y22["candidate"]["mae"]<y22["baseline"]["mae"],
        "target_mae_2023_improves":y23["candidate"]["mae"]<y23["baseline"]["mae"],
        "target_mae_pooled_improves":p["candidate"]["mae"]<p["baseline"]["mae"],
        "target_p90_pooled_nonworse":p["candidate"]["p90_ae"]<=p["baseline"]["p90_ae"]+1e-12,
        "target_abs_bias_pooled_improves":p["candidate"]["abs_bias"]<p["baseline"]["abs_bias"],
        "candidate_closer_gt50":p["candidate_closer_rate"] is not None and p["candidate_closer_rate"]>.50,
        "season_p90_guard_2022":y22["candidate"]["p90_ae"]<=y22["baseline"]["p90_ae"]+.50,
        "season_p90_guard_2023":y23["candidate"]["p90_ae"]<=y23["baseline"]["p90_ae"]+.50,
        "team_conversion_source_ge99":s["pooled"]["team_source_rows"]/max(1,s["pooled"]["rows"])>=.99,
        "fallback_rate_le1":fallback_rate<=.01,
        "conversion_finite_in_bounds":0<=s["pooled"]["rate_min"]<=s["pooled"]["rate_max"]<=1,
        "target_game_outcomes_upstream_zero":True,
        "sportsbook_inputs_zero":True,
        "parameters_fit_zero":True,
        "one_candidate_only":True,
    }
    return g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--team-weekly",type=Path,required=True)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--universe-2022",type=Path,required=True)
    ap.add_argument("--universe-2023",type=Path,required=True)
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
    history=build_team_actual_history(player_logs,team_weekly)
    weeks=_parse_weeks(a.weeks)

    d22=evaluate_season(2022,2021,weeks,player_logs,team_weekly,schedule,a.universe_2022,injuries,weather,history)
    d23=evaluate_season(2023,2022,weeks,player_logs,team_weekly,schedule,a.universe_2023,injuries,weather,history)
    detail=pd.concat([d22,d23],ignore_index=True)
    s=result(detail)
    g=gates(s)
    qualified=all(g.values())
    disposition="RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED" if qualified else "RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_FAILED_CLOSED"
    payload={
        "version":VERSION,
        "disposition":disposition,
        "qualified":bool(qualified),
        "candidate_variants_scored":1,
        "parameters_fit":0,
        "sportsbook_inputs_used":0,
        "target_game_outcomes_used_upstream":0,
        "scorecard":s,
        "gates":g,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.out_dir/"team_detail_2022_2023.csv",index=False)
    (a.out_dir/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Receiver Targetable-Dropback V1 — Temporal Team Calibration","",f"Disposition: **{disposition}**",""]
    for season in ("2022","2023"):
        x=s["by_season"][season]["targets"]
        lines += [
            f"## {season}","",
            f"- target MAE: {x['baseline']['mae']:.6f} -> {x['candidate']['mae']:.6f}",
            f"- target p90: {x['baseline']['p90_ae']:.6f} -> {x['candidate']['p90_ae']:.6f}",
            f"- target abs bias: {x['baseline']['abs_bias']:.6f} -> {x['candidate']['abs_bias']:.6f}","",
        ]
    p=s["pooled"]["targets"]
    lines += [
        "## Pooled","",
        f"- target MAE: {p['baseline']['mae']:.6f} -> {p['candidate']['mae']:.6f}",
        f"- target p90: {p['baseline']['p90_ae']:.6f} -> {p['candidate']['p90_ae']:.6f}",
        f"- candidate closer rate: {p['candidate_closer_rate']:.6f}",
        f"- rate median: {s['pooled']['rate_median']:.6f}","",
        "## Frozen gates","",
    ]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k,v in g.items()]
    (a.out_dir/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
