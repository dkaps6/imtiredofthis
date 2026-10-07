#!/usr/bin/env python3
"""Quantify how player-specific the current projection baseline actually is.

Diagnostic only. No candidate fitting or production mutation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import GROUP_STRENGTH, PRIOR_PLAYER_CAP, ALL_METRICS
from scripts.player_form_v2 import _season_totals

VERSION="PLAYER_INDIVIDUALIZATION_AUDIT_V1"
SEASON=2025
PRIOR=2024
WEEKS=tuple(range(2,19))
FORBIDDEN=("sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
           "spread_line","total_line","moneyline","closing_line","no_vig","implied_prob")

def read(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    bad=[c for c in x.columns if any(t in c for t in FORBIDDEN)]
    if bad: raise RuntimeError(f"forbidden sportsbook fields in {label}: {bad}")
    return x

def num(x): return pd.to_numeric(x,errors="coerce")

def pos_group(v):
    p=str(v or "").upper().strip()
    if p in {"WR","LWR","RWR","SWR","WIDE RECEIVER","SLOT WR"}: return "WR"
    if p in {"RB","FB","HB"}: return "RB"
    if p=="TE": return "TE"
    if p=="QB": return "QB"
    return "OTHER"

def normalize_logs(logs:pd.DataFrame)->pd.DataFrame:
    x=logs.copy()
    for c in ("season","week"): x[c]=num(x[c]).astype("Int64")
    if "player_identity_key" not in x.columns:
        x["player_identity_key"]=x.get("player_clean_key","").astype(str)
    if "position" not in x.columns: x["position"]=""
    return x

def metric_weights(logs:pd.DataFrame, universe:pd.DataFrame)->pd.DataFrame:
    logs=normalize_logs(logs)
    prior_logs=logs.loc[logs["season"].eq(PRIOR)].copy()
    prior_tot=_season_totals(prior_logs)
    pmap=prior_tot.set_index("player_identity_key",drop=False) if len(prior_tot) else pd.DataFrame()

    rows=[]
    for week in WEEKS:
        current_logs=logs.loc[logs["season"].eq(SEASON)&logs["week"].lt(week)].copy()
        cur_tot=_season_totals(current_logs)
        cmap=cur_tot.set_index("player_identity_key",drop=False) if len(cur_tot) else pd.DataFrame()
        u=universe.loc[universe["week"].eq(week)].drop_duplicates(["player_identity_key"]).copy()
        for r in u.itertuples(index=False):
            pid=str(r.player_identity_key)
            ppos=pos_group(getattr(r,"position",""))
            pr=pmap.loc[pid] if isinstance(pmap,pd.DataFrame) and pid in pmap.index else None
            cr=cmap.loc[pid] if isinstance(cmap,pd.DataFrame) and pid in cmap.index else None
            if isinstance(pr,pd.DataFrame): pr=pr.iloc[-1]
            if isinstance(cr,pd.DataFrame): cr=cr.iloc[-1]
            pg=float(pr["games"]) if pr is not None and pd.notna(pr.get("games")) else 0.0
            cg=float(cr["games"]) if cr is not None and pd.notna(cr.get("games")) else 0.0
            for metric in ALL_METRICS:
                g=float(GROUP_STRENGTH[metric])
                pv=float(pr.get(metric)) if pr is not None and pd.notna(pr.get(metric)) else np.nan
                cv=float(cr.get(metric)) if cr is not None and pd.notna(cr.get(metric)) else np.nan
                pw=min(pg,float(PRIOR_PLAYER_CAP[metric])) if np.isfinite(pv) and pg>0 else 0.0
                cw=cg if np.isfinite(cv) and cg>0 else 0.0
                den=g+pw+cw
                rows.append({
                    "season":SEASON,"week":week,"player_identity_key":pid,
                    "player":getattr(r,"player",""),"team":getattr(r,"team",""),
                    "position_group":ppos,"metric":metric,
                    "prior_games":pg,"current_games":cg,
                    "group_weight":g/den,"prior_player_weight":pw/den,
                    "current_player_weight":cw/den,
                    "player_specific_weight":(pw+cw)/den,
                    "prior_metric_available":bool(np.isfinite(pv)),
                    "current_metric_available":bool(np.isfinite(cv)),
                })
    return pd.DataFrame(rows)

def build_universe(logs:pd.DataFrame, baseline:pd.DataFrame)->pd.DataFrame:
    l=normalize_logs(logs)
    tgt=l.loc[l["season"].eq(SEASON)&l["week"].isin(WEEKS)].copy()
    keep=["week","player_identity_key","player","team","position"]
    t=tgt[keep].drop_duplicates(["week","player_identity_key"],keep="last")
    if len(t): return t
    b=baseline.copy()
    b["season"]=num(b["season"]); b["week"]=num(b["week"])
    if "player_identity_key" not in b.columns:
        b["player_identity_key"]=b.get("player_clean_key","").astype(str)
    for c in ("player","team","position"):
        if c not in b.columns: b[c]=""
    return b.loc[b["season"].eq(SEASON)&b["week"].isin(WEEKS),
                 ["week","player_identity_key","player","team","position"]].drop_duplicates(["week","player_identity_key"])

def summarize_weights(w:pd.DataFrame):
    s=(w.groupby(["week","position_group","metric"],dropna=False)
       .agg(rows=("player_identity_key","size"),
            players=("player_identity_key","nunique"),
            median_group_weight=("group_weight","median"),
            p90_group_weight=("group_weight",lambda x: float(x.quantile(.90))),
            median_player_specific_weight=("player_specific_weight","median"),
            p10_player_specific_weight=("player_specific_weight",lambda x: float(x.quantile(.10))),
            median_current_player_weight=("current_player_weight","median"))
       .reset_index())
    return s

def threshold_table():
    rows=[]
    for metric in ALL_METRICS:
        g=float(GROUP_STRENGTH[metric]); cap=float(PRIOR_PLAYER_CAP[metric])
        # Established player assumes full available prior cap.
        for cg in range(0,18):
            den=g+cap+cg
            gw=g/den; pw=(cap+cg)/den; cw=cg/den
            rows.append({"metric":metric,"current_games":cg,
                         "established_group_share":gw,
                         "established_total_player_share":pw,
                         "established_current_share":cw})
    x=pd.DataFrame(rows)
    out=[]
    for metric,g in x.groupby("metric"):
        def first(mask):
            z=g.loc[mask,"current_games"]
            return int(z.min()) if len(z) else None
        out.append({
            "metric":metric,
            "first_current_games_player_specific_gt_50pct":first(g["established_total_player_share"]>.5),
            "first_current_games_current_alone_gt_group":first(g["established_current_share"]>g["established_group_share"]),
            "first_current_games_group_below_25pct":first(g["established_group_share"]<.25),
            "first_current_games_group_below_10pct":first(g["established_group_share"]<.10),
        })
    return pd.DataFrame(out),x

def error_heterogeneity(baseline:pd.DataFrame,logs:pd.DataFrame):
    b=baseline.copy()
    for c in ("season","week"): b[c]=num(b[c])
    b=b.loc[b["season"].eq(SEASON)&b["week"].isin(WEEKS)].copy()
    proj_col="ensemble_proj" if "ensemble_proj" in b.columns else ("final_mean" if "final_mean" in b.columns else None)
    if proj_col is None: raise RuntimeError("baseline missing ensemble/final mean")
    b["projection"]=num(b[proj_col]); b["actual"]=num(b["actual"])
    if "position" not in b.columns or b["position"].isna().all():
        l=normalize_logs(logs)
        k=l.loc[l["season"].eq(SEASON),["season","week","team","player_clean_key","position"]].drop_duplicates()
        b=b.merge(k,on=["season","week","team","player_clean_key"],how="left",suffixes=("","_log"))
        b["position"]=b.get("position_log",b.get("position",""))
    b["position_group"]=b["position"].map(pos_group)
    b["error"]=b["projection"]-b["actual"]
    b["ae"]=b["error"].abs()
    b=b.dropna(subset=["projection","actual"])
    per=(b.groupby(["position_group","market","player_clean_key"],dropna=False)
         .agg(games=("week","size"),player_bias=("error","mean"),player_mae=("ae","mean"))
         .reset_index())
    per=per.loc[per["games"].ge(8)].copy()
    agg=(b.groupby(["position_group","market"],dropna=False)
         .agg(rows=("ae","size"),players=("player_clean_key","nunique"),
              aggregate_bias=("error","mean"),aggregate_mae=("ae","mean"))
         .reset_index())
    out=per.merge(agg,on=["position_group","market"],how="left")
    out["bias_direction_differs_from_aggregate"]=(
        np.sign(out["player_bias"]).ne(np.sign(out["aggregate_bias"]))
        & out["player_bias"].ne(0) & out["aggregate_bias"].ne(0)
    )
    summary=(out.groupby(["position_group","market"],dropna=False)
             .agg(qualified_players=("player_clean_key","size"),
                  aggregate_mae=("aggregate_mae","first"),
                  median_player_mae=("player_mae","median"),
                  p10_player_mae=("player_mae",lambda x: float(x.quantile(.10))),
                  p90_player_mae=("player_mae",lambda x: float(x.quantile(.90))),
                  share_players_opposite_bias=("bias_direction_differs_from_aggregate","mean"))
             .reset_index())
    return out,summary

def architecture_map():
    rows=[
      ("historical_identity","PLAYER_SPECIFIC","stable player_identity_key","YES"),
      ("target_share_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("rush_share_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("catch_rate_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("ypt_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("ypc_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("ypa_baseline","PLAYER_SPECIFIC_PLUS_POSITION_POOL","player prior/current + position empirical-Bayes prior","PARTIAL"),
      ("team_plays_pass_rush","TEAM_SPECIFIC","same team game-script context shared by teammates","NO"),
      ("target_matchup_multiplier","ROLE_OR_POSITION_BUCKET","WR1/WR1.5/SLOT/TE/RB receiving bucket","NO"),
      ("receiving_eff_matchup_multiplier","OPPONENT_TEAM_SPECIFIC","same pass_eff_mult applied to teammate receiving efficiency","NO"),
      ("rushing_eff_matchup_multiplier","OPPONENT_TEAM_SPECIFIC","same rb_rush_eff_mult applied to teammate rush efficiency","NO"),
      ("injury_vacancy_redistribution","PLAYER_X_TEAM_AVAILABILITY","redistribution uses individual posterior shares inside team vacancy event","YES"),
      ("volatility_matchup_multiplier","OPPONENT_TEAM_SPECIFIC","shared matchup volatility multiplier","NO"),
    ]
    return pd.DataFrame(rows,columns=["component","classification","current_behavior","true_player_environment_interaction"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--player-logs",type=Path,required=True)
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    logs=read(a.player_logs,"player logs")
    baseline=read(a.baseline,"football baseline")
    universe=build_universe(logs,baseline)
    weights=metric_weights(logs,universe)
    summary=summarize_weights(weights)
    thresh,curve=threshold_table()
    perr,esum=error_heterogeneity(baseline,logs)
    amap=architecture_map()

    established=weights.loc[weights["prior_games"].ge(8)].copy()
    w5=established.loc[established["week"].eq(5)]
    key_summary=(w5.groupby(["position_group","metric"])
                 .agg(players=("player_identity_key","nunique"),
                      median_group_weight=("group_weight","median"),
                      median_player_specific_weight=("player_specific_weight","median"),
                      median_current_weight=("current_player_weight","median"))
                 .reset_index())
    true_interactions=int(amap["true_player_environment_interaction"].eq("YES").sum())
    shared_env=int(amap["classification"].isin(["ROLE_OR_POSITION_BUCKET","OPPONENT_TEAM_SPECIFIC","TEAM_SPECIFIC"]).sum())
    # Disposition is architecture-based, not predictive.
    # Individual identity/history exists, but most environment transformations are shared/bucketed.
    disposition="PLAYER_INDIVIDUALIZATION_PARTIAL"

    result={
      "version":VERSION,
      "disposition":disposition,
      "audit_season":SEASON,
      "weeks":[2,18],
      "player_week_metric_rows":int(len(weights)),
      "distinct_players":int(weights["player_identity_key"].nunique()),
      "architecture_true_player_environment_interactions":true_interactions,
      "architecture_shared_team_role_position_transforms":shared_env,
      "established_week5_weight_summary":key_summary.to_dict("records"),
      "evaluation_position_market_summary":esum.to_dict("records"),
      "sportsbook_inputs_used":0,
      "candidate_models_fit":0,
      "production_changed":False,
      "outcomes_2026_read":0,
    }
    a.out_dir.mkdir(parents=True,exist_ok=True)
    weights.to_csv(a.out_dir/"player_individualization_weight_trace.csv",index=False)
    summary.to_csv(a.out_dir/"player_individualization_weight_summary.csv",index=False)
    thresh.to_csv(a.out_dir/"player_individualization_mechanical_thresholds.csv",index=False)
    curve.to_csv(a.out_dir/"player_individualization_weight_curve.csv",index=False)
    perr.to_csv(a.out_dir/"player_individualization_per_player_error.csv",index=False)
    esum.to_csv(a.out_dir/"player_individualization_error_summary.csv",index=False)
    amap.to_csv(a.out_dir/"player_individualization_architecture_map.csv",index=False)
    (a.out_dir/"player_individualization_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    print("\nMECHANICAL THRESHOLDS")
    print(thresh.to_string(index=False))
    print("\nARCHITECTURE")
    print(amap.to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
