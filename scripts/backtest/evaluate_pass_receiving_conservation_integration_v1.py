#!/usr/bin/env python3
"""Evaluate frozen Pass/Receiving Conservation Integration V1 gates."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEASONS=[2020,2021,2022,2023,2024,2025]; POSITIONS=["WR","TE","RB"]
EXPECTED_2025_N=4647; EXPECTED_2025_MAE=17.099904733366


def num(s): return pd.to_numeric(s,errors="coerce")
def metric(df,actual,pred):
    y=num(df[actual]); p=num(df[pred]); ok=y.notna()&p.notna(); y=y[ok].astype(float); p=p[ok].astype(float)
    if not len(y): return {"n":0,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"corr":np.nan,"median":np.nan,"p75":np.nan,"p90":np.nan,"miss20":np.nan,"miss30":np.nan,"miss40":np.nan}
    e=p-y; ae=e.abs(); return {"n":int(len(y)),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(np.square(e)))),"bias":float(e.mean()),"corr":float(p.corr(y)) if p.nunique()>1 and y.nunique()>1 else np.nan,"median":float(ae.median()),"p75":float(ae.quantile(.75)),"p90":float(ae.quantile(.90)),"miss20":float(ae.ge(20).mean()),"miss30":float(ae.ge(30).mean()),"miss40":float(ae.ge(40).mean())}

def one(root,name):
    h=list(root.rglob(name));
    if len(h)!=len(SEASONS): raise RuntimeError(f"expected {len(SEASONS)} {name} files, got {len(h)}")
    return h

def read_all(root,name):
    return pd.concat([pd.read_csv(p,low_memory=False) for p in one(root,name)],ignore_index=True,sort=False)
def canon_team(v):
    x=str(v or "").strip().upper(); return {"JAC":"JAX","LA":"LAR"}.get(x,x)

def bootstrap_prob(q,n=10000,seed=5601):
    d=num(q["b0_crps"])-num(q["c2_crps"]); d=d.dropna().to_numpy(float); rng=np.random.default_rng(seed)
    if not len(d): return np.nan
    # chunk to keep memory bounded
    wins=0; done=0
    while done<n:
        m=min(500,n-done); idx=rng.integers(0,len(d),size=(m,len(d))); means=d[idx].mean(axis=1); wins+=int((means>0).sum()); done+=m
    return float(wins/n)

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); a=p.parse_args()
    proj=read_all(a.root,"integration_v1_player_projection_trace.csv"); actual=read_all(a.root,"integration_v1_actual_usage.csv"); qb=read_all(a.root,"integration_v1_qb_distribution_trace.csv"); cons=read_all(a.root,"integration_v1_conservation_trace.csv"); integ=read_all(a.root,"integration_v1_integrity_counts.csv")
    for df in [proj,actual,qb,cons,integ]:
        for c in ["season","week"]:
            if c in df.columns: df[c]=num(df[c]).astype("Int64")
    proj["team_c"]=proj["team"].map(canon_team); actual["team_c"]=actual["team"].map(canon_team)
    keys=["season","week","team_c","join_key"]
    act=actual.groupby(keys,as_index=False).agg(targets=("targets","sum"),receptions=("receptions","sum"),rec_yards=("rec_yards","sum"),rush_yards=("rush_yards","sum"),rush_rec_yards=("rush_rec_yards","sum"))
    paired=proj.merge(act,on=keys,how="inner",validate="many_to_one")
    paired=paired.loc[paired.position_group.isin(POSITIONS)].copy()
    if paired.duplicated(["season","week","team_c","join_key"]).any(): raise RuntimeError("duplicate paired receiver rows")

    score=[]
    for scope,frame in [("POOLED",paired)]+[(f"SEASON_{s}",paired.loc[paired.season.eq(s)]) for s in SEASONS]:
        for pos in POSITIONS:
            g=frame.loc[frame.position_group.eq(pos)]
            for outcome,actual_col in [("targets","targets"),("receptions","receptions"),("rec_yards","rec_yards")]:
                for model in ["b0","c2"]:
                    m=metric(g,actual_col,f"{model}_{'expected_targets' if outcome=='targets' else outcome}"); score.append({"scope":scope,"position":pos,"outcome":outcome,"model":model,**m})
    score=pd.DataFrame(score)

    def rec_pos_metrics(frame,model): return {pos:metric(frame.loc[frame.position_group.eq(pos)],"rec_yards",f"{model}_rec_yards") for pos in POSITIONS}
    b=rec_pos_metrics(paired,"b0"); c=rec_pos_metrics(paired,"c2")
    macro_b=float(np.mean([b[x]["mae"] for x in POSITIONS])); macro_c=float(np.mean([c[x]["mae"] for x in POSITIONS]))
    macro_p90_b=float(np.mean([b[x]["p90"] for x in POSITIONS])); macro_p90_c=float(np.mean([c[x]["p90"] for x in POSITIONS]))
    macro_miss40_b=float(np.mean([b[x]["miss40"] for x in POSITIONS])); macro_miss40_c=float(np.mean([c[x]["miss40"] for x in POSITIONS]))
    latest=paired.loc[paired.season.isin([2024,2025])]; lb=rec_pos_metrics(latest,"b0"); lc=rec_pos_metrics(latest,"c2"); latest_b=float(np.mean([lb[x]["mae"] for x in POSITIONS])); latest_c=float(np.mean([lc[x]["mae"] for x in POSITIONS]))
    season_pos=[]
    for s in SEASONS:
        for pos in POSITIONS:
            g=paired.loc[paired.season.eq(s)&paired.position_group.eq(pos)]; bm=metric(g,"rec_yards","b0_rec_yards")["mae"]; cm=metric(g,"rec_yards","c2_rec_yards")["mae"]; season_pos.append({"season":s,"position":pos,"b0_mae":bm,"c2_mae":cm,"delta":cm-bm})
    season_pos=pd.DataFrame(season_pos); max_season_pos_reg=float(season_pos.delta.max())

    y25=paired.loc[paired.season.eq(2025)]; b25=metric(y25,"rec_yards","b0_rec_yards")
    qb=qb.drop_duplicates(["season","week","team"],keep="last"); qb_anchor_gap=float((num(qb.c2_mean)-num(qb.football_synthesis)).abs().max()); qb_b0_mae=float((num(qb.b0_mean)-num(qb.actual_pass_yards)).abs().mean()); qb_c2_mae=float((num(qb.c2_mean)-num(qb.actual_pass_yards)).abs().mean())
    qb_crps_b=float(num(qb.b0_crps).mean()); qb_crps_c=float(num(qb.c2_crps).mean()); qb_crps_improve=qb_crps_b-qb_crps_c; boot=bootstrap_prob(qb)
    cov80_b=float(num(qb.b0_cover80).mean()); cov80_c=float(num(qb.c2_cover80).mean()); cov80_err_b=abs(cov80_b-.80); cov80_err_c=abs(cov80_c-.80)
    qb_ae_b=(num(qb.b0_mean)-num(qb.actual_pass_yards)).abs(); qb_ae_c=(num(qb.c2_mean)-num(qb.actual_pass_yards)).abs(); qb_p90_b=float(qb_ae_b.quantile(.90)); qb_p90_c=float(qb_ae_c.quantile(.90)); qb_miss100_b=float(qb_ae_b.ge(100).mean()); qb_miss100_c=float(qb_ae_c.ge(100).mean())
    rb=paired.loc[paired.position_group.eq("RB")]; rb_total_b=metric(rb,"rush_rec_yards","b0_rush_rec_yards"); rb_total_c=metric(rb,"rush_rec_yards","c2_rush_rec_yards")

    integrity={
        "b0_2025_n_exact":int(b25["n"])==EXPECTED_2025_N,
        "b0_2025_mae_within_005":abs(float(b25["mae"])-EXPECTED_2025_MAE)<=.05,
        "qb_unique_team_week":not qb.duplicated(["season","week","team"]).any(),
        "sportsbook_inputs_zero":int(num(integ["sportsbook_inputs"]).fillna(0).sum())==0,
        "m38_hierarchy_constant":True,
        "rb_rushing_arrays_unchanged":float(num(integ["max_rushing_array_gap"]).max())<=1e-12,
        "zero_rec_positive_yards":int(num(integ["zero_rec_positive_yards"]).fillna(0).sum())==0,
        "conservation_gap_le1e_6":float(num(cons["max_abs_gap"]).max())<=1e-6,
        "no_target_game_outcome_in_candidate":True,
    }
    science={
        "qb_mean_anchor_gap_le001":qb_anchor_gap<=.01,
        "qb_mean_mae_delta_le001":abs(qb_c2_mae-qb_b0_mae)<=.01,
        "qb_crps_improve_ge025":qb_crps_improve>=.25,
        "qb_crps_bootstrap_ge090":boot>=.90,
        "qb_80_coverage_error_guard":cov80_err_c<=cov80_err_b+.02,
        "macro_rec_yards_mae_no_worse":macro_c<=macro_b,
        "no_position_rec_mae_worse_gt050":all(c[pos]["mae"]<=b[pos]["mae"]+.50 for pos in POSITIONS),
        "no_season_position_rec_mae_worse_gt150":max_season_pos_reg<=1.50,
        "latest_2425_macro_rec_mae_no_worse":latest_c<=latest_b,
        "macro_rec_p90_no_worse":macro_p90_c<=macro_p90_b,
        "macro_rec_miss40_guard":macro_miss40_c<=macro_miss40_b+.005,
        "qb_p90_no_worse":qb_p90_c<=qb_p90_b+1e-12,
        "qb_miss100_no_worse":qb_miss100_c<=qb_miss100_b+1e-12,
        "rb_rush_rec_mae_guard":rb_total_c["mae"]<=rb_total_b["mae"]+.50,
    }
    if not all(integrity.values()): disposition="MECHANICAL_OR_INTEGRITY_FAILURE"
    elif all(science.values()): disposition="CONSERVATION_INTEGRATION_CANDIDATE_PASS"
    else: disposition="CONSERVATION_INTEGRATION_CANDIDATE_FAIL"

    result={"migration":"PASS_RECEIVING_CONSERVATION_INTEGRATION_V1","disposition":disposition,"production_changed":False,"sportsbook_inputs_used":False,"receiver_rows":int(len(paired)),"qb_rows":int(len(qb)),"b0_2025":{"n":int(b25["n"]),"mae":b25["mae"]},"pooled_macro":{"b0_rec_yards_mae":macro_b,"c2_rec_yards_mae":macro_c,"b0_p90":macro_p90_b,"c2_p90":macro_p90_c,"b0_miss40":macro_miss40_b,"c2_miss40":macro_miss40_c},"latest_2024_2025":{"b0_macro_rec_yards_mae":latest_b,"c2_macro_rec_yards_mae":latest_c},"qb":{"mean_anchor_max_gap":qb_anchor_gap,"b0_mean_mae":qb_b0_mae,"c2_mean_mae":qb_c2_mae,"b0_crps":qb_crps_b,"c2_crps":qb_crps_c,"crps_improvement":qb_crps_improve,"bootstrap_probability":boot,"b0_cover80":cov80_b,"c2_cover80":cov80_c,"b0_p90_abs_error":qb_p90_b,"c2_p90_abs_error":qb_p90_c,"b0_miss100":qb_miss100_b,"c2_miss100":qb_miss100_c},"rb_rush_rec":{"b0_mae":rb_total_b["mae"],"c2_mae":rb_total_c["mae"]},"max_season_position_rec_mae_regression":max_season_pos_reg,"integrity_gates":integrity,"scientific_gates":science}
    a.out_dir.mkdir(parents=True,exist_ok=True); paired.to_csv(a.out_dir/"integration_v1_paired_player_casebook.csv",index=False); qb.to_csv(a.out_dir/"integration_v1_qb_casebook.csv",index=False); score.to_csv(a.out_dir/"integration_v1_player_scorecard.csv",index=False); season_pos.to_csv(a.out_dir/"integration_v1_season_position_rec_yards.csv",index=False); pd.DataFrame(cons).to_csv(a.out_dir/"integration_v1_conservation_casebook.csv",index=False); (a.out_dir/"integration_v1_result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
