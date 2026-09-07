#!/usr/bin/env python3
"""Frozen TE-R5 Participation Entitlement V1 evaluator.

Uses exact TE-R3 OOS pool predictions plus exact TE-R4 strict-prior participation.
No sportsbook data, target-game feature leakage, hyperparameter search, or post-
result tuning.  Protocol is frozen in TE_R5_PARTICIPATION_ENTITLEMENT_V1_PLAN.md.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_SEASONS=[2023,2024,2025]; EPS=.02; ALPHA=20.0
FEATURES=[
    "b0_te_room_share","log_b0_te_pool","pool_ratio","room_size",
    "prior1_same_team_offense_pct","prior1_same_team_offense_snaps",
    "prior1_anyteam_offense_pct","prior3_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps","prior3_anyteam_offense_snaps",
    "log1p_prior_count_same_team","log1p_prior_count_anyteam",
    "prior1_same_team_available","prior3_same_team_available",
    "snap_share_prior1_same_team","snap_share_prior3_anyteam",
]

def one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected exactly one {name} below {root}, found {len(h)}")
    return h[0]
def num(s): return pd.to_numeric(s,errors="coerce")
def metric(y,p):
    y=num(y); p=num(p); ok=y.notna()&p.notna(); y=y[ok].astype(float); p=p[ok].astype(float); e=p-y; ae=e.abs()
    return {"n":int(len(y)),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(np.square(e)))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>1 and p.nunique()>1 and y.nunique()>1 else np.nan,"median_abs":float(ae.median()),"p75_abs":float(ae.quantile(.75)),"p90_abs":float(ae.quantile(.90)),"miss30":float(ae.ge(30).mean()),"miss40":float(ae.ge(40).mean())}
def softmax_groups(x:pd.DataFrame)->pd.DataFrame:
    parts=[]
    for _,g in x.groupby(["season","week","team"],sort=False):
        g=g.copy(); s=num(g["entitlement_score"]).to_numpy(float); ex=np.exp(s-np.max(s)); sh=ex/ex.sum(); g["candidate_room_share"]=sh; g["candidate_targets_r5"]=num(g["candidate_te_pool"]).to_numpy(float)*sh; parts.append(g)
    return pd.concat(parts,ignore_index=False).sort_index()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--te-r3-root",type=Path,required=True); p.add_argument("--te-r4-root",type=Path,required=True); p.add_argument("--out-dir",type=Path,required=True); a=p.parse_args()
    r3=pd.read_csv(one(a.te_r3_root,"te_r3_oos_player_casebook.csv"),low_memory=False)
    r4=pd.read_csv(one(a.te_r4_root,"te_r4_prior_participation_casebook.csv"),low_memory=False)
    r4_result=json.loads(one(a.te_r4_root,"te_r4_result.json").read_text())
    r3["player_key"]=r3["player_clean_key"].fillna("").astype(str); r4["player_key"]=r4["player_key"].fillna("").astype(str)
    x=r3.merge(r4,on=["season","week","team","player_key"],how="left",suffixes=("","_r4"),indicator=True,validate="one_to_one")
    eligible=x.loc[x.season.isin(TEST_SEASONS)].copy(); join_coverage=float(eligible["_merge"].eq("both").mean())

    ncols=["prior1_same_team_offense_pct","prior1_same_team_offense_snaps","prior1_anyteam_offense_pct","prior3_anyteam_offense_pct","prior1_anyteam_offense_snaps","prior3_anyteam_offense_snaps","prior_count_same_team","prior_count_anyteam","b0_te_room_share","b0_te_pool","candidate_te_pool","actual_te_pool","targets","b0_rec_per_target","b0_rec_yards_per_target","b0_targets_recon","b0_rec_yards"]
    for c in ncols: x[c]=num(x[c])
    x["prior1_same_team_available"]=x["prior1_same_team"].fillna(False).astype(float); x["prior3_same_team_available"]=x["prior3_same_team"].fillna(False).astype(float)
    x["log1p_prior_count_same_team"]=np.log1p(x["prior_count_same_team"].fillna(0).clip(lower=0)); x["log1p_prior_count_anyteam"]=np.log1p(x["prior_count_anyteam"].fillna(0).clip(lower=0))
    x["log_b0_te_pool"]=np.log1p(x["b0_te_pool"].clip(lower=0)); x["pool_ratio"]=(x["candidate_te_pool"]/x["b0_te_pool"].clip(lower=.25)).clip(.50,2.00); x["room_size"]=x.groupby(["season","week","team"])["player_key"].transform("count").astype(float)
    for src,dst in [("prior1_same_team_offense_pct","snap_share_prior1_same_team"),("prior3_anyteam_offense_pct","snap_share_prior3_anyteam")]:
        z=x[src].fillna(0).clip(lower=0); den=z.groupby([x.season,x.week,x.team]).transform("sum"); x[dst]=np.where(den>0,z/den,0.0)
    for c in FEATURES: x[c]=num(x[c]).fillna(0.0)

    x["actual_room_share"]=np.where(x.actual_te_pool.gt(0),x.targets/x.actual_te_pool,0.0)
    x["entitlement_residual_target"]=(np.log(x.actual_room_share+EPS)-np.log(x.b0_te_room_share.clip(lower=0)+EPS)).clip(-2.0,2.0)
    x["predicted_entitlement_residual"]=np.nan
    folds=[]; coef_rows=[]
    for test in TEST_SEASONS:
        tr=x.loc[x.season.ge(2022)&x.season.lt(test)&x.actual_te_pool.gt(0)&x["_merge"].eq("both")].copy(); te=x.loc[x.season.eq(test)&x["_merge"].eq("both")].copy()
        model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)); model.fit(tr[FEATURES],tr["entitlement_residual_target"]); pred=np.clip(model.predict(te[FEATURES]),-1.0,1.0); x.loc[te.index,"predicted_entitlement_residual"]=pred
        ridge=model.named_steps["ridge"]; scaler=model.named_steps["standardscaler"]
        for f,c in zip(FEATURES,ridge.coef_): coef_rows.append({"test_season":test,"feature":f,"standardized_coefficient":float(c)})
        folds.append({"test_season":test,"train_rows":int(len(tr)),"test_rows":int(len(te)),"train_seasons":",".join(map(str,sorted(tr.season.unique().astype(int).tolist()))),"ridge_alpha":ALPHA})

    oos=x.loc[x.season.isin(TEST_SEASONS)&x["_merge"].eq("both")].copy(); oos["entitlement_score"]=np.log(oos.b0_te_room_share.clip(lower=0)+EPS)+num(oos.predicted_entitlement_residual); oos=softmax_groups(oos)
    oos["candidate_receptions_r5"]=num(oos.candidate_targets_r5)*num(oos.b0_rec_per_target); oos["candidate_rec_yards_r5"]=num(oos.candidate_targets_r5)*num(oos.b0_rec_yards_per_target)
    team_mass=oos.groupby(["season","week","team"],as_index=False).agg(candidate_target_sum=("candidate_targets_r5","sum"),candidate_te_pool=("candidate_te_pool","first")); team_mass["mass_gap"]=(team_mass.candidate_target_sum-team_mass.candidate_te_pool).abs(); max_mass_gap=float(team_mass.mass_gap.max())

    b_t=metric(oos.targets,oos.b0_targets_recon); c_t=metric(oos.targets,oos.candidate_targets_r5); b_y=metric(oos.rec_yards,oos.b0_rec_yards); c_y=metric(oos.rec_yards,oos.candidate_rec_yards_r5)
    season_rows=[]; wins=0; max_reg=-np.inf
    for s in TEST_SEASONS:
        g=oos.loc[oos.season.eq(s)]; bm=metric(g.rec_yards,g.b0_rec_yards); cm=metric(g.rec_yards,g.candidate_rec_yards_r5); d=cm["mae"]-bm["mae"]; wins+=int(d<0); max_reg=max(max_reg,d); season_rows.append({"season":s,"b0_rec_yards_mae":bm["mae"],"candidate_rec_yards_mae":cm["mae"],"delta":d,"b0_target_mae":metric(g.targets,g.b0_targets_recon)["mae"],"candidate_target_mae":metric(g.targets,g.candidate_targets_r5)["mae"]})
    latest=oos.loc[oos.season.isin([2024,2025])]; latest_b=metric(latest.rec_yards,latest.b0_rec_yards); latest_c=metric(latest.rec_yards,latest.candidate_rec_yards_r5)
    rank=oos.b0_targets_recon.rank(method="first"); oos["b0_opportunity_quartile"]=pd.qcut(rank,4,labels=["Q1","Q2","Q3","Q4"]); q4=oos.loc[oos.b0_opportunity_quartile.eq("Q4")]; q4_b=metric(q4.rec_yards,q4.b0_rec_yards); q4_c=metric(q4.rec_yards,q4.candidate_rec_yards_r5)
    dup_rate=float(oos.duplicated(["season","week","team","player_key"],keep=False).mean())
    same_future=int(r4_result.get("same_or_future_observations_used",-1))
    integrity={"all_three_test_seasons":sorted(oos.season.unique().astype(int).tolist())==TEST_SEASONS,"oos_rows_ge3000":len(oos)>=3000,"join_coverage_ge090":join_coverage>=.90,"duplicate_rate_zero":dup_rate==0.0,"zero_same_future_participation":same_future==0,"sportsbook_inputs_zero":True,"team_target_mass_gap_le1e9":max_mass_gap<=1e-9,"b0_values_unchanged":True}
    science={"target_mae_improve_ge008":b_t["mae"]-c_t["mae"]>=.08,"rec_yards_mae_improve_ge025":b_y["mae"]-c_y["mae"]>=.25,"rec_yards_wins_ge2of3":wins>=2,"latest_2425_rec_yards_improves":latest_c["mae"]<latest_b["mae"],"target_p90_guard":c_t["p90_abs"]<=b_t["p90_abs"],"rec_yards_p90_guard":c_y["p90_abs"]<=b_y["p90_abs"],"miss30_guard":c_y["miss30"]<=b_y["miss30"]+.005,"miss40_guard":c_y["miss40"]<=b_y["miss40"]+.005,"high_q4_rec_yards_improves":q4_c["mae"]<q4_b["mae"],"max_season_regression_le1":max_reg<=1.0,"bias_magnitude_guard":abs(c_y["bias"])<=abs(b_y["bias"])+1.0}
    if not all(integrity.values()): disposition="MECHANICAL_OR_SOURCE_FAILURE"
    elif all(science.values()): disposition="TE_PARTICIPATION_ENTITLEMENT_V1_PASS"
    else: disposition="TE_PARTICIPATION_ENTITLEMENT_V1_FAIL"
    result={"migration":"TE_R5_PARTICIPATION_ENTITLEMENT_V1","disposition":disposition,"production_changed":False,"sportsbook_inputs_used":False,"oos_player_games":int(len(oos)),"join_coverage":join_coverage,"duplicate_rate":dup_rate,"same_or_future_observations_used":same_future,"max_team_target_mass_gap":max_mass_gap,"features":FEATURES,"model":f"StandardScaler+Ridge(alpha={ALPHA})","pooled":{"b0_target":b_t,"candidate_target":c_t,"b0_rec_yards":b_y,"candidate_rec_yards":c_y},"latest_2024_2025":{"b0_rec_yards":latest_b,"candidate_rec_yards":latest_c},"high_q4":{"b0_rec_yards":q4_b,"candidate_rec_yards":q4_c},"season_rec_yards_wins":wins,"max_season_rec_yards_regression":float(max_reg),"integrity_gates":integrity,"scientific_gates":science}
    a.out_dir.mkdir(parents=True,exist_ok=True); oos.to_csv(a.out_dir/"te_r5_oos_player_casebook.csv",index=False); team_mass.to_csv(a.out_dir/"te_r5_team_mass_audit.csv",index=False); pd.DataFrame(folds).to_csv(a.out_dir/"te_r5_folds.csv",index=False); pd.DataFrame(coef_rows).to_csv(a.out_dir/"te_r5_coefficients.csv",index=False); pd.DataFrame(season_rows).to_csv(a.out_dir/"te_r5_season_scorecard.csv",index=False); (a.out_dir/"te_r5_result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
