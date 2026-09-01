#!/usr/bin/env python3
"""STACK6 failure atlas: no-fit audit of compact football-defined role concepts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd

def num(s):return pd.to_numeric(s,errors="coerce")
def one(root,name):
 h=list(root.rglob(name))
 if len(h)!=1:raise RuntimeError(f"expected one {name}; found {len(h)}")
 x=pd.read_csv(h[0],low_memory=False);x.columns=[str(c).lower() for c in x.columns];return x

def fallback(x,base):
 same=f"sr_same_p3_{base}";anyc=f"sr_any_p3_{base}"
 a=num(x[same]) if same in x else pd.Series(np.nan,index=x.index)
 b=num(x[anyc]) if anyc in x else pd.Series(np.nan,index=x.index)
 return a.where(a.notna(),b)
def fallback1(x,base):
 same=f"sr_same_p1_{base}";anyc=f"sr_any_p1_{base}"
 a=num(x[same]) if same in x else pd.Series(np.nan,index=x.index)
 b=num(x[anyc]) if anyc in x else pd.Series(np.nan,index=x.index)
 return a.where(a.notna(),b)
def safe_mean(cols):return pd.concat(cols,axis=1).mean(axis=1,skipna=True)
def corr(a,b,method="spearman"):
 q=pd.DataFrame({"a":num(a),"b":num(b)}).dropna();return float(q.a.corr(q.b,method=method)) if len(q)>=10 and q.a.nunique()>1 and q.b.nunique()>1 else np.nan

def add_concepts(x):
 z=x.copy()
 mapping={
  "role_presence":"hist_rb_presence_share","role_rush":"hist_rush_share",
  "role_early":"hist_early_down_presence_share","role_third":"hist_third_down_presence_share",
  "role_third_long":"hist_third_long_presence_share","role_two_min":"hist_two_minute_presence_share",
  "role_short":"hist_short_yardage_presence_share","role_rz":"hist_red_zone_presence_share",
  "role_i10":"hist_inside10_presence_share","role_i5":"hist_inside5_presence_share",
  "role_shotgun":"hist_shotgun_presence_share","role_uc":"hist_under_center_presence_share",
 }
 for out,base in mapping.items():z[out]=fallback(z,base);z[f"{out}_p1"]=fallback1(z,base)
 z["concept_rushing_role"]=safe_mean([z.role_rush,z.role_early,z.role_short,z.role_rz])
 z["concept_passing_role"]=safe_mean([z.role_third,z.role_third_long,z.role_two_min])
 z["concept_role_balance"]=z.concept_rushing_role-z.concept_passing_role
 z["concept_goal_line"]=safe_mean([z.role_i10,z.role_i5])
 z["concept_rush_vs_presence"]=z.role_rush-z.role_presence
 z["concept_early_vs_passing"]=z.role_early-z.concept_passing_role
 z["concept_rush_momentum"]=z.role_rush_p1-z.role_rush
 z["concept_early_momentum"]=z.role_early_p1-z.role_early
 z["concept_role_stability"]=-safe_mean([(z.role_rush_p1-z.role_rush).abs(),(z.role_early_p1-z.role_early).abs(),(z.role_third_p1-z.role_third).abs()])
 hhi="sr_team_p3_team_hhi_rb_presence"
 z["concept_team_concentration"]=num(z[hhi]) if hhi in z else np.nan
 z["actual_carry_residual"]=num(z.actual_rush_att)-num(z.parent_att)
 z["parent_carry_error"]=num(z.parent_att)-num(z.actual_rush_att)
 z["parent_yard_error_abs"]=(num(z.parent_yards)-num(z.actual_rush_yards)).abs()
 return z

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stack6-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 z=add_concepts(one(a.stack6_root,"stack6_2025_casebook.csv"));m=z.stack6_model_eligible.astype(bool)&num(z.week).ge(6);q=z.loc[m].copy()
 concepts=[c for c in z.columns if c.startswith("concept_")]+["role_presence","role_rush","role_early","role_third","role_two_min","role_short","role_rz","role_i10","role_i5"]
 rows=[];bins=[]
 for c in concepts:
  g=q[[c,"actual_carry_residual","actual_rush_att","parent_att","parent_yard_error_abs"]].dropna()
  rows.append({"concept":c,"n":len(g),"nonnull_rate":len(g)/max(len(q),1),"spearman_residual":corr(g[c],g.actual_carry_residual),"pearson_residual":corr(g[c],g.actual_carry_residual,"pearson"),"spearman_actual_carries":corr(g[c],g.actual_rush_att),"spearman_parent_att":corr(g[c],g.parent_att)})
  if len(g)>=50 and g[c].nunique()>=4:
   try:g["bin"]=pd.qcut(g[c],4,labels=False,duplicates="drop")+1
   except ValueError:continue
   for b,gg in g.groupby("bin",observed=True):bins.append({"concept":c,"quartile":int(b),"n":len(gg),"concept_mean":float(num(gg[c]).mean()),"actual_carry_residual_mean":float(num(gg.actual_carry_residual).mean()),"actual_carries_mean":float(num(gg.actual_rush_att).mean()),"parent_att_mean":float(num(gg.parent_att).mean()),"parent_yard_abs_error_mean":float(num(gg.parent_yard_error_abs).mean())})
 corrdf=pd.DataFrame(rows);bindf=pd.DataFrame(bins)
 # Predeclared compact-family support: directional ordering, not best-feature selection.
 def spread(c):
  g=bindf.loc[bindf.concept.eq(c)].sort_values("quartile")
  return float(g.iloc[-1].actual_carry_residual_mean-g.iloc[0].actual_carry_residual_mean) if len(g)>=2 else np.nan
 support=pd.DataFrame([{
  "rushing_role_q4_minus_q1_residual":spread("concept_rushing_role"),
  "role_balance_q4_minus_q1_residual":spread("concept_role_balance"),
  "rush_share_q4_minus_q1_residual":spread("role_rush"),
  "goal_line_q4_minus_q1_residual":spread("concept_goal_line"),
  "passing_role_q4_minus_q1_residual":spread("concept_passing_role"),
 }])
 # Advance only if at least one rushing-role concept has >=0.50 carry monotonic spread in expected positive direction.
 vals=[support.iloc[0][c] for c in ["rushing_role_q4_minus_q1_residual","role_balance_q4_minus_q1_residual","rush_share_q4_minus_q1_residual"]]
 advance=int(any(pd.notna(v) and v>=.50 for v in vals))
 disp=pd.DataFrame([{"compact_role_family_advance":advance,"model_fit":0,"sportsbook_upstream":0,"feature_selection":0,"disposition":"STACK6B_COMPACT_ROLE_CONTRACT_JUSTIFIED" if advance else "STACK6_ROLE_FAMILY_NOT_ORDERING_FUTURE_RESIDUAL","next":"STACK6B_FIXED_COMPACT_CONCEPT_MODEL" if advance else "NEW_INFORMATION_FAMILY"}])
 corrdf.to_csv(a.out_dir/"stack6_failure_atlas_correlations.csv",index=False);bindf.to_csv(a.out_dir/"stack6_failure_atlas_quartiles.csv",index=False);support.to_csv(a.out_dir/"stack6_failure_atlas_support.csv",index=False);disp.to_csv(a.out_dir/"stack6_failure_atlas_disposition.csv",index=False)
 print("=== support ===");print(support.to_string(index=False));print("=== correlations ===");print(corrdf.sort_values("spearman_residual",ascending=False).to_string(index=False));print("=== disposition ===");print(disp.to_string(index=False))
if __name__=="__main__":main()
