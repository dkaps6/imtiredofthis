#!/usr/bin/env python3
"""RB-STACK4 frozen M96C efficiency portability on the STACK3 parent."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd
TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
def tm(v):
 s=str(v).strip().upper() if not pd.isna(v) else "";return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v):return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def lower(x):x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def one(root,name):
 h=list(root.rglob(name));
 if len(h)!=1:raise RuntimeError(f"expected one {name} under {root}; found {len(h)}")
 return lower(pd.read_csv(h[0],low_memory=False))
def num(s):return pd.to_numeric(s,errors="coerce")
def prep(x):
 z=x.copy();z["season"]=num(z.get("season",2025)).fillna(2025).astype(int);z["week"]=num(z.week).astype(int);z["team"]=z.team.map(tm)
 if "player_join_key" in z.columns:z["join_key"]=z.player_join_key.astype(str).map(nk)
 elif "player_clean_key" in z.columns:z["join_key"]=z.player_clean_key.astype(str).map(nk)
 elif "name_key" in z.columns:z["join_key"]=z.name_key.astype(str).map(nk)
 else:z["join_key"]=z.player.map(nk)
 return z
def metric(y,p):
 y=num(y);p=num(p);ok=y.notna()&p.notna();y=y[ok].astype(float);p=p[ok].astype(float)
 if not len(y):return {"n":0}
 e=p-y;return {"n":int(len(y)),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt(np.square(e).mean())),"bias":float(e.mean()),"corr":float(np.corrcoef(p,y)[0,1]) if len(y)>1 and y.std()>0 and p.std()>0 else np.nan,"actual_mean":float(y.mean()),"pred_mean":float(p.mean())}
ARM={"P3_PARENT":"p3_parent","P3_PLUS_E_ENRICHED_SCALE":"p3_e","P3_PLUS_P_ENRICHED_SCALE":"p3_p","P3_PLUS_D_ENRICHED_SCALE":"p3_d","P3_PLUS_D_NATIVE_SCALE":"p3_d_native","P3_PLUS_D_NONRISK":"p3_d_nonrisk"}
def add_arms(x):
 z=x.copy();parent=num(z.arm_week1_stack);enr_att=num(z.enriched_att);m94_att=num(z.m94c_att)
 risk=num(z.cal_prob_20).fillna(0).ge(.25)|num(z.m95f_p90).fillna(0).ge(20)
 z["p3_parent"]=parent;z["state_m95f_risk_stack4"]=risk.astype(int)
 for lbl,delta in [("e","delta_e"),("p","delta_p"),("d","delta_d")]:
  d=num(z[delta]);corr=(enr_att*d).where(d.notna(),0.0);z[f"p3_{lbl}"]=parent+corr
 d=num(z.delta_d);z["p3_d_native"]=parent+(m94_att*d).where(d.notna(),0.0);z["p3_d_nonrisk"]=parent+((enr_att*d).where(d.notna()&(~risk),0.0))
 return z
def masks(x):
 a=num(x.actual_rush_att);w=num(x.week);risk=num(x.state_m95f_risk_stack4).eq(1);d=num(x.delta_d).notna()
 return {"all_rb":pd.Series(True,index=x.index),"week1":w.eq(1),"weeks2_5":w.between(2,5),"weeks6_18":w.ge(6),"actual_0_5":a.le(5),"actual_6_10":a.between(6,10),"actual_11_14":a.between(11,14),"actual_15_19":a.between(15,19),"actual_20_plus":a.ge(20),"actual_25_plus":a.ge(25),"pregame_m95f_risk":risk,"pregame_m95f_nonrisk":~risk,"m96c_residual_available":d}
def score(x):
 rows=[]
 for sl,mask in masks(x).items():
  q=x.loc[mask]
  for a,c in ARM.items():rows.append({"slice":sl,"arm":a,**metric(q.actual_rush_yards,q[c])})
 return pd.DataFrame(rows)
def market(x,cb):
 c=prep(cb);keys=["week","team","join_key"];q=x.merge(c[keys+["consensus_line"]].drop_duplicates(keys),on=keys,how="inner",validate="one_to_one")
 rows=[];edges=[];bins=[0,2.5,5,10,np.inf];labs=["0_2.5","2.5_5","5_10","10_plus"]
 for a,col in {**ARM,"VEGAS_CONSENSUS":"consensus_line"}.items():rows.append({"arm":a,**metric(q.actual_rush_yards,q[col])})
 for a,col in ARM.items():
  z=q[["actual_rush_yards","consensus_line",col]].dropna().copy();z["edge"]=num(z[col])-num(z.consensus_line);z["bucket"]=pd.cut(z.edge.abs(),bins=bins,labels=labs,right=False);z["closer"]=(np.abs(num(z[col])-num(z.actual_rush_yards))<np.abs(num(z.consensus_line)-num(z.actual_rush_yards))).astype(float)
  for b,g in z.groupby("bucket",observed=True):edges.append({"arm":a,"edge_bucket":str(b),"n":len(g),"model_closer_rate":float(g.closer.mean()),"model_mae":float(np.abs(num(g[col])-num(g.actual_rush_yards)).mean()),"vegas_mae":float(np.abs(num(g.consensus_line)-num(g.actual_rush_yards)).mean()),"mean_edge":float(g.edge.mean())})
 return q,pd.DataFrame(rows),pd.DataFrame(edges)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stack3-root",type=Path,required=True);ap.add_argument("--m96c-root",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 s=prep(one(a.stack3_root,"stack3_2025_casebook.csv"));m=prep(one(a.m96c_root,"m96c_oof_trace.csv"));cb=one(a.market_root,"rb_market_casebook.csv")
 keys=["season","week","team","join_key"];need=["delta_e","delta_p","delta_d","candidate_rush_att"]
 missing=[c for c in need if c not in m.columns]
 if missing:raise RuntimeError(f"M96C trace missing {missing}")
 mm=m[keys+need].drop_duplicates(keys);x=s.merge(mm,on=keys,how="left",validate="one_to_one",suffixes=("","_m96c"))
 if "m94c_att" not in x.columns:x["m94c_att"]=num(x.candidate_rush_att)
 x=add_arms(x);sc=score(x);mq,mk,ed=market(x,cb)
 cov=pd.DataFrame([{"rows":len(x),"m96c_delta_e_rows":int(num(x.delta_e).notna().sum()),"m96c_delta_p_rows":int(num(x.delta_p).notna().sum()),"m96c_delta_d_rows":int(num(x.delta_d).notna().sum()),"m96c_delta_d_coverage":float(num(x.delta_d).notna().mean()),"market_rows":len(mq)}])
 disp=pd.DataFrame([{"disposition":"STACK4_FROZEN_EFFICIENCY_PORTABILITY_DEVELOPMENT_ONLY","model_fit":0,"feature_search":0,"threshold_search":0,"sportsbook_upstream":0,"validation_status":"2025_EXPOSED_DEVELOPMENT","next":"RETAIN_INCREMENTAL_CAPABILITIES_OR_REVERSE_ENGINEER_REMAINING_MARKET_GAP"}])
 for n,d in [("stack4_coverage.csv",cov),("stack4_slice_metrics.csv",sc),("stack4_market_metrics.csv",mk),("stack4_market_edge_buckets.csv",ed),("stack4_2025_casebook.csv",x),("stack4_disposition.csv",disp)]:d.to_csv(a.out_dir/n,index=False)
 print("=== coverage ===");print(cov.to_string(index=False));print("=== all RB ===");print(sc.loc[sc.slice.eq("all_rb")].to_string(index=False));print("=== M96C window ===");print(sc.loc[sc.slice.eq("weeks6_18")].to_string(index=False));print("=== workload ===");print(sc.loc[sc.slice.isin(["actual_20_plus","actual_25_plus","pregame_m95f_risk","pregame_m95f_nonrisk"])].to_string(index=False));print("=== market ===");print(mk.to_string(index=False));print("=== edges ===");print(ed.to_string(index=False))
if __name__=="__main__":main()
