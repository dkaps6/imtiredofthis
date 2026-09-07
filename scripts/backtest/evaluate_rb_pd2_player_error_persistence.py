#!/usr/bin/env python3
"""RB-PD4 frozen role-stability-gated player residual calibration test."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
EXPECTED_ROWS=1393; HIST=8; MIN_PRIOR=4; ALPHA=.25; CARRY_CAP=2.0; EFF_YARD_CAP=8.0
TEAM_ALIAS={"JAC":"JAX","JAX":"JAX","LA":"LAR","LAR":"LAR"}
def one(root,name):
 h=list(root.rglob(name))
 if len(h)!=1: raise RuntimeError(f"expected one {name}, got {len(h)}")
 return h[0]
def read(p):
 x=pd.read_csv(p,low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]
 if x.empty: raise RuntimeError(f"empty {p}")
 return x
def key(v): return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def team(v):
 r=str(v or "").strip().upper(); return TEAM_ALIAS.get(r,r)
def num(s): return pd.to_numeric(s,errors="coerce")
def wide(s):
 q=s.loc[num(s.season).eq(2025)&num(s.week).between(1,18)&s.market.astype(str).str.lower().isin(["rush_att","rush_yards"])].copy(); q["team"]=q.team.map(team); q["player_key"]=q.get("player_clean_key",q.get("player","")).map(key); rows=[]
 for k,g in q.groupby(["season","week","team","player_key"],sort=False,dropna=False):
  r=dict(zip(["season","week","team","player_key"],k)); r["player"]=g.iloc[0].get("player","")
  for m,suf in [("rush_att","carry"),("rush_yards","yard")]:
   z=g.loc[g.market.astype(str).str.lower().eq(m)]
   if len(z)!=1: raise RuntimeError(f"duplicate/missing {m} {k}: {len(z)}")
   rr=z.iloc[0]; r[f"pred_{suf}"]=float(pd.to_numeric(pd.Series([rr.get("ensemble_2024_frozen")]),errors="coerce").iloc[0]); r[f"actual_{suf}"]=float(pd.to_numeric(pd.Series([rr.get("actual")]),errors="coerce").iloc[0])
  rows.append(r)
 x=pd.DataFrame(rows)
 if len(x)!=EXPECTED_ROWS: raise RuntimeError(f"row drift {len(x)}")
 x["season"]=num(x.season).astype(int); x["week"]=num(x.week).astype(int); return x
def build(x):
 q=x.sort_values(["week","player_key"],kind="stable").copy(); hist={}; rows=[]; leak=0
 for r in q.itertuples(index=False):
  h=hist.get(r.player_key,[]); h8=h[-HIST:]; h4=h[-4:]
  rec={"season":r.season,"week":r.week,"team":r.team,"player":r.player,"player_key":r.player_key,"pred_carry":r.pred_carry,"actual_carry":r.actual_carry,"pred_yard":r.pred_yard,"actual_yard":r.actual_yard,"prior_games":len(h8)}
  if h8:
   d8=pd.DataFrame(h8); rec["last_prior_week"]=int(d8.iloc[-1].week); rec["prior8_carry_bias"]=float(d8.carry_error.mean()); rec["prior8_eff_resid_bias"]=float(d8.eff_resid.mean())
  else: rec.update(last_prior_week=np.nan,prior8_carry_bias=np.nan,prior8_eff_resid_bias=np.nan)
  if len(h4)==4:
   d4=pd.DataFrame(h4); mu=float(d4.actual_carry.mean()); sd=float(d4.actual_carry.std(ddof=0)); rec["prior4_actual_carry_mean"]=mu; rec["prior4_actual_carry_cv"]=float(sd/mu) if mu>0 else np.inf
  else: rec.update(prior4_actual_carry_mean=np.nan,prior4_actual_carry_cv=np.nan)
  if pd.notna(rec["last_prior_week"]) and rec["last_prior_week"]>=r.week: leak+=1
  cb=float(r.pred_carry-r.actual_carry); yb=float(r.pred_yard-r.actual_yard); ypc=float(r.pred_yard/r.pred_carry) if np.isfinite(r.pred_carry) and r.pred_carry>0 else 0.0; eff=float(yb-cb*ypc)
  hist.setdefault(r.player_key,[]).append({"week":r.week,"carry_error":cb,"eff_resid":eff,"actual_carry":float(r.actual_carry)})
  rows.append(rec)
 o=pd.DataFrame(rows)
 if len(o)!=EXPECTED_ROWS or leak: raise RuntimeError(f"walkforward integrity failure leak={leak}")
 return o
def metric(a,p):
 z=pd.DataFrame({"a":num(a),"p":num(p)}).dropna(); e=z.p-z.a; ae=e.abs()
 return {"n":len(z),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(z.p.corr(z.a)) if len(z)>2 else np.nan,"median_abs":float(ae.median()),"p75_abs":float(ae.quantile(.75)),"p90_abs":float(ae.quantile(.90)),"miss20":float(ae.ge(20).mean()),"miss30":float(ae.ge(30).mean()),"miss40":float(ae.ge(40).mean()),"miss3":float(ae.ge(3).mean()),"miss5":float(ae.ge(5).mean()),"miss7":float(ae.ge(7).mean())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--stack1-root",type=Path,required=True); ap.add_argument("--stack2-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
 x=wide(read(one(a.stack1_root,"stack1_2025_rb_trace.csv"))); wf=build(x); g=wf.loc[wf.prior_games.ge(MIN_PRIOR)].copy()
 g["stable_role"]=(num(g.last_prior_week).ge(num(g.week)-2)&num(g.prior4_actual_carry_mean).ge(8.0)&num(g.prior4_actual_carry_cv).le(.35)&(num(g.pred_carry)-num(g.prior4_actual_carry_mean)).abs().le(4.0))
 g["carry_adj_raw"]=(ALPHA*num(g.prior8_carry_bias)).clip(-CARRY_CAP,CARRY_CAP); g["carry_adj"]=np.where(g.stable_role,g.carry_adj_raw,0.0); g["cand_carry"]=(num(g.pred_carry)-g.carry_adj).clip(lower=0)
 base_ypc=np.where(num(g.pred_carry)>0,num(g.pred_yard)/num(g.pred_carry),0.0); g["eff_adj_raw"]=(ALPHA*num(g.prior8_eff_resid_bias)).clip(-EFF_YARD_CAP,EFF_YARD_CAP); g["eff_adj"]=np.where(g.stable_role,g.eff_adj_raw,0.0); g["cand_yard"]=(base_ypc*num(g.cand_carry)-g.eff_adj).clip(lower=0); g.loc[~g.stable_role,"cand_yard"]=num(g.loc[~g.stable_role,"pred_yard"])
 rows=[]; q75=float(num(g.pred_carry).quantile(.75)); slices=[("POOLED",g),("STABLE",g.loc[g.stable_role]),("W5_12",g.loc[g.week.between(5,12)]),("W13_18",g.loc[g.week.between(13,18)]),("TOP_CARRY_Q",g.loc[num(g.pred_carry).ge(q75)])]
 for sl,z in slices:
  for market,act,b0,cand in [("carry","actual_carry","pred_carry","cand_carry"),("yard","actual_yard","pred_yard","cand_yard")]:
   rows.append({"slice":sl,"market":market,"variant":"B0",**metric(z[act],z[b0])}); rows.append({"slice":sl,"market":market,"variant":"CAND",**metric(z[act],z[cand])})
 m=pd.DataFrame(rows)
 def get(sl,market,var,col): return float(m.loc[(m.slice==sl)&(m.market==market)&(m.variant==var),col].iloc[0])
 outside=g.loc[~g.stable_role]; exact_outside=bool(np.allclose(num(outside.pred_carry),num(outside.cand_carry),equal_nan=True) and np.allclose(num(outside.pred_yard),num(outside.cand_yard),equal_nan=True))
 gates={
 "carry_mae_improve_ge_0_03":get("POOLED","carry","B0","mae")-get("POOLED","carry","CAND","mae")>=.03,
 "yard_mae_improve_ge_0_15":get("POOLED","yard","B0","mae")-get("POOLED","yard","CAND","mae")>=.15,
 "carry_p90_not_worse":get("POOLED","carry","CAND","p90_abs")<=get("POOLED","carry","B0","p90_abs")+1e-12,
 "yard_p90_not_worse":get("POOLED","yard","CAND","p90_abs")<=get("POOLED","yard","B0","p90_abs")+1e-12,
 "carry_5miss_guard":get("POOLED","carry","CAND","miss5")-get("POOLED","carry","B0","miss5")<=.005,
 "yard_30miss_guard":get("POOLED","yard","CAND","miss30")-get("POOLED","yard","B0","miss30")<=.005,
 "yard_40miss_guard":get("POOLED","yard","CAND","miss40")-get("POOLED","yard","B0","miss40")<=.005,
 "late_yard_p90_not_worse":get("W13_18","yard","CAND","p90_abs")<=get("W13_18","yard","B0","p90_abs")+1e-12,
 "late_yard_mae_guard":get("W13_18","yard","CAND","mae")-get("W13_18","yard","B0","mae")<=.25,
 "topq_carry_guard":get("TOP_CARRY_Q","carry","CAND","mae")-get("TOP_CARRY_Q","carry","B0","mae")<=.10,
 "topq_yard_guard":get("TOP_CARRY_Q","yard","CAND","mae")-get("TOP_CARRY_Q","yard","B0","mae")<=.50,
 "stable_carry_mae_better":get("STABLE","carry","CAND","mae")<get("STABLE","carry","B0","mae"),
 "stable_yard_mae_better":get("STABLE","yard","CAND","mae")<get("STABLE","yard","B0","mae")}
 integrity={"source_rows_exact":len(wf)==EXPECTED_ROWS,"scoreable_rows_ge_700":len(g)>=700,"stable_rows_ge_200":int(g.stable_role.sum())>=200,"walkforward_leakage_violations":0,"candidate_equals_baseline_outside_stable":exact_outside,"sportsbook_inputs_used":False,"alpha":ALPHA,"carry_cap":CARRY_CAP,"eff_yard_cap":EFF_YARD_CAP,"stable_role_rows":int(g.stable_role.sum()),"scoreable_rows":len(g)}
 passed=all(gates.values()) and all(v for k,v in integrity.items() if k in ["source_rows_exact","scoreable_rows_ge_700","stable_rows_ge_200","candidate_equals_baseline_outside_stable"])
 players=[]
 for pk,z in g.groupby("player_key"):
  if len(z)<6: continue
  players.append({"player_key":pk,"player":z.player.iloc[-1],"games":len(z),"stable_games":int(z.stable_role.sum()),"b0_carry_mae":metric(z.actual_carry,z.pred_carry)["mae"],"cand_carry_mae":metric(z.actual_carry,z.cand_carry)["mae"],"b0_yard_mae":metric(z.actual_yard,z.pred_yard)["mae"],"cand_yard_mae":metric(z.actual_yard,z.cand_yard)["mae"]})
 prevalence=g.groupby("week",as_index=False).agg(rows=("player_key","size"),stable_rows=("stable_role","sum")); prevalence["stable_rate"]=prevalence.stable_rows/prevalence.rows
 result={"migration":"RB_PD4_ROLE_STABILITY_GATED_CALIBRATION","production_changed":False,"sportsbook_inputs_used":False,"integrity":integrity,"scientific_gates":gates,"disposition":"RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_PASS" if passed else "RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_FAIL"}
 a.out_dir.mkdir(parents=True,exist_ok=True); g.to_csv(a.out_dir/"rb_pd4_casebook.csv",index=False); m.to_csv(a.out_dir/"rb_pd4_metrics.csv",index=False); pd.DataFrame(players).to_csv(a.out_dir/"rb_pd4_player_scorecard.csv",index=False); prevalence.to_csv(a.out_dir/"rb_pd4_role_prevalence.csv",index=False); (a.out_dir/"rb_pd4_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True)); print(m.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
