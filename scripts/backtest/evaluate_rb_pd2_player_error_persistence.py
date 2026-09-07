#!/usr/bin/env python3
"""RB-PD3 frozen walk-forward player residual calibration test."""
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
def attach_rookie(x,s2):
 d=s2.loc[num(s2.season).eq(2025)&num(s2.week).between(1,18)].copy(); d["team"]=d.team.map(team); d["player_key"]=d.get("player_clean_key",d.get("player","")).map(key)
 if "rookie_flag" not in d: raise RuntimeError("missing rookie_flag")
 d=d[["season","week","team","player_key","rookie_flag"]].drop_duplicates(["season","week","team","player_key"],keep="last")
 o=x.merge(d,on=["season","week","team","player_key"],how="left",validate="one_to_one",indicator=True)
 if len(o)!=EXPECTED_ROWS or not o._merge.eq("both").all(): raise RuntimeError("identity drift")
 o=o.drop(columns="_merge"); o["rookie_flag"]=num(o.rookie_flag).fillna(0); return o
def build(x):
 q=x.sort_values(["week","player_key"],kind="stable").copy(); hist={}; rows=[]; leak=0
 for r in q.itertuples(index=False):
  h=hist.get(r.player_key,[])[-HIST:]; rec={"season":r.season,"week":r.week,"team":r.team,"player":r.player,"player_key":r.player_key,"rookie_flag":r.rookie_flag,"pred_carry":r.pred_carry,"actual_carry":r.actual_carry,"pred_yard":r.pred_yard,"actual_yard":r.actual_yard,"prior_games":len(h)}
  if h:
   d=pd.DataFrame(h); rec["last_prior_week"]=int(d.iloc[-1].week); rec["prior8_carry_bias"]=float(d.carry_error.mean()); rec["prior8_eff_resid_bias"]=float(d.eff_resid.mean())
  else: rec.update(last_prior_week=np.nan,prior8_carry_bias=np.nan,prior8_eff_resid_bias=np.nan)
  if pd.notna(rec["last_prior_week"]) and rec["last_prior_week"]>=r.week: leak+=1
  cb=float(r.pred_carry-r.actual_carry); yb=float(r.pred_yard-r.actual_yard); ypc=float(r.pred_yard/r.pred_carry) if np.isfinite(r.pred_carry) and r.pred_carry>0 else 0.0; eff=float(yb-cb*ypc)
  hist.setdefault(r.player_key,[]).append({"week":r.week,"carry_error":cb,"yard_error":yb,"eff_resid":eff})
  rows.append(rec)
 o=pd.DataFrame(rows)
 if len(o)!=EXPECTED_ROWS or leak: raise RuntimeError(f"walkforward integrity failure leak={leak}")
 return o
def metric(a,p):
 z=pd.DataFrame({"a":num(a),"p":num(p)}).dropna(); e=z.p-z.a; ae=e.abs();
 return {"n":len(z),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(z.p.corr(z.a)) if len(z)>2 else np.nan,"median_abs":float(ae.median()),"p75_abs":float(ae.quantile(.75)),"p90_abs":float(ae.quantile(.90)),"miss20":float(ae.ge(20).mean()),"miss30":float(ae.ge(30).mean()),"miss40":float(ae.ge(40).mean()),"miss3":float(ae.ge(3).mean()),"miss5":float(ae.ge(5).mean()),"miss7":float(ae.ge(7).mean())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--stack1-root",type=Path,required=True); ap.add_argument("--stack2-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args()
 x=attach_rookie(wide(read(one(a.stack1_root,"stack1_2025_rb_trace.csv"))),read(one(a.stack2_root,"stack2_2025_casebook.csv"))); wf=build(x); g=wf.loc[wf.prior_games.ge(MIN_PRIOR)].copy()
 g["carry_adj"]=(ALPHA*num(g.prior8_carry_bias)).clip(-CARRY_CAP,CARRY_CAP); g["cand_carry"]=(num(g.pred_carry)-g.carry_adj).clip(lower=0)
 base_ypc=np.where(num(g.pred_carry)>0,num(g.pred_yard)/num(g.pred_carry),0.0); g["eff_adj"]=(ALPHA*num(g.prior8_eff_resid_bias)).clip(-EFF_YARD_CAP,EFF_YARD_CAP); g["cand_yard"]=(base_ypc*num(g.cand_carry)-g.eff_adj).clip(lower=0)
 rows=[]
 slices=[("POOLED",g),("W5_12",g.loc[g.week.between(5,12)]),("W13_18",g.loc[g.week.between(13,18)])]
 q75=float(num(g.pred_carry).quantile(.75)); slices.append(("TOP_CARRY_Q",g.loc[num(g.pred_carry).ge(q75)]))
 for sl,z in slices:
  for market,act,b0,cand in [("carry","actual_carry","pred_carry","cand_carry"),("yard","actual_yard","pred_yard","cand_yard")]:
   rows.append({"slice":sl,"market":market,"variant":"B0",**metric(z[act],z[b0])}); rows.append({"slice":sl,"market":market,"variant":"CAND",**metric(z[act],z[cand])})
 m=pd.DataFrame(rows)
 def get(sl,market,var,col): return float(m.loc[(m.slice==sl)&(m.market==market)&(m.variant==var),col].iloc[0])
 gates={
 "carry_mae_improve_ge_0_05":get("POOLED","carry","B0","mae")-get("POOLED","carry","CAND","mae")>=.05,
 "yard_mae_improve_ge_0_25":get("POOLED","yard","B0","mae")-get("POOLED","yard","CAND","mae")>=.25,
 "carry_p90_not_worse":get("POOLED","carry","CAND","p90_abs")<=get("POOLED","carry","B0","p90_abs")+1e-12,
 "yard_p90_not_worse":get("POOLED","yard","CAND","p90_abs")<=get("POOLED","yard","B0","p90_abs")+1e-12,
 "carry_5miss_guard":get("POOLED","carry","CAND","miss5")-get("POOLED","carry","B0","miss5")<=.005,
 "yard_30miss_guard":get("POOLED","yard","CAND","miss30")-get("POOLED","yard","B0","miss30")<=.005,
 "yard_40miss_guard":get("POOLED","yard","CAND","miss40")-get("POOLED","yard","B0","miss40")<=.005,
 "early_carry_guard":get("W5_12","carry","CAND","mae")-get("W5_12","carry","B0","mae")<=.10,
 "late_carry_guard":get("W13_18","carry","CAND","mae")-get("W13_18","carry","B0","mae")<=.10,
 "early_yard_guard":get("W5_12","yard","CAND","mae")-get("W5_12","yard","B0","mae")<=.50,
 "late_yard_guard":get("W13_18","yard","CAND","mae")-get("W13_18","yard","B0","mae")<=.50,
 "topq_carry_guard":get("TOP_CARRY_Q","carry","CAND","mae")-get("TOP_CARRY_Q","carry","B0","mae")<=.10,
 "topq_yard_guard":get("TOP_CARRY_Q","yard","CAND","mae")-get("TOP_CARRY_Q","yard","B0","mae")<=.50}
 integrity={"source_rows_exact":len(wf)==EXPECTED_ROWS,"scoreable_rows_ge_700":len(g)>=700,"walkforward_leakage_violations":0,"sportsbook_inputs_used":False,"alpha":ALPHA,"carry_cap":CARRY_CAP,"eff_yard_cap":EFF_YARD_CAP}; passed=all(gates.values()) and integrity["source_rows_exact"] and integrity["scoreable_rows_ge_700"]
 players=[]
 for pk,z in g.groupby("player_key"):
  if len(z)<6: continue
  players.append({"player_key":pk,"player":z.player.iloc[-1],"games":len(z),"b0_carry_mae":metric(z.actual_carry,z.pred_carry)["mae"],"cand_carry_mae":metric(z.actual_carry,z.cand_carry)["mae"],"b0_yard_mae":metric(z.actual_yard,z.pred_yard)["mae"],"cand_yard_mae":metric(z.actual_yard,z.cand_yard)["mae"]})
 result={"migration":"RB_PD3_PLAYER_RESIDUAL_CALIBRATION","production_changed":False,"sportsbook_inputs_used":False,"scoreable_rows":len(g),"integrity":integrity,"scientific_gates":gates,"disposition":"RB_PD3_PLAYER_RESIDUAL_CALIBRATION_PASS" if passed else "RB_PD3_PLAYER_RESIDUAL_CALIBRATION_FAIL"}
 a.out_dir.mkdir(parents=True,exist_ok=True); g.to_csv(a.out_dir/"rb_pd3_casebook.csv",index=False); m.to_csv(a.out_dir/"rb_pd3_metrics.csv",index=False); pd.DataFrame(players).to_csv(a.out_dir/"rb_pd3_player_scorecard.csv",index=False); (a.out_dir/"rb_pd3_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print(json.dumps(result,indent=2,sort_keys=True)); print(m.to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
