#!/usr/bin/env python3
"""Evaluate frozen Role/Room Concentration Opportunity Experiment V1.

Plan: docs/research/ROLE_ROOM_CONCENTRATION_OPPORTUNITY_EXPERIMENT_V1.md
No sportsbook fields are read. Candidate features are strict-prior context fields.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.research.build_role_room_redundancy_audit import build_production_opportunity_state

KEY=["season","week","team","player_identity_key"]
FAMILIES={
 "RB_RUSH":{"position":"RB","num":"rushes","den":"team_rushes","domain":"rush","features":["prior_rush_share_game_top1","prior_rush_share_game_top2"]},
 "WR_TARGET":{"position":"WR","num":"targets","den":"team_targets","domain":"tgt","features":["prior_tgt_share_game_top1","prior_tgt_share_game_top2"]},
 "TE_TARGET":{"position":"TE","num":"targets","den":"team_targets","domain":"tgt","features":["prior_tgt_share_game_top1","prior_tgt_share_game_top2"]},
}

def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def metrics(y,p):
 y=np.asarray(y,float); p=np.asarray(p,float); e=p-y; ae=np.abs(e)
 def corr(kind):
  return float(pd.Series(y).corr(pd.Series(p),method=kind)) if len(y)>=3 else np.nan
 return {"rows":len(y),"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"median_ae":float(np.median(ae)),"p75_ae":float(np.quantile(ae,.75)),"p90_ae":float(np.quantile(ae,.90)),"pearson":corr("pearson"),"spearman":corr("spearman")}

def fit_predict(train,test,features,target):
 tr=train[[target,*features]].dropna(); te=test[[target,*features]].dropna()
 X=np.column_stack([np.ones(len(tr)),tr[features].to_numpy(float)]); y=tr[target].to_numpy(float)
 beta,*_=np.linalg.lstsq(X,y,rcond=None)
 Xt=np.column_stack([np.ones(len(te)),te[features].to_numpy(float)])
 return te[target].to_numpy(float), Xt@beta, len(tr)

def rel(candidate,baseline): return (candidate/baseline-1.0) if baseline else np.nan

def evaluate(history,context):
 h=history.copy(); h.columns=[str(c).lower() for c in h.columns]
 c=context.copy(); c.columns=[str(x).lower() for x in c.columns]
 state=build_production_opportunity_state(h)
 keep=KEY+["position","rushes","team_rushes","targets","team_targets"]
 x=h[keep].merge(c,on=KEY,how="inner",validate="one_to_one",suffixes=("","_ctx")).merge(state,on=KEY,how="left",validate="one_to_one")
 if "position_ctx" in x: x["position"]=x["position"].fillna(x["position_ctx"])
 rows=[]; gates=[]
 for fam,s in FAMILIES.items():
  z=x[x.position.astype(str).str.upper().eq(s["position"])].copy()
  den=pd.to_numeric(z[s["den"]],errors="coerce"); num=pd.to_numeric(z[s["num"]],errors="coerce")
  z["outcome_share"]=np.where(den>0,num/den,np.nan)
  d=s["domain"]; base=[f"prod_{d}_prior_share",f"prod_{d}_prior_games",f"prod_{d}_current_share",f"prod_{d}_current_games",f"prod_{d}_playerform_blend"]
  allf=base+s["features"]
  z=z.dropna(subset=["outcome_share",*allf])
  train=z[z.season<=2023]
  fammetrics={}
  for yr in [2024,2025]:
   test=z[z.season.eq(yr)]
   # Score on identical complete-case rows for fair baseline/candidate comparison.
   y,pb,ntr=fit_predict(train,test,base,"outcome_share"); _,pc,_=fit_predict(train,test,allf,"outcome_share")
   mb=metrics(y,pb); mc=metrics(y,pc); fammetrics[yr]=(mb,mc)
   for model,m in [("BASELINE",mb),("CANDIDATE",mc)]: rows.append({"family":fam,"evaluation_season":yr,"model":model,"train_rows":ntr,"train_seasons":"2019-2023",**m})
  for yr in [2024,2025]:
   b,cand=fammetrics[yr]
   primary=yr==2024
   gs={"mae_gate": rel(cand["mae"],b["mae"])<=(-.01 if primary else -1e-12),"rmse_gate":rel(cand["rmse"],b["rmse"])<=.0025,"p90_gate":rel(cand["p90_ae"],b["p90_ae"])<=.01,"bias_gate":abs(cand["bias"])<=abs(b["bias"])+.0025,"rows_gate":cand["rows"]>=300}
   if primary: gs["correlation_gate"]=(cand["pearson"]>=b["pearson"]-.005 and cand["spearman"]>=b["spearman"]-.005)
   passed=all(gs.values())
   gates.append({"family":fam,"evaluation_season":yr,"mae_relative_delta":rel(cand["mae"],b["mae"]),"rmse_relative_delta":rel(cand["rmse"],b["rmse"]),"p90_relative_delta":rel(cand["p90_ae"],b["p90_ae"]),"bias_abs_delta":abs(cand["bias"])-abs(b["bias"]),**gs,"season_gate_pass":passed})
  p24=gates[-2]["season_gate_pass"]; p25=gates[-1]["season_gate_pass"]
  disp="MECHANISM_REPLICATED" if p24 and p25 else ("MECHANISM_PASS_REPLICATION_FAILED" if p24 else "MECHANISM_FAIL_CLOSED_V1")
  gates[-2]["family_disposition"]=disp; gates[-1]["family_disposition"]=disp
 return pd.DataFrame(rows),pd.DataFrame(gates)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--history",type=Path,required=True); p.add_argument("--context",type=Path,required=True); p.add_argument("--metrics-out",type=Path,required=True); p.add_argument("--gates-out",type=Path,required=True); p.add_argument("--manifest-out",type=Path,required=True); a=p.parse_args()
 m,g=evaluate(pd.read_csv(a.history),pd.read_csv(a.context)); a.metrics_out.parent.mkdir(parents=True,exist_ok=True); m.to_csv(a.metrics_out,index=False); g.to_csv(a.gates_out,index=False)
 manifest={"experiment":"ROLE_ROOM_CONCENTRATION_OPPORTUNITY_EXPERIMENT_V1","history_sha256":sha256(a.history),"context_sha256":sha256(a.context),"sportsbook_read":False,"dispositions":g.groupby("family")["family_disposition"].first().to_dict()}
 a.manifest_out.write_text(json.dumps(manifest,indent=2)+"\n")
 print(g.to_string(index=False)); print(json.dumps(manifest,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
