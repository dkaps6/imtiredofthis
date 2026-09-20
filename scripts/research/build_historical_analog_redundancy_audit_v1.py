#!/usr/bin/env python3
"""Outcome-free production-state redundancy audit for Historical Analog State V1."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from scripts.research.build_role_room_redundancy_audit import build_production_opportunity_state

KEY=["season","week","team","player_identity_key"]
CONT=["nearest_distance","mean_k_distance","effective_analog_count"]
PROD=["prod_tgt_prior_share","prod_tgt_prior_games","prod_tgt_current_share","prod_tgt_current_games","prod_tgt_playerform_blend","prod_rush_prior_share","prod_rush_prior_games","prod_rush_current_share","prod_rush_current_games","prod_rush_playerform_blend"]


def _xy(df:pd.DataFrame,target:str):
    z=df[["season",target,*PROD]].copy()
    for c in [target,*PROD]: z[c]=pd.to_numeric(z[c],errors="coerce")
    # Missing production shares are part of canonical cold-start state; retain rows deterministically.
    z[PROD]=z[PROD].fillna(0.0)
    z=z.dropna(subset=[target])
    return z[z.season<=2023],z[z.season>=2024]


def _r2(train,test,target):
    if len(train)<500 or len(test)<200:return np.nan
    Xtr=np.column_stack([np.ones(len(train)),train[PROD].to_numpy(float)])
    Xte=np.column_stack([np.ones(len(test)),test[PROD].to_numpy(float)])
    beta,*_=np.linalg.lstsq(Xtr,train[target].to_numpy(float),rcond=None)
    y=test[target].to_numpy(float); pred=Xte@beta; sst=float(((y-y.mean())**2).sum())
    return 1-float(((y-pred)**2).sum())/sst if sst>0 else np.nan


def audit(history:pd.DataFrame,states:pd.DataFrame,qualification:pd.DataFrame)->pd.DataFrame:
    prod=build_production_opportunity_state(history)
    s=states.copy(); s.columns=[str(c).strip().lower() for c in s.columns]
    if s.duplicated(KEY).any(): raise RuntimeError("analog state has duplicate canonical keys")
    x=s.merge(prod,on=KEY,how="left",validate="one_to_one")
    q=qualification.copy(); q.columns=[str(c).strip().lower() for c in q.columns]
    rows=[]
    for pos,p in x.groupby("position",sort=True):
        qa=q[q.position==pos]
        hard=bool(len(qa)==1 and qa.iloc[0][["support_gate","coverage_gate","diversity_gate","chronology_gate"]].astype(bool).all() and float(qa.iloc[0].max_valid_season_share)<=.50)
        valid=p[p.analog_state=="VALID_ANALOG"].copy()
        for target in CONT:
            tr,te=_xy(valid,target); r2=_r2(tr,te,target)
            if np.isfinite(r2) and r2>=.90: rd="HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
            elif np.isfinite(r2) and r2>=.75: rd="REDUNDANCY_REVIEW"
            else: rd="INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
            disp="READY_FOR_FROZEN_EXPERIMENT" if hard and rd=="INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE" else (rd if rd!="INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE" else "DESCRIPTIVE_ONLY")
            rows.append({"position":pos,"descriptor":target,"descriptor_type":"continuous","train_rows":len(tr),"holdout_rows":len(te),"holdout_reconstructibility_r2":r2,"balanced_accuracy":np.nan,"precision":np.nan,"recall":np.nan,"f1":np.nan,"hard_qualification_gates":hard,"redundancy_disposition":rd,"qualification_disposition":disp,"outcomes_read":False,"sportsbook_read":False})
        # Explicit no-support state: audit classification convention without outcomes.
        b=p.copy(); b["no_analog_support_flag"]=(b.analog_state!="VALID_ANALOG").astype(int)
        tr,te=_xy(b,"no_analog_support_flag")
        ba=pr=rc=f1=np.nan
        if len(tr)>=500 and len(te)>=200 and tr.no_analog_support_flag.nunique()>1 and te.no_analog_support_flag.nunique()>1:
            m=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=0).fit(tr[PROD],tr.no_analog_support_flag.astype(int))
            pred=m.predict(te[PROD]); y=te.no_analog_support_flag.astype(int)
            ba=float(balanced_accuracy_score(y,pred)); pr=float(precision_score(y,pred,zero_division=0)); rc=float(recall_score(y,pred,zero_division=0)); f1=float(f1_score(y,pred,zero_division=0))
        redundant=bool(np.isfinite(ba) and np.isfinite(f1) and ba>=.90 and f1>=.80)
        rd="HIGHLY_RECONSTRUCTIBLE_REDUNDANT" if redundant else "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
        disp="READY_FOR_FROZEN_EXPERIMENT" if hard and not redundant else (rd if redundant else "DESCRIPTIVE_ONLY")
        rows.append({"position":pos,"descriptor":"no_analog_support_flag","descriptor_type":"binary","train_rows":len(tr),"holdout_rows":len(te),"holdout_reconstructibility_r2":np.nan,"balanced_accuracy":ba,"precision":pr,"recall":rc,"f1":f1,"hard_qualification_gates":hard,"redundancy_disposition":rd,"qualification_disposition":disp,"outcomes_read":False,"sportsbook_read":False})
    return pd.DataFrame(rows).sort_values(["position","descriptor"]).reset_index(drop=True)


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--history",type=Path,required=True); ap.add_argument("--states",type=Path,required=True); ap.add_argument("--qualification",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    out=audit(pd.read_csv(a.history),pd.read_csv(a.states),pd.read_csv(a.qualification)); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False); print(out.to_string(index=False)); print(f"[analog_redundancy] rows={len(out)} -> {a.out}"); return 0
if __name__=="__main__": raise SystemExit(main())
