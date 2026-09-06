#!/usr/bin/env python3
"""RB-PD2 frozen walk-forward individual player-error persistence diagnostic."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS=1393
HIST=8
MIN_PRIOR=4
MIN_ROWS=700
TEAM_ALIAS={"JAC":"JAX","JAX":"JAX","LA":"LAR","LAR":"LAR"}


def _one(root:Path,name:str)->Path:
    h=list(root.rglob(name))
    if len(h)!=1: raise RuntimeError(f"expected one {name}, got {len(h)}")
    return h[0]

def _read(p:Path)->pd.DataFrame:
    x=pd.read_csv(p,low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]
    if x.empty: raise RuntimeError(f"empty {p}")
    return x

def _key(v)->str: return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def _team(v)->str:
    r=str(v or "").strip().upper(); return TEAM_ALIAS.get(r,r)
def _num(s)->pd.Series: return pd.to_numeric(s,errors="coerce")


def wide_stack1(s:pd.DataFrame)->pd.DataFrame:
    q=s.loc[_num(s["season"]).eq(2025)&_num(s["week"]).between(1,18)&s["market"].astype(str).str.lower().isin(["rush_att","rush_yards"])].copy()
    q["team"]=q["team"].map(_team); q["player_key"]=q.get("player_clean_key",q.get("player","")).map(_key)
    keys=["season","week","team","player_key"]; rows=[]
    for k,g in q.groupby(keys,sort=False,dropna=False):
        r=dict(zip(keys,k)); r["player"]=g.iloc[0].get("player","")
        for market,suf in [("rush_att","carry"),("rush_yards","yard")]:
            z=g.loc[g["market"].astype(str).str.lower().eq(market)]
            if len(z)!=1: raise RuntimeError(f"duplicate/missing {market} {k}: {len(z)}")
            rr=z.iloc[0]; r[f"pred_{suf}"]=float(pd.to_numeric(pd.Series([rr.get("ensemble_2024_frozen")]),errors="coerce").iloc[0]); r[f"actual_{suf}"]=float(pd.to_numeric(pd.Series([rr.get("actual")]),errors="coerce").iloc[0])
        rows.append(r)
    x=pd.DataFrame(rows)
    if len(x)!=EXPECTED_ROWS: raise RuntimeError(f"STACK1 row drift {len(x)}")
    x["season"]=_num(x["season"]).astype(int); x["week"]=_num(x["week"]).astype(int)
    return x


def attach_rookie(x:pd.DataFrame,s2:pd.DataFrame)->pd.DataFrame:
    d=s2.loc[_num(s2["season"]).eq(2025)&_num(s2["week"]).between(1,18)].copy(); d["team"]=d["team"].map(_team); d["player_key"]=d.get("player_clean_key",d.get("player","")).map(_key)
    keep=[c for c in ["season","week","team","player_key","rookie_flag"] if c in d.columns]
    if "rookie_flag" not in keep: raise RuntimeError("STACK2 missing rookie_flag")
    d=d[keep].drop_duplicates(["season","week","team","player_key"],keep="last")
    out=x.merge(d,on=["season","week","team","player_key"],how="left",validate="one_to_one",indicator=True)
    if len(out)!=EXPECTED_ROWS or not out["_merge"].eq("both").all(): raise RuntimeError(f"STACK2 identity drift matched={int(out['_merge'].eq('both').sum())}")
    out=out.drop(columns="_merge"); out["rookie_flag"]=_num(out["rookie_flag"]).fillna(0)
    return out


def build_walkforward(x:pd.DataFrame)->pd.DataFrame:
    q=x.sort_values(["week","player_key"],kind="stable").copy(); q["carry_error"]=_num(q["pred_carry"])-_num(q["actual_carry"]); q["carry_abs"] = q["carry_error"].abs(); q["yard_error"]=_num(q["pred_yard"])-_num(q["actual_yard"]); q["yard_abs"]=q["yard_error"].abs()
    hist:dict[str,list[dict]]={}; rows=[]
    for r in q.itertuples(index=False):
        h=hist.get(r.player_key,[])[-HIST:]
        rec={"season":r.season,"week":r.week,"team":r.team,"player":r.player,"player_key":r.player_key,"rookie_flag":r.rookie_flag,"target_carry_error":r.carry_error,"target_carry_abs_error":r.carry_abs,"target_yard_error":r.yard_error,"target_yard_abs_error":r.yard_abs,"prior_games":len(h)}
        if h:
            d=pd.DataFrame(h); rec["prior8_carry_bias"]=float(d["carry_error"].mean()); rec["prior8_carry_mae"]=float(d["carry_abs"].mean()); rec["prior8_yard_bias"]=float(d["yard_error"].mean()); rec["prior8_yard_mae"]=float(d["yard_abs"].mean()); rec["last_prior_week"]=int(d.iloc[-1]["week"])
        else:
            for c in ["prior8_carry_bias","prior8_carry_mae","prior8_yard_bias","prior8_yard_mae","last_prior_week"]: rec[c]=np.nan
        rows.append(rec); hist.setdefault(r.player_key,[]).append({"week":r.week,"carry_error":r.carry_error,"carry_abs":r.carry_abs,"yard_error":r.yard_error,"yard_abs":r.yard_abs})
    out=pd.DataFrame(rows)
    if len(out)!=EXPECTED_ROWS or len(out.loc[out["last_prior_week"].notna()&_num(out["last_prior_week"]).ge(_num(out["week"]))]): raise RuntimeError("walk-forward integrity/leakage failure")
    return out


def _gap(g,feat,outcome):
    f=_num(g[feat]); y=_num(g[outcome]); q25=float(f.quantile(.25)); q75=float(f.quantile(.75)); return float(y.loc[f.ge(q75)].mean()-y.loc[f.le(q25)].mean())
def _slice_gap(g,feat,outcome,lo,hi):
    q=g.loc[g["week"].between(lo,hi)]; return _gap(q,feat,outcome) if len(q)>=100 and _num(q[feat]).nunique()>=4 else np.nan


def score(wf):
    g=wf.loc[wf["prior_games"].ge(MIN_PRIOR)].copy(); rows=[]
    specs=[
        ("CARRY_DIRECTIONAL_PERSISTENCE","prior8_carry_bias","target_carry_error",1.0,"sign",0.5),
        ("CARRY_DIFFICULTY_PERSISTENCE","prior8_carry_mae","target_carry_abs_error",0.75,"nosign",0),
        ("YARD_DIRECTIONAL_PERSISTENCE","prior8_yard_bias","target_yard_error",6.0,"sign",3.0),
        ("YARD_DIFFICULTY_PERSISTENCE","prior8_yard_mae","target_yard_abs_error",5.0,"nosign",0),
    ]
    for name,feat,outcome,mingap,kind,signmin in specs:
        sp=float(_num(g[feat]).corr(_num(g[outcome]),method="spearman")); gap=_gap(g,feat,outcome); early=_slice_gap(g,feat,outcome,5,12); late=_slice_gap(g,feat,outcome,13,18)
        if kind=="sign":
            q=g.loc[_num(g[feat]).abs().ge(signmin)]; sign=float((np.sign(_num(q[feat]))==np.sign(_num(q[outcome]))).mean()) if len(q) else np.nan; signok=np.isfinite(sign) and sign>=.55
        else: sign=np.nan; signok=True
        passes=bool(len(g)>=MIN_ROWS and sp>=.08 and gap>=mingap and signok and np.isfinite(early) and early>0 and np.isfinite(late) and late>0)
        rows.append({"diagnostic":name,"rows":len(g),"spearman":sp,"quartile_gap":gap,"sign_agreement":sign,"gap_weeks5_12":early,"gap_weeks13_18":late,"passes":passes})
    m=pd.DataFrame(rows); winners=m.loc[m["passes"],"diagnostic"].tolist()
    rookie=[]
    for val in [1,0]:
        q=g.loc[_num(g["rookie_flag"]).eq(val)]; rookie.append({"rookie_flag":val,"rows":len(q),"carry_mae":float(_num(q["target_carry_abs_error"]).mean()) if len(q) else np.nan,"yard_mae":float(_num(q["target_yard_abs_error"]).mean()) if len(q) else np.nan})
    summary={"migration":"RB_PD2_PLAYER_ERROR_PERSISTENCE","source_rows":len(wf),"scoreable_rows":len(g),"players":int(wf["player_key"].nunique()),"history_window":HIST,"minimum_prior_games":MIN_PRIOR,"walk_forward_leakage_violations":0,"sportsbook_inputs_used":False,"model_fitting_used":False,"production_changed":False,"passing_diagnostics":winners,"rookie_descriptive":rookie,"disposition":"RB_PLAYER_ERROR_PERSISTENCE_DETECTED" if winners else "NO_ACTIONABLE_RB_PLAYER_ERROR_PERSISTENCE"}
    return m,summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stack1-root",type=Path,required=True); ap.add_argument("--stack2-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_pd2_player_error_persistence")); a=ap.parse_args()
    s1=_read(_one(a.stack1_root,"stack1_2025_rb_trace.csv")); s2=_read(_one(a.stack2_root,"stack2_2025_casebook.csv")); x=attach_rookie(wide_stack1(s1),s2); wf=build_walkforward(x); metrics,summary=score(wf)
    a.out_dir.mkdir(parents=True,exist_ok=True); wf.to_csv(a.out_dir/"rb_pd2_walkforward_casebook.csv",index=False); metrics.to_csv(a.out_dir/"rb_pd2_metrics.csv",index=False); (a.out_dir/"rb_pd2_result.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(metrics.to_string(index=False)); print(json.dumps(summary,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
