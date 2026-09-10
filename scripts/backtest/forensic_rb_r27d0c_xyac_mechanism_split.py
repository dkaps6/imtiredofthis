#!/usr/bin/env python3
"""R27D0C retrospective xYAC mechanism split. Diagnostic only; no model/candidate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canonicalize_player_name_safe
from scripts.utils.pbp import get_pbp

RB_POS={"RB","FB","HB","TB"}

def key(v):
    try:
        _,k=canonicalize_player_name_safe(v)
        if k:return str(k)
    except Exception: pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())

def col(df,names): return next((c for c in names if c in df.columns),None)
def nb(s): return s.astype("string").fillna("").str.strip().ne("")
def num(s): return pd.to_numeric(s,errors="coerce")

def weekly(season):
    import nflreadpy as nfl
    q=_normalize_weekly(_to_pandas(nfl.load_player_stats(seasons=[int(season)],summary_level="week")),int(season)).copy()
    q["week"]=num(q.week); q["team"]=q.team.map(canon_team)
    q["player_id_norm"]=q.get("player_id","").astype("string").fillna("").str.strip()
    q["player_clean_key"]=q.get("player_clean_key",q.get("player","")).astype("string").fillna("").map(key)
    q["position"]=q.get("position","").astype("string").fillna("").str.upper().str.strip()
    return q

def rb_games(season):
    p=get_pbp(int(season),min_rows=1).copy(); p.columns=[str(c).strip().lower() for c in p.columns]
    if "season_type" in p:
        r=p[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if len(r): p=r
    namec=col(p,["receiver_player_name","receiver_name","receiver"]); idc=col(p,["receiver_player_id","receiver_id"])
    req=["week","posteam","complete_pass","yards_after_catch","xyac_mean_yardage"]
    miss=[c for c in req if c not in p.columns]
    if (namec is None and idc is None) or miss: raise RuntimeError(f"{season}: missing {miss or ['receiver_identity']}")
    p["season"]=int(season); p["week"]=num(p.week); p["team"]=p.posteam.map(canon_team)
    p["rid"]=p[idc].astype("string").fillna("").str.strip() if idc else ""
    p["rname"]=p[namec].astype("string").fillna("").str.strip() if namec else ""
    p["rkey"]=p.rname.map(key)
    t=p[(nb(p.rid)|nb(p.rname)) & p.week.notna() & p.team.ne("")].copy()
    w=weekly(season)
    im=w.loc[nb(w.player_id_norm),["week","team","player_id_norm","player_clean_key","position"]].drop_duplicates(["week","team","player_id_norm"]).rename(columns={"player_id_norm":"rid","player_clean_key":"ikey","position":"ipos"})
    nm=w.loc[nb(w.player_clean_key),["week","team","player_clean_key","position"]].drop_duplicates(["week","team","player_clean_key"]).rename(columns={"player_clean_key":"rkey","position":"npos"})
    t=t.merge(im,on=["week","team","rid"],how="left",validate="many_to_one").merge(nm,on=["week","team","rkey"],how="left",validate="many_to_one")
    t["player_clean_key"]=t.ikey.replace("",pd.NA).combine_first(t.rkey.replace("",pd.NA)).fillna("")
    t["pos"]=t.ipos.replace("",pd.NA).combine_first(t.npos).fillna("")
    rb=t[t.pos.astype(str).str.upper().isin(RB_POS)&t.player_clean_key.ne("")].copy()
    rb["complete"]=num(rb.complete_pass).fillna(0)
    rb["yac"]=num(rb.yards_after_catch); rb["xyac"]=num(rb.xyac_mean_yardage)
    c=rb[rb.complete.eq(1)].copy(); c["xyac_obs"]=c.yac.notna()&c.xyac.notna(); c["yacoe"]=c.yac-c.xyac
    c["yac_xyac_obs"]=np.where(c.xyac_obs,c.yac,np.nan)
    c["expected_yac_xyac_obs"]=np.where(c.xyac_obs,c.xyac,np.nan)
    c["yacoe_xyac_obs"]=np.where(c.xyac_obs,c.yacoe,np.nan)
    keys=["season","week","team","player_clean_key"]
    g=c.groupby(keys,dropna=False)
    out=g.agg(receptions=("complete","size"),xyac_obs_receptions=("xyac_obs","sum"),actual_yac=("yac_xyac_obs","mean"),expected_yac=("expected_yac_xyac_obs","mean"),yacoe=("yacoe_xyac_obs","mean")).reset_index()
    out["xyac_coverage"]=np.where(out.receptions.gt(0),out.xyac_obs_receptions/out.receptions,np.nan)
    return out

def summarize(df,name,mask):
    g=df.loc[mask].copy(); rows=[]
    for m in ["actual_yac","expected_yac","yacoe"]:
        s=num(g[m]).dropna()
        rows.append({"cohort":name,"metric":m,"n":len(s),"mean":s.mean() if len(s) else np.nan,"median":s.median() if len(s) else np.nan,"p25":s.quantile(.25) if len(s) else np.nan,"p75":s.quantile(.75) if len(s) else np.nan})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--out-dir",required=True); a=ap.parse_args()
    od=Path(a.out_dir); od.mkdir(parents=True,exist_ok=True)
    pr=pd.read_csv(a.predictions); pr=pr[pr.actual_rec_yards.notna()].copy(); pr["season"]=num(pr.season).astype(int); pr["week"]=num(pr.week).astype(int); pr["team"]=pr.team.map(canon_team); pr["player_clean_key"]=pr.player_clean_key.astype(str).map(key)
    for p in ["b1","c1"]: pr[f"{p}_ae"]=(num(pr[f"{p}_rec_yards"])-num(pr.actual_rec_yards)).abs()
    src=pd.concat([rb_games(s) for s in range(2020,2026)],ignore_index=True)
    k=["season","week","team","player_clean_key"]; x=pr.merge(src,on=k,how="left",validate="many_to_one",indicator="xyac_join")
    vac=x.vacancy_active.eq(1); rb1=vac&x.vacancy_incumbent.eq(1)&x.role.eq("RB1"); rb2=vac&x.vacancy_incumbent.eq(1)&x.role.eq("RB2+")
    masks={"2023_VACANCY_RB1_INCUMBENT":rb1&x.season.eq(2023),"NON2023_VACANCY_RB1_INCUMBENT":rb1&x.season.ne(2023),"VACANCY_RB1_INCUMBENT":rb1,"VACANCY_RB2PLUS_INCUMBENT":rb2,"VACANCY_ACTIVE":vac,"RB1_INTO_30PLUS":rb1&x.b1_ae.lt(30)&x.c1_ae.ge(30),"RB1_BOTH_30PLUS":rb1&x.b1_ae.ge(30)&x.c1_ae.ge(30)}
    rows=[]
    for n,m in masks.items(): rows+=summarize(x,n,m)
    dist=pd.DataFrame(rows); dist.to_csv(od/"r27d0c_xyac_distributions.csv",index=False)
    def mean(c,m):
        z=dist[(dist.cohort.eq(c))&(dist.metric.eq(m))]; return float(z.iloc[0]["mean"]) if len(z) else np.nan
    c23="2023_VACANCY_RB1_INCUMBENT"; co="NON2023_VACANCY_RB1_INCUMBENT"
    actual=mean(c23,"actual_yac")-mean(co,"actual_yac"); exp=mean(c23,"expected_yac")-mean(co,"expected_yac"); oe=mean(c23,"yacoe")-mean(co,"yacoe")
    primary=x[rb1].copy(); coverage=primary.groupby(primary.season.eq(2023).map({True:"2023",False:"NON2023"})).apply(lambda g: float(num(g.xyac_coverage).dropna().mean()) if g.xyac_coverage.notna().any() else 0.0)
    summary={"status":"R27D0C_XYAC_MECHANISM_SPLIT_COMPLETE","diagnostic_only":True,"new_model_fit":False,"new_candidate_created":False,"sportsbook_inputs":0,"production_changed":False,"r26_changed":False,"r22_changed":False,"rows":int(len(x)),"primary_2023_xyac_reception_coverage":float(coverage.get("2023",0)),"primary_non2023_xyac_reception_coverage":float(coverage.get("NON2023",0)),"actual_yac_diff_2023_minus_non2023":actual,"expected_yac_diff_2023_minus_non2023":exp,"yacoe_diff_2023_minus_non2023":oe,"decomposition_gap":actual-(exp+oe),"expected_yac_share_of_signed_gap":(exp/actual if abs(actual)>1e-12 else np.nan),"yacoe_share_of_signed_gap":(oe/actual if abs(actual)>1e-12 else np.nan)}
    summary["integrity_pass"]=bool(summary["primary_2023_xyac_reception_coverage"]>=.98 and summary["primary_non2023_xyac_reception_coverage"]>=.98 and abs(summary["decomposition_gap"])<1e-9)
    (od/"r27d0c_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    x.to_csv(od/"r27d0c_joined_rows.csv",index=False)
    print(json.dumps(summary,indent=2,sort_keys=True))
    if not summary["integrity_pass"]: raise RuntimeError("R27D0C integrity/coverage failure")
if __name__=="__main__": main()
