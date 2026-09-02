#!/usr/bin/env python3
"""RB STACK6R: no-fit designed-run context occupancy vs call-tendency oracle."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STATES=("lead","neutral","trail")
CONTEXTS=("first_down","second_short_med","second_long","late_short","late_long","other")
PSEUDO=24.0
TEAM_MAP={"JAX":"JAC","LAR":"LA","STL":"LA","OAK":"LV","SD":"LAC"}
EXPECTED_2025=544
EXPECTED_W6=388


def num(v): return pd.to_numeric(v,errors="coerce")
def canon(v):
    s=str(v).strip().upper(); return TEAM_MAP.get(s,s)
def lower(df):
    z=df.copy(); z.columns=[str(c).strip().lower() for c in z.columns]; return z
def one(root:Path,name:str):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0],low_memory=False))
def prior_mask(df,season,week):
    s=num(df.season); w=num(df.week); return s.lt(season)|(s.eq(season)&w.lt(week))


def load_pbp():
    import nflreadpy as nfl
    p=lower(nfl.load_pbp(seasons=[2023,2024,2025]).to_pandas())
    if "season_type" in p.columns:
        q=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if not q.empty: p=q
    for c in ["rush_attempt","qb_dropback","qb_scramble","qb_kneel","down","ydstogo"]:
        p[c]=num(p[c]) if c in p.columns else np.nan
    p["team"]=p.posteam.map(canon)
    p["off_play"]=(p.rush_attempt.eq(1)|p.qb_dropback.eq(1)).astype(int)
    p=p.loc[p.off_play.eq(1)&p.team.ne("")].copy()
    if "score_differential" in p.columns: d=num(p.score_differential)
    elif {"posteam_score","defteam_score"}.issubset(p.columns): d=num(p.posteam_score)-num(p.defteam_score)
    else: raise RuntimeError("STACK6R missing score differential")
    p["score_diff"]=d.fillna(0)
    p["state"]=np.select([p.score_diff.gt(3),p.score_diff.lt(-3)],["lead","trail"],default="neutral")
    p["designed"]=(p.rush_attempt.eq(1)&~p.qb_scramble.fillna(0).eq(1)&~p.qb_kneel.fillna(0).eq(1)).astype(int)
    down=num(p.down); ytg=num(p.ydstogo)
    conds=[
        down.eq(1),
        down.eq(2)&ytg.le(6),
        down.eq(2)&ytg.ge(7),
        down.isin([3,4])&ytg.le(3),
        down.isin([3,4])&ytg.ge(4),
    ]
    p["context"]=np.select(conds,CONTEXTS[:-1],default="other")
    p["context_count"] = 1
    return p


def build_games(p):
    rows=[]
    for (season,week,team),g in p.groupby(["season","week","team"],dropna=False):
        rec={"season":int(season),"week":int(week),"team":str(team),"off_plays":float(len(g)),"actual_designed":float(g.designed.sum())}
        for s in STATES:
            sg=g.loc[g.state.eq(s)]
            rec[f"{s}_plays"]=float(len(sg))
            for c in CONTEXTS:
                q=sg.loc[sg.context.eq(c)]
                rec[f"{s}_{c}_plays"]=float(len(q))
                rec[f"{s}_{c}_designed"]=float(q.designed.sum())
        rows.append(rec)
    x=pd.DataFrame(rows).sort_values(["season","week","team"]).reset_index(drop=True)
    if x.empty or x.duplicated(["season","week","team"]).any(): raise RuntimeError("STACK6R game table invalid")
    return x


def league_rates(games,season,week,state,context):
    lg=games.loc[prior_mask(games,season,week)]
    state_plays=num(lg[f"{state}_plays"]).fillna(0).sum()
    ctx_plays=num(lg[f"{state}_{context}_plays"]).fillna(0).sum()
    ctx_des=num(lg[f"{state}_{context}_designed"]).fillna(0).sum()
    share=float(ctx_plays/state_plays) if state_plays>0 else 1/len(CONTEXTS)
    rate=float(ctx_des/ctx_plays) if ctx_plays>0 else 0.0
    return share,rate


def predict_one(target,games,n):
    season,week,team=int(target.season),int(target.week),canon(target.team)
    hist=games.loc[games.team.eq(team)&prior_mask(games,season,week)].sort_values(["season","week"]).tail(n)
    base=0.0; occ=0.0
    for s in STATES:
        actual_state=float(target[f"{s}_plays"])
        team_state=num(hist.get(f"{s}_plays",0)).fillna(0).sum() if len(hist) else 0.0
        for c in CONTEXTS:
            lshare,lrate=league_rates(games,season,week,s,c)
            team_ctx=num(hist.get(f"{s}_{c}_plays",0)).fillna(0).sum() if len(hist) else 0.0
            team_des=num(hist.get(f"{s}_{c}_designed",0)).fillna(0).sum() if len(hist) else 0.0
            pshare=float((team_ctx+PSEUDO*lshare)/(team_state+PSEUDO))
            prate=float((team_des+PSEUDO*lrate)/(team_ctx+PSEUDO))
            actual_ctx=float(target[f"{s}_{c}_plays"])
            base += actual_state*pshare*prate
            occ += actual_ctx*prate
    return base,occ


def metric(y,p):
    y,p=num(y),num(p); ok=y.notna()&p.notna(); y,p=y[ok],p[ok]; e=p-y
    return {"n":int(len(y)),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}


def score(df,scheme,pop):
    b=metric(df.actual_designed,df[f"{scheme}_base"]); o=metric(df.actual_designed,df[f"{scheme}_occ"])
    return pd.DataFrame([
        {"scheme":scheme,"population":pop,"arm":"BASE_CONTEXT",**b},
        {"scheme":scheme,"population":pop,"arm":"ORACLE_CONTEXT_OCCUPANCY",**o},
        {"scheme":scheme,"population":pop,"arm":"ORACLE_BOTH","n":len(df),"mae":0.0,"rmse":0.0,"bias":0.0,"corr":1.0},
    ])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    h=one(a.stack6h_root,"stack6h_team_trace.csv")
    h["season"]=num(h.season).astype(int); h["week"]=num(h.week).astype(int); h["team"]=h.team.map(canon)
    pbp=load_pbp(); games=build_games(pbp)
    g25=games.loc[games.season.eq(2025)].copy()
    t=g25.merge(h[["season","week","team","pool_over_5","pool_under_5"]],on=["season","week","team"],how="inner",validate="one_to_one")
    for n in [5,8]:
        vals=[predict_one(r,games,n) for _,r in t.iterrows()]
        t[f"team{n}_shrunk_base"]=[v[0] for v in vals]
        t[f"team{n}_shrunk_occ"]=[v[1] for v in vals]
    t["oracle_both"]=t.actual_designed
    w=t.loc[t.week.ge(6)].copy()

    context_sum=pd.Series(0.0,index=pbp.index)
    for c in CONTEXTS: context_sum += pbp.context.eq(c).astype(int)
    context_identity_max=float((context_sum-1).abs().max())
    other_share=float(pbp.loc[pbp.season.eq(2025)&pbp.week.ge(6),"context"].eq("other").mean())
    oracle_both_max=0.0

    schemes=["team5_shrunk","team8_shrunk"]
    parts=[]
    pops={"ALL_W6_18":pd.Series(True,index=w.index),"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"W13_18":w.week.ge(13)}
    for s in schemes:
        for name,m in pops.items(): parts.append(score(w.loc[m],s,name))
    scores=pd.concat(parts,ignore_index=True)

    attrs=[]
    for s in schemes:
        for pop in pops:
            z=scores.loc[(scores.scheme.eq(s))&(scores.population.eq(pop))]
            b=float(z.loc[z.arm.eq("BASE_CONTEXT"),"mae"].iloc[0]); o=float(z.loc[z.arm.eq("ORACLE_CONTEXT_OCCUPANCY"),"mae"].iloc[0])
            rec=b-o; frac=rec/b if b>0 else np.nan
            attrs.append({"scheme":s,"population":pop,"base_mae":b,"occupancy_oracle_mae":o,"occupancy_recovery":rec,"occupancy_fraction":frac,"conditional_call_remainder":o,"conditional_fraction":o/b if b>0 else np.nan})
    attribution=pd.DataFrame(attrs)

    overall=attribution.loc[attribution.population.eq("ALL_W6_18")]
    over=attribution.loc[attribution.population.eq("POOL_OVER_5")]
    under=attribution.loc[attribution.population.eq("POOL_UNDER_5")]
    occ_dom=bool((overall.occupancy_fraction.ge(0.50)).all() and (over.occupancy_recovery.gt(0)).all() and (under.occupancy_recovery.gt(0)).all())
    call_dom=bool((overall.occupancy_fraction.le(0.25)).all())
    if occ_dom: disp="CONTEXT_OCCUPANCY_DOMINANT"
    elif call_dom: disp="CONDITIONAL_CALL_DOMINANT"
    else: disp="MIXED_DESIGNED_RUN_MECHANICS"

    integrity_pass=int(len(t)==EXPECTED_2025 and len(w)==EXPECTED_W6 and context_identity_max<=1e-12 and oracle_both_max<=1e-9)
    if not integrity_pass: disp="STACK6R_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    integrity=pd.DataFrame([{"pbp_rows":len(pbp),"game_rows":len(games),"joined_2025_rows":len(t),"w6_18_rows":len(w),"context_identity_max_abs_error":context_identity_max,"other_context_share_2025_w6_18":other_share,"oracle_both_max_abs_error":oracle_both_max,"strict_prior_construction":1,"fitted_models":0,"feature_search":0,"threshold_search":0,"model_family_search":0,"hyperparameter_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1,"integrity_pass":integrity_pass}])
    disposition=pd.DataFrame([{"disposition":disp,"production_change":0,"predictive_model_authorized":0}])
    t.to_csv(a.out_dir/"stack6r_team_trace.csv",index=False); scores.to_csv(a.out_dir/"stack6r_scores.csv",index=False); attribution.to_csv(a.out_dir/"stack6r_attribution.csv",index=False); integrity.to_csv(a.out_dir/"stack6r_integrity.csv",index=False); disposition.to_csv(a.out_dir/"stack6r_disposition.csv",index=False)
    print("=== STACK6R integrity ==="); print(integrity.to_string(index=False)); print("=== STACK6R attribution ==="); print(attribution.to_string(index=False)); print("=== STACK6R disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6R_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
