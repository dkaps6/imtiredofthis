#!/usr/bin/env python3
"""RB STACK6T: no-fit fine-state conditional-call context oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES=("lead","neutral","trail")
CONTEXTS=("first_down","second_short_med","second_long","late_short","late_long","other")
FIELDS=("red_zone","opp_mid","own_mid","backed_up","field_other")
PHASES=("early","late","phase_other")
SCHEMES={"team5_shrunk":5,"team8_shrunk":8}
PSEUDO=24.0
EXPECTED={"team5_shrunk":3.9636153118306288,"team8_shrunk":4.015161235681258}
TEAM_MAP={"JAX":"JAC","LAR":"LA","STL":"LA","OAK":"LV","SD":"LAC"}


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
    required={"season","week","posteam","rush_attempt","qb_dropback","score_differential","down","ydstogo","yardline_100","qtr"}
    missing=required-set(p.columns)
    if missing: raise RuntimeError(f"STACK6T PBP missing {sorted(missing)}")
    for c in ["rush_attempt","qb_dropback","qb_scramble","qb_kneel","score_differential","down","ydstogo","yardline_100","qtr"]:
        p[c]=num(p[c]) if c in p.columns else np.nan
    p["team"]=p.posteam.map(canon)
    p["off_play"]=(p.rush_attempt.eq(1)|p.qb_dropback.eq(1)).astype(int)
    p=p.loc[p.off_play.eq(1)&p.team.ne("")].copy()
    p["state"]=np.select([p.score_differential.gt(3),p.score_differential.lt(-3)],["lead","trail"],default="neutral")
    p["designed"]=(p.rush_attempt.eq(1)&~p.qb_scramble.fillna(0).eq(1)&~p.qb_kneel.fillna(0).eq(1)).astype(int)
    d=p.down; y=p.ydstogo
    p["context"]=np.select([
        d.eq(1),d.eq(2)&y.le(6),d.eq(2)&y.ge(7),d.isin([3,4])&y.le(3),d.isin([3,4])&y.ge(4)
    ],CONTEXTS[:-1],default="other")
    yl=p.yardline_100
    p["field_zone"]=np.select([
        yl.notna()&yl.le(20),
        yl.between(21,50,inclusive="both"),
        yl.between(51,79,inclusive="both"),
        yl.notna()&yl.ge(80),
    ],FIELDS[:-1],default="field_other")
    qtr=p.qtr
    p["phase"]=np.select([qtr.between(1,3,inclusive="both"),qtr.ge(4)],["early","late"],default="phase_other")
    p["fine_cell"]=p.field_zone.astype(str)+"__"+p.phase.astype(str)
    return p


def build_games(p):
    rows=[]
    fine_cells=[f"{f}__{ph}" for f in FIELDS for ph in PHASES]
    for (season,week,team),g in p.groupby(["season","week","team"],dropna=False):
        rec={"season":int(season),"week":int(week),"team":str(team),"actual_designed":float(g.designed.sum())}
        for s in STATES:
            sg=g.loc[g.state.eq(s)]
            for c in CONTEXTS:
                cg=sg.loc[sg.context.eq(c)]
                rec[f"{s}_{c}_plays"]=float(len(cg))
                rec[f"{s}_{c}_designed"]=float(cg.designed.sum())
                for fine in fine_cells:
                    q=cg.loc[cg.fine_cell.eq(fine)]
                    rec[f"{s}_{c}_{fine}_plays"]=float(len(q))
                    rec[f"{s}_{c}_{fine}_designed"]=float(q.designed.sum())
        rows.append(rec)
    x=pd.DataFrame(rows).sort_values(["season","week","team"]).reset_index(drop=True)
    if x.empty or x.duplicated(["season","week","team"]).any(): raise RuntimeError("STACK6T game table invalid")
    return x


def league_rate(games,season,week,ncol,dcol,default=0.43):
    g=games.loc[prior_mask(games,season,week)]
    den=num(g.get(dcol,0)).fillna(0).sum(); val=num(g.get(ncol,0)).fillna(0).sum()
    return float(val/den) if den>0 else default


def team_rate(hist,ncol,dcol,lg):
    den=num(hist.get(dcol,0)).fillna(0).sum() if len(hist) else 0.0
    val=num(hist.get(ncol,0)).fillna(0).sum() if len(hist) else 0.0
    return float((val+PSEUDO*lg)/(den+PSEUDO))


def predict_one(target,games,n):
    season,week,team=int(target.season),int(target.week),canon(target.team)
    hist=games.loc[games.team.eq(team)&prior_mask(games,season,week)].sort_values(["season","week"]).tail(n)
    parent=0.0; fine_pred=0.0
    fine_cells=[f"{f}__{ph}" for f in FIELDS for ph in PHASES]
    for s in STATES:
        for c in CONTEXTS:
            pden=f"{s}_{c}_plays"; pnum=f"{s}_{c}_designed"
            lgp=league_rate(games,season,week,pnum,pden)
            pr=team_rate(hist,pnum,pden,lgp)
            actual_parent=float(target[pden])
            parent += actual_parent*pr
            for fine in fine_cells:
                fden=f"{s}_{c}_{fine}_plays"; fnum=f"{s}_{c}_{fine}_designed"
                lgf=league_rate(games,season,week,fnum,fden,pr)
                fr=team_rate(hist,fnum,fden,lgf)
                fine_pred += float(target[fden])*fr
    return parent,fine_pred


def metric(y,p):
    y,p=num(y),num(p); ok=y.notna()&p.notna(); y,p=y[ok],p[ok]; e=p-y
    return {"n":int(len(y)),"mae":float(e.abs().mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean()),"corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}


def score(df,scheme,pop):
    b=metric(df.actual_designed,df[f"{scheme}_parent"]); f=metric(df.actual_designed,df[f"{scheme}_fine"])
    rec=b["mae"]-f["mae"]
    return pd.DataFrame([
        {"scheme":scheme,"population":pop,"arm":"PARENT_CONTEXT",**b,"mae_recovery_vs_parent":0.0},
        {"scheme":scheme,"population":pop,"arm":"FINE_CONTEXT",**f,"mae_recovery_vs_parent":rec},
        {"scheme":scheme,"population":pop,"arm":"ORACLE_BOTH","n":len(df),"mae":0.0,"rmse":0.0,"bias":0.0,"corr":1.0,"mae_recovery_vs_parent":b["mae"]},
    ])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    h=one(a.stack6h_root,"stack6h_team_trace.csv")
    h["season"]=num(h.season).astype(int); h["week"]=num(h.week).astype(int); h["team"]=h.team.map(canon)
    p=load_pbp(); games=build_games(p)
    g25=games.loc[games.season.eq(2025)].copy()
    t=g25.merge(h[["season","week","team","pool_over_5","pool_under_5"]],on=["season","week","team"],how="inner",validate="one_to_one")
    for scheme,n in SCHEMES.items():
        vals=[predict_one(r,games,n) for _,r in t.iterrows()]
        t[f"{scheme}_parent"]=[v[0] for v in vals]
        t[f"{scheme}_fine"]=[v[1] for v in vals]
    w=t.loc[t.week.ge(6)].copy()
    pops={"ALL_W6_18":pd.Series(True,index=w.index),"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"W13_18":w.week.ge(13)}
    scores=pd.concat([score(w.loc[m],s,pop) for s in SCHEMES for pop,m in pops.items()],ignore_index=True)
    attrs=[]
    for s in SCHEMES:
        for pop in pops:
            z=scores.loc[(scores.scheme.eq(s))&(scores.population.eq(pop))]
            b=float(z.loc[z.arm.eq("PARENT_CONTEXT"),"mae"].iloc[0]); f=float(z.loc[z.arm.eq("FINE_CONTEXT"),"mae"].iloc[0]); rec=b-f
            attrs.append({"scheme":s,"population":pop,"parent_mae":b,"fine_mae":f,"fine_recovery":rec,"fine_recovery_fraction":rec/b if b>0 else np.nan})
    attr=pd.DataFrame(attrs)
    parent_deltas={}
    for s,expected in EXPECTED.items():
        obs=float(attr.loc[(attr.scheme.eq(s))&attr.population.eq("ALL_W6_18"),"parent_mae"].iloc[0])
        parent_deltas[s]=abs(obs-expected)
    parent_identity=int(all(v<=1e-9 for v in parent_deltas.values()))
    p6=p.loc[p.season.eq(2025)&p.week.ge(6)].copy()
    field_other=float(p6.field_zone.eq("field_other").mean()); phase_other=float(p6.phase.eq("phase_other").mean()); unclassified_union=float((p6.field_zone.eq("field_other")|p6.phase.eq("phase_other")).mean())
    integrity=int(len(t)==544 and len(w)==388 and parent_identity and unclassified_union<=0.01)
    overall=attr.loc[attr.population.eq("ALL_W6_18")]
    over=attr.loc[attr.population.eq("POOL_OVER_5")]; under=attr.loc[attr.population.eq("POOL_UNDER_5")]; late=attr.loc[attr.population.eq("W13_18")]
    material=bool((overall.fine_recovery_fraction.ge(.20)).all()&(over.fine_recovery.gt(0)).all()&(under.fine_recovery.gt(0)).all()&(late.fine_recovery.gt(0)).all())
    not_primary=bool((overall.fine_recovery_fraction.le(.10)).all())
    if not integrity: disp="STACK6T_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    elif material: disp="FINE_STATE_CONTEXT_MATERIAL"
    elif not_primary: disp="FINE_STATE_CONTEXT_NOT_PRIMARY"
    else: disp="FINE_STATE_CONTEXT_MIXED"
    integ=pd.DataFrame([{"pbp_rows":len(p),"game_rows":len(games),"joined_2025":len(t),"w6_18":len(w),"team5_parent_mae_abs_delta_vs_stack6r":parent_deltas["team5_shrunk"],"team8_parent_mae_abs_delta_vs_stack6r":parent_deltas["team8_shrunk"],"parent_identity_pass":parent_identity,"field_other_share":field_other,"phase_other_share":phase_other,"unclassified_union_share":unclassified_union,"oracle_both_exact":1,"strict_prior_construction":1,"fitted_models":0,"feature_search":0,"threshold_search":0,"model_family_search":0,"hyperparameter_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1,"integrity_pass":integrity}])
    disposition=pd.DataFrame([{"disposition":disp,"production_change":0,"predictive_model_authorized":0}])
    t.to_csv(a.out_dir/"stack6t_team_trace.csv",index=False); scores.to_csv(a.out_dir/"stack6t_scores.csv",index=False); attr.to_csv(a.out_dir/"stack6t_attribution.csv",index=False); integ.to_csv(a.out_dir/"stack6t_integrity.csv",index=False); disposition.to_csv(a.out_dir/"stack6t_disposition.csv",index=False)
    print("=== STACK6T integrity ==="); print(integ.to_string(index=False)); print("=== STACK6T attribution ==="); print(attr.to_string(index=False)); print("=== STACK6T disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6T_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
