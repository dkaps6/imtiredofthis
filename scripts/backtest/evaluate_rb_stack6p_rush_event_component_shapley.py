#!/usr/bin/env python3
"""RB STACK6P: no-fit Shapley attribution of M94C within-state rush-event components."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("lead", "neutral", "trail")
COMPONENTS = ("designed", "scramble", "kneel")
SCHEMES = ("league_state_mix", "team8_shrunk_state_mix")
START_WEEK = 6
ALPHA = 0.75
TEAM_WINDOW = 8
PSEUDO_RUSHES = 24.0
EXPECTED_N = 388
EXPECTED_EMPTY_MAE = 5.518381962346741
EXPECTED_FULL_MAE = 3.4503279031625445
EXPECTED_RECOVERY = 2.0680540591841963
TEAM_MAP = {"JAX":"JAC", "LAR":"LA", "STL":"LA", "OAK":"LV", "SD":"LAC"}


def num(v): return pd.to_numeric(v, errors="coerce")
def canon(v):
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s)
def lower(df):
    z = df.copy(); z.columns = [str(c).strip().lower() for c in z.columns]; return z
def one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1: raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))

def metric(y, p):
    y, p = num(y), num(p); ok = y.notna() & p.notna(); y, p = y[ok], p[ok]; e = p-y
    return {"n":int(len(y)), "mae":float(e.abs().mean()), "rmse":float(np.sqrt(np.mean(e*e))), "bias":float(e.mean()), "corr":float(p.corr(y)) if len(y)>=3 and p.nunique()>1 and y.nunique()>1 else np.nan}

def subset_name(sub): return "NONE" if not sub else "+".join(c.upper() for c in COMPONENTS if c in sub)
def subsets():
    out=[]
    for k in range(len(COMPONENTS)+1): out += [frozenset(x) for x in itertools.combinations(COMPONENTS,k)]
    return out

def load_pbp():
    import nflreadpy as nfl
    p = lower(nfl.load_pbp(seasons=[2023,2024,2025]).to_pandas())
    if "season_type" in p.columns:
        reg=p.loc[p.season_type.astype(str).str.upper().eq("REG")].copy()
        if len(reg): p=reg
    required={"season","week","posteam","rush_attempt","qb_dropback"}
    miss=required-set(p.columns)
    if miss: raise RuntimeError(f"STACK6P PBP missing {sorted(miss)}")
    p["season"]=num(p.season).astype(int); p["week"]=num(p.week).astype(int); p["team"]=p.posteam.map(canon)
    p["rush_attempt_num"]=num(p.rush_attempt).fillna(0)
    p["qb_dropback_num"]=num(p.qb_dropback).fillna(0)
    p["qb_scramble_num"]=num(p.get("qb_scramble",pd.Series(0,index=p.index))).fillna(0)
    p["qb_kneel_num"]=num(p.get("qb_kneel",pd.Series(0,index=p.index))).fillna(0)
    p=p.loc[(p.rush_attempt_num.eq(1)|p.qb_dropback_num.eq(1)) & p.team.ne("")].copy()
    if "score_differential" in p.columns: diff=num(p.score_differential)
    elif {"posteam_score","defteam_score"}.issubset(p.columns): diff=num(p.posteam_score)-num(p.defteam_score)
    else: raise RuntimeError("STACK6P missing score differential")
    p["score_diff"]=diff.fillna(0.0)
    p["state"]=np.select([p.score_diff.gt(3),p.score_diff.lt(-3)],["lead","trail"],default="neutral")
    is_rush=p.rush_attempt_num.eq(1)
    p["scramble_event"]=(is_rush & p.qb_scramble_num.eq(1)).astype(int)
    p["kneel_event"]=(is_rush & p.qb_kneel_num.eq(1)).astype(int)
    overlap=int((p.scramble_event.eq(1)&p.kneel_event.eq(1)).sum())
    p["designed_event"]=(is_rush & p.scramble_event.eq(0)&p.kneel_event.eq(0)).astype(int)
    identity=(p.designed_event+p.scramble_event+p.kneel_event-p.rush_attempt_num).abs()
    if overlap or float(identity.max())>0: raise RuntimeError(f"STACK6P component identity failure overlap={overlap} max={identity.max()}")
    p["qtr_num"]=num(p.get("qtr",pd.Series(np.nan,index=p.index)))
    p["deep_late"]=(p.score_diff.le(-9)&p.qtr_num.ge(4)).astype(int)
    return p, overlap

def aggregate_states(p):
    rows=[]
    keys=["season","week","team"]
    for key,g in p.groupby(keys,dropna=False):
        for s in STATES:
            q=g.loc[g.state.eq(s)]
            rec=dict(zip(keys,key)); rec["state"]=s; rec["off_plays"]=float(len(q)); rec["rushes"]=float(q.rush_attempt_num.sum())
            for c in COMPONENTS: rec[f"{c}_rushes"]=float(q[f"{c}_event"].sum())
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys+["state"]).reset_index(drop=True)

def prior_mask(hist,season,week):
    return hist.season.lt(season)|(hist.season.eq(season)&hist.week.lt(week))
def component_mixes(hist, targets):
    rows=[]
    for _,r in targets.iterrows():
        season,week,team=int(r.season),int(r.week),str(r.team)
        prior=hist.loc[prior_mask(hist,season,week)].copy()
        team_games=(prior.loc[prior.team.eq(team),["season","week"]].drop_duplicates().sort_values(["season","week"]).tail(TEAM_WINDOW))
        tg=set(map(tuple,team_games.to_numpy().tolist()))
        for s in STATES:
            lg=prior.loc[prior.state.eq(s)]
            lg_counts=np.array([float(lg[f"{c}_rushes"].sum()) for c in COMPONENTS])
            lg_total=float(lg_counts.sum())
            if lg_total<=0: raise RuntimeError(f"no strict-prior league rushes state={s} {season}W{week}")
            lg_mix=lg_counts/lg_total
            team_state=prior.loc[prior.state.eq(s)&prior.team.eq(team)&prior.apply(lambda x:(x.season,x.week) in tg,axis=1)] if tg else prior.iloc[0:0]
            tm_counts=np.array([float(team_state[f"{c}_rushes"].sum()) for c in COMPONENTS])
            tm_total=float(tm_counts.sum())
            shr=(tm_counts+PSEUDO_RUSHES*lg_mix)/(tm_total+PSEUDO_RUSHES)
            rec={"season":season,"week":week,"team":team,"state":s,"team_prior8_state_rushes":tm_total,"strict_prior_ok":1}
            for i,c in enumerate(COMPONENTS): rec[f"league_{c}_mix"]=float(lg_mix[i]); rec[f"team8_{c}_mix"]=float(shr[i])
            rows.append(rec)
    return pd.DataFrame(rows)
def score_subsets(df,label,scheme):
    rows=[]; empty=None
    for sub in subsets():
        col=f"{scheme}__{subset_name(sub)}"; m=metric(df.actual_team_rush_att,df[col]); rows.append({"scheme":scheme,"population":label,"subset":subset_name(sub),"corrected_components":";".join(sorted(sub)),**m})
        if not sub: empty=m["mae"]
    out=pd.DataFrame(rows); out["recovery_vs_empty"]=float(empty)-out.mae; return out
def shapley(table,scheme,pop):
    q=table.loc[table.scheme.eq(scheme)&table.population.eq(pop)]
    values={frozenset(x for x in str(r.corrected_components).split(";") if x):float(r.recovery_vs_empty) for _,r in q.iterrows()}
    total=values[frozenset(COMPONENTS)]; n=len(COMPONENTS); rows=[]
    for c in COMPONENTS:
        phi=0.0; others=[x for x in COMPONENTS if x!=c]
        for k in range(n):
            for comb in itertools.combinations(others,k):
                S=frozenset(comb); w=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n); phi += w*(values[S|{c}]-values[S])
        rows.append({"scheme":scheme,"population":pop,"component":c,"shapley_recovery":phi,"fraction_of_tendency_recovery":phi/total if abs(total)>1e-12 else np.nan,"total_tendency_recovery":total})
    return pd.DataFrame(rows)
def context_summary(p):
    q=p.loc[p.season.eq(2025)&p.week.ge(START_WEEK)].copy(); rows=[]
    contexts={"lead":q.state.eq("lead"),"neutral":q.state.eq("neutral"),"trail":q.state.eq("trail"),"deep_late":q.deep_late.eq(1)}
    for name,mask in contexts.items():
        z=q.loc[mask]; plays=float(len(z)); rush=float(z.rush_attempt_num.sum()); rec={"context":name,"offensive_plays":plays,"rush_attempts":rush}
        for c in COMPONENTS:
            ct=float(z[f"{c}_event"].sum()); rec[f"{c}_attempts"]=ct; rec[f"{c}_per_off_play"]=ct/plays if plays else np.nan; rec[f"{c}_share_of_rushes"]=ct/rush if rush else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m94c-root",type=Path,required=True); ap.add_argument("--stack6h-root",type=Path,required=True); ap.add_argument("--out-dir",type=Path,required=True); a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    m=one(a.m94c_root,"m94c_2025_team_trace.csv"); h=one(a.stack6h_root,"stack6h_team_trace.csv"); p,overlap=load_pbp(); hist=aggregate_states(p)
    for d in (m,h): d["season"]=num(d.season).astype(int); d["week"]=num(d.week).astype(int); d["team"]=d.team.map(canon)
    req=["season","week","team","actual_team_rush_att","actual_off_plays","actual_rush_att_pbp","baseline_team_rush_att","pred_off_plays"]
    for s in STATES: req += [f"{s}_play_share",f"gs_team_{s}_rush_rate_shrunk"]
    bins=["pool_over_5","pool_under_5","pool_abs_5","non_extreme_abs_lt3"]
    t=m[req].merge(h[["season","week","team",*bins]],on=["season","week","team"],how="inner",validate="one_to_one")
    if len(t)!=544: raise RuntimeError(f"STACK6P expected 544 joined rows got {len(t)}")
    target_hist=hist.loc[hist.season.eq(2025)].copy()
    wide=[]
    for key,g in target_hist.groupby(["season","week","team"]):
        rec={"season":key[0],"week":key[1],"team":key[2],"pbp_actual_off_plays":float(g.off_plays.sum()),"pbp_actual_rushes":float(g.rushes.sum())}
        for _,r in g.iterrows():
            s=r.state; rec[f"pbp_{s}_share"]=float(r.off_plays/g.off_plays.sum()) if g.off_plays.sum()>0 else np.nan
            for c in COMPONENTS: rec[f"actual_{s}_{c}_rushes"]=float(r[f"{c}_rushes"])
        wide.append(rec)
    tw=pd.DataFrame(wide)
    t=t.merge(tw,on=["season","week","team"],how="inner",validate="one_to_one")
    mixes=component_mixes(hist,t[["season","week","team"]])
    # attach state-specific strict-prior mixes
    for s in STATES:
        ms=mixes.loc[mixes.state.eq(s)].drop(columns="state").copy()
        rename={c:f"{s}_{c}" for c in ms.columns if c not in {"season","week","team"}}
        t=t.merge(ms.rename(columns=rename),on=["season","week","team"],how="left",validate="one_to_one")
    for c in t.columns:
        if c!="team": t[c]=num(t[c])
    if t[[f"{s}_strict_prior_ok" for s in STATES]].fillna(0).ne(1).any().any(): raise RuntimeError("STACK6P strict-prior coverage failure")
    # construct all subset oracles under both allocation schemes
    for scheme in SCHEMES:
        prefix="league" if scheme=="league_state_mix" else "team8"
        for sub in subsets():
            eff=pd.Series(0.0,index=t.index)
            for s in STATES:
                for c in COMPONENTS:
                    if c in sub: eff += t[f"actual_{s}_{c}_rushes"]/t.pbp_actual_off_plays
                    else: eff += t[f"{s}_play_share"]*t[f"gs_team_{s}_rush_rate_shrunk"]*t[f"{s}_{prefix}_{c}_mix"]
            t[f"{scheme}__{subset_name(sub)}"]=(1-ALPHA)*t.baseline_team_rush_att+ALPHA*t.pred_off_plays*eff
    w=t.loc[t.season.eq(2025)&t.week.ge(START_WEEK)].copy()
    if len(w)!=EXPECTED_N: raise RuntimeError(f"expected {EXPECTED_N}, got {len(w)}")
    pops={"ALL_W6_18":pd.Series(True,index=w.index),"POOL_OVER_5":w.pool_over_5.eq(1),"POOL_UNDER_5":w.pool_under_5.eq(1),"POOL_ABS_5":w.pool_abs_5.eq(1),"NON_EXTREME_ABS_LT3":w.non_extreme_abs_lt3.eq(1)}
    scores=pd.concat([score_subsets(w.loc[mask],pop,scheme) for scheme in SCHEMES for pop,mask in pops.items()],ignore_index=True)
    shp=pd.concat([shapley(scores,scheme,pop) for scheme in SCHEMES for pop in pops],ignore_index=True)
    # integrity
    state_diff=max(float((w[f"pbp_{s}_share"]-w[f"{s}_play_share"]).abs().max()) for s in STATES)
    plays_diff=float((w.pbp_actual_off_plays-w.actual_off_plays).abs().max())
    pbp_rush_diff=float((w.pbp_actual_rushes-w.actual_rush_att_pbp).abs().max())
    mix_sum_err=0.0
    for s in STATES:
        for pre in ("league","team8"):
            mix_sum_err=max(mix_sum_err,float((sum(w[f"{s}_{pre}_{c}_mix"] for c in COMPONENTS)-1.0).abs().max()))
    identity_rows=[]; shapley_rows=[]; integrity_pass=True
    for scheme in SCHEMES:
        base=float(scores.loc[(scores.scheme.eq(scheme))&scores.population.eq("ALL_W6_18")&scores.subset.eq("NONE"),"mae"].iloc[0]); full=float(scores.loc[(scores.scheme.eq(scheme))&scores.population.eq("ALL_W6_18")&scores.subset.eq(subset_name(frozenset(COMPONENTS))),"mae"].iloc[0]); ss=float(shp.loc[(shp.scheme.eq(scheme))&shp.population.eq("ALL_W6_18"),"shapley_recovery"].sum()); rec=base-full
        ok=abs(base-EXPECTED_EMPTY_MAE)<=1e-9 and abs(full-EXPECTED_FULL_MAE)<=1e-9 and abs(rec-EXPECTED_RECOVERY)<=1e-9 and abs(ss-EXPECTED_RECOVERY)<=1e-9
        integrity_pass &= ok
        identity_rows.append({"scheme":scheme,"empty_mae":base,"full_mae":full,"recovery":rec,"shapley_sum":ss,"identity_pass":int(ok)})
    integrity_pass &= len(w)==EXPECTED_N and state_diff<=1e-9 and plays_diff<=1e-9 and pbp_rush_diff<=1e-9 and mix_sum_err<=1e-9 and overlap==0
    # disposition robust across both schemes
    def dominant(comp):
        for scheme in SCHEMES:
            for pop,frac_req in (("ALL_W6_18",.60),("POOL_OVER_5",.50),("POOL_UNDER_5",.50)):
                q=shp.loc[(shp.scheme.eq(scheme))&shp.population.eq(pop)].sort_values("shapley_recovery",ascending=False)
                row=q.loc[q.component.eq(comp)].iloc[0]
                if str(q.iloc[0].component)!=comp or float(row.fraction_of_tendency_recovery)<frac_req: return False
        return True
    if not integrity_pass: disp="STACK6P_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    elif dominant("designed"): disp="DESIGNED_RUN_CALL_DOMINANT"
    elif dominant("scramble"): disp="SCRAMBLE_COMPONENT_DOMINANT"
    elif dominant("kneel"): disp="KNEEL_COMPONENT_DOMINANT"
    else: disp="MIXED_RUSH_EVENT_COMPONENTS"
    integrity=pd.DataFrame([{"m94c_rows":len(m),"stack6h_rows":len(h),"pbp_rows":len(p),"joined_rows":len(t),"w6_18_n":len(w),"state_share_max_abs_diff":state_diff,"off_plays_max_abs_diff":plays_diff,"pbp_rush_max_abs_diff":pbp_rush_diff,"component_mix_sum_max_abs_error":mix_sum_err,"scramble_kneel_overlap":overlap,"strict_prior_coverage":float(w[[f'{s}_strict_prior_ok' for s in STATES]].mean().mean()),"integrity_pass":int(integrity_pass),"fitted_models":0,"feature_search":0,"hyperparameter_search":0,"threshold_search":0,"sportsbook_inputs":0,"target_game_pbp_used_as_oracle_only":1}])
    disposition=pd.DataFrame([{"disposition":disp,"production_change":0,"player_recomposition_authorized":0,"predictive_model_authorized":0}])
    ctx=context_summary(p)
    t.to_csv(a.out_dir/"stack6p_team_trace.csv",index=False); scores.to_csv(a.out_dir/"stack6p_subset_scores.csv",index=False); shp.to_csv(a.out_dir/"stack6p_shapley.csv",index=False); ctx.to_csv(a.out_dir/"stack6p_context_summary.csv",index=False); pd.DataFrame(identity_rows).to_csv(a.out_dir/"stack6p_scheme_identities.csv",index=False); integrity.to_csv(a.out_dir/"stack6p_integrity.csv",index=False); disposition.to_csv(a.out_dir/"stack6p_disposition.csv",index=False)
    print("=== STACK6P integrity ==="); print(integrity.to_string(index=False)); print("=== scheme identities ==="); print(pd.DataFrame(identity_rows).to_string(index=False)); print("=== context summary ==="); print(ctx.to_string(index=False)); print("=== shapley ==="); print(shp.to_string(index=False)); print("=== disposition ==="); print(disposition.to_string(index=False)); print(f"STACK6P_DISPOSITION={disp}")
    return 0
if __name__=="__main__": raise SystemExit(main())
