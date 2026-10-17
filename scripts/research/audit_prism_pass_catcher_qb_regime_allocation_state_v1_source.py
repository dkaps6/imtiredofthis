#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly

VERSION="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_AUDIT"
SEASONS=(2024,2025); FAMS=("WR","TE")

def n(x): return pd.to_numeric(x,errors="coerce")
def pdx(x):
    if hasattr(x,"to_pandas"): return x.to_pandas()
    if hasattr(x,"to_dicts"): return pd.DataFrame(x.to_dicts())
    return pd.DataFrame(x)

def weekly():
    import nflreadpy as nfl
    out=[]
    for s in (2023,2024,2025):
        x=_normalize_weekly(pdx(nfl.load_player_stats(seasons=[s],summary_level="week")),s)
        x["season"]=s;x["week"]=n(x["week"]).astype("Int64");x["team"]=x["team"].map(canon_team)
        x["player_id"]=x["player_id"].astype(str).str.strip()
        x["position"]=x["position"].astype(str).str.upper().str.strip()
        out.append(x)
    return pd.concat(out,ignore_index=True)

def universe(path):
    x=pd.read_csv(path,low_memory=False);x.columns=[str(c).lower() for c in x.columns]
    need={"season","week","team","player_id","position","family","stable_roster_identity","known_official_inactive"}
    miss=need-set(x.columns)
    if miss: raise RuntimeError(f"pregame universe missing {sorted(miss)}")
    x["season"]=n(x["season"]).astype("Int64");x["week"]=n(x["week"]).astype("Int64")
    x["team"]=x["team"].map(canon_team);x["player_id"]=x["player_id"].astype(str).str.strip()
    x["family"]=x["family"].astype(str).str.upper().str.strip()
    x["stable_roster_identity"]=n(x["stable_roster_identity"]).fillna(0).astype(int)
    x=x[x["season"].isin(SEASONS)].copy()
    if n(x["known_official_inactive"]).fillna(0).sum()!=0: raise RuntimeError("active universe contains inactive rows")
    if x.duplicated(["season","week","team","player_id"]).any(): raise RuntimeError("duplicate universe identity")
    return x

def team_sources(logs):
    team=logs.groupby(["season","week","team"],as_index=False).agg(team_targets=("targets","sum"))
    q=logs[logs["position"].eq("QB")].copy();q["pass_att"]=n(q["pass_att"]).fillna(0)
    q=q.sort_values(["season","week","team","pass_att","player_id"],ascending=[1,1,1,0,1])
    primary=q.drop_duplicates(["season","week","team"])[["season","week","team","player_id","pass_att"]]
    primary=primary.rename(columns={"player_id":"primary_qb_id","pass_att":"primary_qb_pass_att"})
    return team,primary

def qb_regimes(u,logs):
    qbu=u[u["family"].eq("QB")]
    ql=logs[logs["position"].eq("QB")].copy();ql["pass_att"]=n(ql["pass_att"]).fillna(0)
    groups={(int(s),str(t),str(p)):g for (s,t,p),g in ql.groupby(["season","team","player_id"])}
    totals=ql.groupby(["season","team","player_id"])["pass_att"].sum().to_dict()
    rows=[]
    for (s,w,t),g in qbu.groupby(["season","week","team"],sort=True):
        s=int(s);w=int(w);t=str(t);cand=[]
        for p in sorted(g["player_id"].astype(str)):
            h=groups.get((s,t,p));same=0.0
            if h is not None: same=float(n(h.loc[n(h["week"]).lt(w),"pass_att"]).fillna(0).sum())
            prior=float(totals.get((s-1,t,p),0.0));cand.append((p,same,prior))
        z=[r for r in cand if r[1]>0]
        source="QB_REGIME_SOURCE_UNAVAILABLE";pid="";sa=pa=0.0
        if z:
            pid,sa,pa=sorted(z,key=lambda r:(-r[1],r[0]))[0];source="STRICT_PRIOR_SAME_SEASON_SAME_TEAM_PASS_ATT"
        else:
            z=[r for r in cand if r[2]>0]
            if z:
                pid,sa,pa=sorted(z,key=lambda r:(-r[2],r[0]))[0];source="PRIOR_SEASON_SAME_TEAM_PASS_ATT_FALLBACK"
        rows.append({"season":s,"week":w,"team":t,"active_qb_count":len(cand),"selected_qb_id":pid,
                     "qb_regime_source":source,"qb_regime_source_valid":bool(pid),
                     "strict_prior_same_season_pass_att":sa,"prior_season_same_team_pass_att":pa})
    return pd.DataFrame(rows)

def pc_history(u,logs,team,primary):
    x=u[u["family"].isin(FAMS)].copy()
    st=logs.groupby(["season","week","team","player_id"],as_index=False).agg(targets=("targets","sum"))
    x=x.merge(st,on=["season","week","team","player_id"],how="left",validate="one_to_one")
    x["targets"]=n(x["targets"]).fillna(0)
    x=x.merge(team,on=["season","week","team"],how="left",validate="many_to_one")
    x=x.merge(primary,on=["season","week","team"],how="left",validate="many_to_one")
    x["target_identity_valid"]=x["stable_roster_identity"].eq(1)&x["player_id"].ne("")&n(x["team_targets"]).gt(0)
    x["target_share"]=np.where(n(x["team_targets"]).gt(0),n(x["targets"])/n(x["team_targets"]),np.nan)
    x["primary_qb_identity_valid"]=x["primary_qb_id"].astype(str).ne("nan")&n(x["primary_qb_pass_att"]).gt(0)
    return x

def target_states(u,reg,hist):
    t=u[u["family"].isin(FAMS)].merge(reg,on=["season","week","team"],how="left",validate="many_to_one")
    groups={(int(s),str(tm),str(p)):g.sort_values("week") for (s,tm,p),g in hist.groupby(["season","team","player_id"])}
    rows=[]
    for r in t.itertuples(index=False):
        s=int(r.season);w=int(r.week);tm=str(r.team);p=str(r.player_id);q=str(r.selected_qb_id or "")
        h=groups.get((s,tm,p),pd.DataFrame())
        if len(h): h=h[n(h["week"]).lt(w)&h["target_identity_valid"].eq(True)].copy()
        same=h[h["primary_qb_id"].astype(str).eq(q)].copy() if len(h) and q else h.iloc[0:0].copy()
        def mean(d): return float(n(d["target_share"]).mean()) if len(d) else np.nan
        def last(d,k=1):
            z=n(d.sort_values("week")["target_share"]).dropna()
            return float(z.iloc[-k:].mean()) if len(z)>=k else np.nan
        pg=int(h["week"].nunique()) if len(h) else 0;sg=int(same["week"].nunique()) if len(same) else 0
        rows.append({"season":s,"week":w,"team":tm,"player_id":p,"family":str(r.family),"position":str(r.position),
                     "stable_roster_identity":int(r.stable_roster_identity),"selected_qb_id":q,
                     "qb_regime_source":str(r.qb_regime_source),"qb_regime_source_valid":bool(r.qb_regime_source_valid),
                     "prior_games":pg,"overall_prior_target_share_mean":mean(h),"overall_prior_target_share_latest":last(h),
                     "overall_prior_target_share_latest2":last(h,2),"same_qb_prior_games":sg,
                     "same_qb_prior_target_share_mean":mean(same),"same_qb_prior_target_share_latest":last(same),
                     "same_qb_prior_target_share_latest2":last(same,2),
                     "same_qb_fraction_prior_games":float(sg/pg) if pg else np.nan,
                     "max_history_week":int(n(h["week"]).max()) if len(h) else pd.NA,
                     "chronology_violation":bool(len(h) and n(h["week"]).ge(w).any())})
    return pd.DataFrame(rows)

def source_metrics(u,reg,hist,state):
    rows=[]
    for s in SEASONS:
        r=reg[reg["season"].eq(s)];qcov=float(r["qb_regime_source_valid"].mean())
        hh=hist[hist["season"].eq(s)]
        pcov=float(hh["primary_qb_identity_valid"].mean());ticov=float(hh["target_identity_valid"].mean())
        for f in FAMS:
            x=state[(state["season"].eq(s))&state["family"].eq(f)];v=x[x["qb_regime_source_valid"].eq(True)]
            stable=float(x["stable_roster_identity"].eq(1).mean());g2=float(v["same_qb_prior_games"].ge(2).mean());g3=float(v["same_qb_prior_games"].ge(3).mean())
            sup=v[v["same_qb_prior_games"].ge(2)];sr=len(sup);sp=sup["player_id"].nunique();chron=int(x["chronology_violation"].sum())
            gates={"roster_identity_100":stable>=1-1e-12,"qb_regime_90":qcov>=.90,"stable_identity_99":stable>=.99,
                   "primary_qb_99":pcov>=.99,"target_identity_99":ticov>=.99,"same_qb_ge2_70":g2>=.70,
                   "same_qb_ge3_50":g3>=.50,"support":sr>=(1000 if f=="WR" else 600) and sp>=(150 if f=="WR" else 80),
                   "chronology_zero":chron==0}
            rows.append({"season":s,"family":f,"target_rows":len(x),"qb_regime_source_valid_teamweek_rate":qcov,
                         "target_stable_identity_rate":stable,"prior_primary_qb_identity_rate":pcov,
                         "player_target_team_target_identity_rate":ticov,"same_qb_ge2_rate":g2,"same_qb_ge3_rate":g3,
                         "same_qb_ge2_support_rows":sr,"same_qb_ge2_distinct_players":sp,"chronology_violations":chron,
                         **{f"gate_{k}":v for k,v in gates.items()},"source_ready":all(gates.values())})
    return pd.DataFrame(rows)

def main():
    a=argparse.ArgumentParser();a.add_argument("--pregame-universe",type=Path,required=True);a.add_argument("--out-dir",type=Path,required=True);z=a.parse_args();z.out_dir.mkdir(parents=True,exist_ok=True)
    u=universe(z.pregame_universe);logs=weekly();team,primary=team_sources(logs);reg=qb_regimes(u,logs);hist=pc_history(u,logs,team,primary);state=target_states(u,reg,hist);m=source_metrics(u,reg,hist,state)
    ready=int(m["source_ready"].sum());disp="PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_READY_ALL" if ready==len(m) else ("PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_READY_PARTIAL" if ready else "PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_BLOCKED")
    result={"version":VERSION,"disposition":disp,"target_seasons":list(SEASONS),"target_families":list(FAMS),"pregame_universe_rows":len(u),"qb_regime_teamweeks":len(reg),"player_target_state_rows":len(state),"ready_family_seasons":ready,"total_family_seasons":len(m),"metrics":m.to_dict("records"),"integrity":{"target_game_outcomes_used_in_source_state":0,"sportsbook_inputs_used":0,"oddsapi_calls":0,"predictive_candidates_scored":0,"coefficients_fit":0,"thresholds_searched":0,"production_changed":False,"chronology_violations":int(state["chronology_violation"].sum())}}
    reg.to_csv(z.out_dir/"qb_regime_teamweek_source.csv",index=False);hist.to_csv(z.out_dir/"active_passcatcher_completed_game_history.csv",index=False);state.to_csv(z.out_dir/"passcatcher_qb_regime_target_state.csv",index=False);m.to_csv(z.out_dir/"passcatcher_qb_regime_source_metrics.csv",index=False)
    (z.out_dir/"passcatcher_qb_regime_source_result.json").write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
if __name__=="__main__": main()

# workflow-trigger-v1
