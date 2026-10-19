#!/usr/bin/env python3
"""Score the frozen TE receiving team-context Stage-1 participation reset V1."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    discrete_crps,
    mixture_binom_cdf,
)

VERSION="PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1"
FREEZE_COMMIT="929d6565b594c00c870c78717481c9082175911c"
SOURCE_RUN=39255619282
SOURCE_ARTIFACT=12187947053
SOURCE_DIGEST="sha256:885f3ec53f553f3d2aa1de04d87776497c2d76624bb6d35a600597b31064d266"
PARENT_RUN=38957575645
PARENT_ARTIFACT=12070774378
PARENT_DIGEST="sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
SEASONS=(2024,2025)
BOOT_REPS=5000
BOOT_SEED=20261019
TOL=1e-12
PARITY_TOL=1e-10

def num(s:Any)->pd.Series: return pd.to_numeric(s,errors="coerce")

def bseries(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype("string").str.strip().str.lower().map(
        {"true":True,"false":False,"1":True,"0":False}
    ).fillna(False).astype(bool)

def read_csv(path:Path,label:str)->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0: raise RuntimeError(f"missing {label}: {path}")
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    forbidden=[c for c in x.columns if any(t in c for t in (
        "sportsbook","bookmaker","prop_line","market_line","over_odds","under_odds",
        "moneyline","closing_line","no_vig","implied_prob"
    ))]
    if forbidden: raise RuntimeError(f"forbidden sportsbook columns in {label}: {forbidden}")
    return x

def prepare_source(x:pd.DataFrame)->pd.DataFrame:
    req={"transition_id","season","team","stable_id","transition_kind","target_week","game_key",
         "stage1_p_active","reset_p_active","actionable_stage1_reset"}
    if req-set(x.columns): raise RuntimeError(f"source missing {sorted(req-set(x.columns))}")
    z=x.copy()
    z["season"]=num(z["season"]).astype("Int64"); z["target_week"]=num(z["target_week"]).astype("Int64")
    for c in ("transition_id","team","stable_id","transition_kind","game_key"):
        z[c]=z[c].astype("string").fillna("").str.strip()
    z["stage1_p_active"]=num(z["stage1_p_active"]); z["reset_p_active"]=num(z["reset_p_active"])
    z["actionable_stage1_reset"]=bseries(z["actionable_stage1_reset"])
    z=z.loc[z["season"].isin(SEASONS)&z["actionable_stage1_reset"]].copy()
    if z["transition_id"].duplicated().any(): raise RuntimeError("duplicate actionable transition")
    if not (np.isfinite(z["reset_p_active"]).all()&z["reset_p_active"].gt(0).all()&z["reset_p_active"].lt(1).all()):
        raise RuntimeError("invalid reset p")
    return z.reset_index(drop=True)

def prepare_players(x:pd.DataFrame)->pd.DataFrame:
    req={"season","week","game_key","team","cell","family","player_id","stage1_p_active","n","y",
         "authority_q","canonical_zero_share","candidate_expected_count","candidate_crps","candidate_ae"}
    if req-set(x.columns): raise RuntimeError(f"parent player rows missing {sorted(req-set(x.columns))}")
    z=x.copy()
    z["season"]=num(z["season"]).astype("Int64"); z["week"]=num(z["week"]).astype("Int64")
    for c in ("game_key","team","cell","family","player_id"):
        z[c]=z[c].astype("string").fillna("").str.strip()
    for c in ("stage1_p_active","n","y","authority_q","candidate_expected_count","candidate_crps","candidate_ae"):
        z[c]=num(z[c])
    z["canonical_zero_share"]=bseries(z["canonical_zero_share"])
    z=z.loc[z["season"].isin(SEASONS)&z["cell"].eq("TE_RECEIVING")&z["family"].eq("TE")].copy()
    if z.duplicated(["season","week","game_key","team","player_id"]).any(): raise RuntimeError("duplicate parent player row")
    return z.reset_index(drop=True)

def prepare_rooms(x:pd.DataFrame)->pd.DataFrame:
    req={"season","week","game_key","team","cell","candidate_room_error"}
    if req-set(x.columns): raise RuntimeError(f"parent room rows missing {sorted(req-set(x.columns))}")
    z=x.copy(); z["season"]=num(z["season"]).astype("Int64"); z["week"]=num(z["week"]).astype("Int64")
    for c in ("game_key","team","cell"): z[c]=z[c].astype("string").fillna("").str.strip()
    z["candidate_room_error"]=num(z["candidate_room_error"])
    z=z.loc[z["season"].isin(SEASONS)&z["cell"].eq("TE_RECEIVING")].copy()
    if z.duplicated(["season","week","game_key","team"]).any(): raise RuntimeError("duplicate parent room")
    return z.reset_index(drop=True)

def enumerate_states(q:np.ndarray,p:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    q=np.asarray(q,float); p=np.asarray(p,float)
    if len(q)!=len(p): raise RuntimeError("q/p length mismatch")
    if not np.isfinite(q).all() or (q<=0).any(): raise RuntimeError("invalid q")
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise RuntimeError("invalid p")
    q0=1.0-float(q.sum())
    if q0 < -TOL: raise RuntimeError("canonical mass exceeds one")
    if abs(q0)<=TOL: q0=0.0
    det=p>=1.0-TOL; idx=np.flatnonzero(~det)
    ws=[]; rs=[]; comps=[]
    for bits in itertools.product((0,1),repeat=len(idx)):
        a=np.ones(len(q),float)
        if len(idx):
            a[idx]=np.asarray(bits,float); pp=p[idx]; bb=a[idx]
            w=float(np.prod(np.where(bb>0,pp,1.0-pp)))
        else: w=1.0
        den=float(q0+np.dot(a,q))
        if den>TOL: r=(a*q)/den; r0=q0/den
        else: r=np.zeros(len(q)); r0=1.0
        if abs(float(r.sum())+float(r0)-1.0)>TOL: raise RuntimeError("state mass drift")
        ws.append(w); rs.append(r); comps.append(float(r0))
    w=np.asarray(ws,float); r=np.vstack(rs); r0=np.asarray(comps,float)
    if abs(float(w.sum())-1.0)>TOL: raise RuntimeError("state probability drift")
    return w,r,r0

def room_error(expected:np.ndarray,comp:float,y:np.ndarray,N:int)->float:
    obscomp=float(N-int(y.sum()))/N
    if obscomp < -TOL: raise RuntimeError("observed positive-q counts exceed room N")
    ev=np.append(expected,comp); ov=np.append(y.astype(float)/N,obscomp)
    if abs(float(ev.sum())-1.0)>TOL: raise RuntimeError("expected mass drift")
    return 0.5*float(np.abs(ev-ov).sum())

def score_event(e:pd.Series,players:pd.DataFrame,rooms:pd.DataFrame)->dict[str,Any]:
    full=players.loc[
        players["season"].eq(int(e["season"]))&players["week"].eq(int(e["target_week"]))&
        players["game_key"].eq(str(e["game_key"]))&players["team"].eq(str(e["team"]))
    ].copy()
    if full.empty: raise RuntimeError(f"missing room {e['transition_id']}")
    pos=full.loc[full["authority_q"].gt(TOL)].copy().sort_values("player_id",kind="mergesort").reset_index(drop=True)
    if pos.empty: raise RuntimeError("empty positive-q room")
    idx=np.flatnonzero(pos["player_id"].astype(str).eq(str(e["stable_id"])).to_numpy())
    if len(idx)!=1: raise RuntimeError(f"target count {len(idx)}")
    j=int(idx[0])
    if bool(pos.iloc[j]["canonical_zero_share"]): raise RuntimeError("target canonical zero")

    Nvals=num(full["n"]).to_numpy(float)
    if not np.isfinite(Nvals).all() or not np.allclose(Nvals,Nvals[0],atol=TOL): raise RuntimeError("N drift")
    if abs(float(Nvals[0])-round(float(Nvals[0])))>TOL or Nvals[0]<=0: raise RuntimeError("invalid N")
    N=int(round(float(Nvals[0])))

    q=num(pos["authority_q"]).to_numpy(float)
    pb=num(pos["stage1_p_active"]).to_numpy(float)
    if abs(float(pb[j])-float(e["stage1_p_active"]))>TOL: raise RuntimeError("source parent p mismatch")
    pc=pb.copy(); pc[j]=float(e["reset_p_active"])

    yraw=num(pos["y"]).to_numpy(float)
    if not np.allclose(yraw,np.rint(yraw),atol=TOL): raise RuntimeError("noninteger y")
    y=np.rint(yraw).astype(int)
    if (y<0).any() or (y>N).any(): raise RuntimeError("y outside room")

    wb,rb,rb0=enumerate_states(q,pb); wc,rc,rc0=enumerate_states(q,pc)
    eb=wb@rb; ec=wc@rc; eb0=float(wb@rb0); ec0=float(wc@rc0)

    yi=int(y[j])
    bcr=discrete_crps(mixture_binom_cdf(N,rb[:,j],wb),yi)
    ccr=discrete_crps(mixture_binom_cdf(N,rc[:,j],wc),yi)
    bmu=N*float(eb[j]); cmu=N*float(ec[j])
    bae=abs(yi-bmu); cae=abs(yi-cmu)

    stored=pos.iloc[j]
    rr=rooms.loc[
        rooms["season"].eq(int(e["season"]))&rooms["week"].eq(int(e["target_week"]))&
        rooms["game_key"].eq(str(e["game_key"]))&rooms["team"].eq(str(e["team"]))
    ]
    if len(rr)!=1: raise RuntimeError("room summary identity failure")

    bre=room_error(eb,eb0,y,N); cre=room_error(ec,ec0,y,N)

    return {
        "transition_id":str(e["transition_id"]),"season":int(e["season"]),
        "transition_kind":str(e["transition_kind"]),"target_week":int(e["target_week"]),
        "game_key":str(e["game_key"]),"team":str(e["team"]),"stable_id":str(e["stable_id"]),
        "baseline_p_active":float(pb[j]),"candidate_reset_p_active":float(pc[j]),
        "y":yi,"N":N,
        "baseline_expected_count":bmu,"candidate_expected_count":cmu,
        "baseline_crps":bcr,"candidate_crps":ccr,"delta_crps":bcr-ccr,
        "baseline_ae":bae,"candidate_ae":cae,"delta_ae":bae-cae,
        "baseline_room_error":bre,"candidate_room_error":cre,"delta_room_error":bre-cre,
        "baseline_expected_count_abs_diff":abs(bmu-float(stored["candidate_expected_count"])),
        "baseline_crps_abs_diff":abs(bcr-float(stored["candidate_crps"])),
        "baseline_ae_abs_diff":abs(bae-float(stored["candidate_ae"])),
        "baseline_room_error_abs_diff":abs(bre-float(rr.iloc[0]["candidate_room_error"])),
        "baseline_state_weight_gap":abs(float(wb.sum())-1.0),
        "candidate_state_weight_gap":abs(float(wc.sum())-1.0),
        "baseline_mass_gap":abs(float(eb.sum())+eb0-1.0),
        "candidate_mass_gap":abs(float(ec.sum())+ec0-1.0),
    }

def cluster_boot(df:pd.DataFrame,val:str,cluster:str,seed:int)->dict[str,Any]:
    z=df[[val,cluster]].dropna(); agg=z.groupby(cluster,sort=True)[val].agg(["sum","count"])
    if len(agg)<2: return {"clusters":int(len(agg)),"valid_reps":0,"ci95_low":math.nan,"ci95_high":math.nan}
    a=agg[["sum","count"]].to_numpy(float); rng=np.random.default_rng(seed); out=np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        ix=rng.integers(0,len(a),size=len(a)); t=a[ix].sum(axis=0); out[i]=t[0]/t[1]
    return {"clusters":int(len(a)),"valid_reps":BOOT_REPS,"ci95_low":float(np.quantile(out,.025)),"ci95_high":float(np.quantile(out,.975))}

def player_frame(g:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame([
        {"stable_id":str(pid),"events":int(len(x)),"mean_delta_crps":float(x["delta_crps"].mean()),
         "mean_baseline_crps":float(x["baseline_crps"].mean()),"mean_delta_ae":float(x["delta_ae"].mean())}
        for pid,x in g.groupby("stable_id",sort=True)
    ])

def player_boot(p:pd.DataFrame,seed:int)->dict[str,Any]:
    v=num(p["mean_delta_crps"]).dropna().to_numpy(float)
    if len(v)<2:return {"players":int(len(v)),"valid_reps":0,"ci95_low":math.nan,"ci95_high":math.nan}
    rng=np.random.default_rng(seed); out=np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        ix=rng.integers(0,len(v),size=len(v)); out[i]=float(v[ix].mean())
    return {"players":int(len(v)),"valid_reps":BOOT_REPS,"ci95_low":float(np.quantile(out,.025)),"ci95_high":float(np.quantile(out,.975))}

def season_score(rows:pd.DataFrame,season:int)->tuple[dict[str,Any],pd.DataFrame]:
    g=rows.loc[rows["season"].eq(season)].copy(); p=player_frame(g); v=num(p["mean_delta_crps"]).dropna()
    rec={
        "season":season,"actionable_events":int(len(g)),"stable_ids":int(g["stable_id"].nunique()),"games":int(g["game_key"].nunique()),
        "mean_delta_crps":float(g["delta_crps"].mean()),"mean_delta_ae":float(g["delta_ae"].mean()),
        "mean_delta_room_error":float(g["delta_room_error"].mean()),
        "equal_player_mean_improvement":float(v.mean()),"equal_player_median_improvement":float(v.median()),
        "equal_player_p10_improvement":float(v.quantile(.10)),
        "equal_player_improved_share":float((v>TOL).mean()),"equal_player_tied_share":float((v.abs()<=TOL).mean()),
        "equal_player_worsened_share":float((v<-TOL).mean()),
        "baseline_equal_player_crps":float(p["mean_baseline_crps"].mean()),
        "max_baseline_expected_count_parity_diff":float(g["baseline_expected_count_abs_diff"].max()),
        "max_baseline_crps_parity_diff":float(g["baseline_crps_abs_diff"].max()),
        "max_baseline_ae_parity_diff":float(g["baseline_ae_abs_diff"].max()),
        "max_baseline_room_error_parity_diff":float(g["baseline_room_error_abs_diff"].max()),
        "max_baseline_state_weight_gap":float(g["baseline_state_weight_gap"].max()),
        "max_candidate_state_weight_gap":float(g["candidate_state_weight_gap"].max()),
        "max_baseline_mass_gap":float(g["baseline_mass_gap"].max()),
        "max_candidate_mass_gap":float(g["candidate_mass_gap"].max()),
        "transition_kind_diagnostics":{
            str(k):{"events":int(len(x)),"stable_ids":int(x["stable_id"].nunique()),"mean_delta_crps":float(x["delta_crps"].mean())}
            for k,x in g.groupby("transition_kind",sort=True)
        },
    }
    rec["player_cluster_bootstrap"]=cluster_boot(g,"delta_crps","stable_id",BOOT_SEED+season)
    rec["game_cluster_bootstrap"]=cluster_boot(g,"delta_crps","game_key",BOOT_SEED+10000+season)
    rec["equal_player_bootstrap"]=player_boot(p,BOOT_SEED+20000+season)
    return rec,p

def parity(s): return all(s[k]<=PARITY_TOL for k in (
    "max_baseline_expected_count_parity_diff","max_baseline_crps_parity_diff",
    "max_baseline_ae_parity_diff","max_baseline_room_error_parity_diff"))
def conservation(s): return all(s[k]<=PARITY_TOL for k in (
    "max_baseline_state_weight_gap","max_candidate_state_weight_gap","max_baseline_mass_gap","max_candidate_mass_gap"))

def qualify(d:dict[str,Any],c:dict[str,Any])->tuple[dict[str,Any],bool]:
    tail=-.25*float(c["baseline_equal_player_crps"]) if np.isfinite(c["baseline_equal_player_crps"]) else math.nan
    g={
        "development_actionable_events_ge_15":d["actionable_events"]>=15,
        "development_distinct_ids_ge_15":d["stable_ids"]>=15,
        "development_baseline_parity":parity(d),
        "development_mean_crps_gain_positive":d["mean_delta_crps"]>0,
        "development_mean_ae_gain_nonnegative":d["mean_delta_ae"]>=0,
        "development_equal_player_mean_positive":d["equal_player_mean_improvement"]>0,
        "development_median_player_nonnegative":d["equal_player_median_improvement"]>=0,
        "development_improved_share_gt_worsened":d["equal_player_improved_share"]>d["equal_player_worsened_share"],
        "development_room_error_gain_nonnegative":d["mean_delta_room_error"]>=0,
        "development_conservation":conservation(d),
        "confirmation_actionable_events_ge_15":c["actionable_events"]>=15,
        "confirmation_distinct_ids_ge_15":c["stable_ids"]>=15,
        "confirmation_baseline_parity":parity(c),
        "confirmation_mean_crps_gain_positive":c["mean_delta_crps"]>0,
        "confirmation_player_cluster_ci_low_gt_zero":c["player_cluster_bootstrap"]["ci95_low"]>0,
        "confirmation_game_cluster_ci_low_gt_zero":c["game_cluster_bootstrap"]["ci95_low"]>0,
        "confirmation_equal_player_mean_positive":c["equal_player_mean_improvement"]>0,
        "confirmation_equal_player_ci_low_gt_zero":c["equal_player_bootstrap"]["ci95_low"]>0,
        "confirmation_median_player_nonnegative":c["equal_player_median_improvement"]>=0,
        "confirmation_improved_share_gt_worsened":c["equal_player_improved_share"]>c["equal_player_worsened_share"],
        "confirmation_mean_ae_gain_nonnegative":c["mean_delta_ae"]>=0,
        "confirmation_room_error_gain_nonnegative":c["mean_delta_room_error"]>=0,
        "confirmation_tail_harm_guard":np.isfinite(tail) and c["equal_player_p10_improvement"]>=tail,
        "confirmation_conservation":conservation(c),
        "tail_floor":tail,
    }
    return g,all(v for k,v in g.items() if k!="tail_floor")

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-rows",type=Path,required=True)
    ap.add_argument("--parent-player-rows",type=Path,required=True)
    ap.add_argument("--parent-room-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    try:
        source=prepare_source(read_csv(a.source_rows,"source"))
        players=prepare_players(read_csv(a.parent_player_rows,"parent players"))
        rooms=prepare_rooms(read_csv(a.parent_room_rows,"parent rooms"))
        if len(source.loc[source["season"].eq(2024)])!=17: raise RuntimeError("unexpected 2024 cohort")
        if len(source.loc[source["season"].eq(2025)])!=18: raise RuntimeError("unexpected 2025 cohort")
        scored=pd.DataFrame([score_event(r,players,rooms) for _,r in source.iterrows()])
        d,p24=season_score(scored,2024); c,p25=season_score(scored,2025)
        if not parity(d) or not parity(c):
            disposition="PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_INTEGRITY_FAILURE"
            gates={"development_baseline_parity":parity(d),"confirmation_baseline_parity":parity(c)}
        else:
            gates,ok=qualify(d,c)
            disposition="PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_QUALIFIED" if ok else "PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_PREDICTIVE_CLOSED"
        scored.to_csv(a.out_dir/"te_receiving_team_context_stage1_reset_scored_rows.csv",index=False)
        pd.concat([p24.assign(season=2024),p25.assign(season=2025)],ignore_index=True).to_csv(
            a.out_dir/"te_receiving_team_context_stage1_reset_player_diagnostics.csv",index=False
        )
        result={
            "version":VERSION,"freeze_commit":FREEZE_COMMIT,
            "source_parent":{"run":SOURCE_RUN,"artifact":SOURCE_ARTIFACT,"digest":SOURCE_DIGEST},
            "qualified_te_parent":{"run":PARENT_RUN,"artifact":PARENT_ARTIFACT,"digest":PARENT_DIGEST},
            "development_2024":d,"confirmation_2025":c,"qualification_gates":gates,"disposition":disposition,
            "bootstrap_reps":BOOT_REPS,"bootstrap_seed":BOOT_SEED,
            "coefficients_fit":0,"thresholds_searched":0,"subgroup_rescues":0,
            "yards_scored":False,"touchdowns_scored":False,"sportsbook_inputs_used":0,"oddsapi_calls":0,
            "outcomes_2026_used":0,"production_changed":False,
        }
    except Exception as exc:
        result={"version":VERSION,"freeze_commit":FREEZE_COMMIT,
                "disposition":"PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_INTEGRITY_FAILURE",
                "error":str(exc),"coefficients_fit":0,"thresholds_searched":0,"subgroup_rescues":0,
                "sportsbook_inputs_used":0,"oddsapi_calls":0,"outcomes_2026_used":0,"production_changed":False}
        (a.out_dir/"te_receiving_team_context_stage1_reset_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(json.dumps(result,indent=2,sort_keys=True)); raise
    (a.out_dir/"te_receiving_team_context_stage1_reset_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
