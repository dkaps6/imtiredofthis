#!/usr/bin/env python3
"""Source-only audit for TE receiving team-context Stage-1 participation reset V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION="PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE"
FREEZE_COMMIT="37dc741d23fea0e030d03c33e8b63c094efb710f"
TEAM_CONTEXT_RUN=39182076127
TEAM_CONTEXT_ARTIFACT=12156204035
TEAM_CONTEXT_DIGEST="sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8"
TE_PARENT_RUN=38957575645
TE_PARENT_ARTIFACT=12070774378
TE_PARENT_DIGEST="sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
STAGE1_RUN=38907459996
STAGE1_ARTIFACT=12049120604
STAGE1_DIGEST="sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b"
SEASONS=(2024,2025)
TRANSITION_KINDS={"OFFSEASON_TEAM_CHANGE","INSEASON_TEAM_CHANGE"}
TOL=1e-12

def txt(v:Any)->str:
    if v is None or pd.isna(v): return ""
    s=str(v).strip()
    return "" if s.upper() in {"","NAN","NONE","<NA>"} else s

def bseries(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype("string").str.strip().str.lower().map(
        {"true":True,"false":False,"1":True,"0":False}
    ).fillna(False).astype(bool)

def readcols(path:Path,label:str,cols:list[str])->pd.DataFrame:
    if not path.exists() or path.stat().st_size<=0:
        raise RuntimeError(f"missing {label}: {path}")
    h=pd.read_csv(path,nrows=0)
    mp={str(c).strip().lower():c for c in h.columns}
    miss=[c for c in cols if c not in mp]
    if miss: raise RuntimeError(f"{label} missing {miss}")
    x=pd.read_csv(path,usecols=[mp[c] for c in cols],low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    return x

def prep_events(path:Path)->tuple[pd.DataFrame,pd.DataFrame]:
    x=readcols(path,"team-context transitions",[
        "season","week","team","stable_id","family","transition_kind","prior_team_support_ge4"
    ])
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team)
    x["stable_id"]=x["stable_id"].map(txt)
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["transition_kind"]=x["transition_kind"].astype("string").fillna("").str.strip()
    x["prior_team_support_ge4"]=bseries(x["prior_team_support_ge4"])
    full=x.loc[
        x["season"].isin(SEASONS)&x["family"].eq("TE")&
        x["transition_kind"].ne("")&x["stable_id"].ne("")
    ].copy()
    full["season"]=full["season"].astype(int); full["week"]=full["week"].astype(int)
    eligible=full.loc[
        full["transition_kind"].isin(TRANSITION_KINDS)&full["prior_team_support_ge4"]
    ].copy().sort_values(["season","stable_id","week","team"],kind="mergesort").reset_index(drop=True)
    eligible["transition_id"]=[
        f"{int(s)}:{int(w)}:{t}:{pid}:{kind}"
        for s,w,t,pid,kind in eligible[
            ["season","week","team","stable_id","transition_kind"]
        ].itertuples(index=False,name=None)
    ]
    if eligible["transition_id"].duplicated().any(): raise RuntimeError("duplicate transition")
    return full.reset_index(drop=True),eligible

def prep_states(path:Path)->pd.DataFrame:
    x=readcols(path,"team-context states",["season","week","team","stable_id","family"])
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team)
    x["stable_id"]=x["stable_id"].map(txt)
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x=x.loc[x["season"].isin(SEASONS)&x["family"].eq("TE")&x["stable_id"].ne("")].copy()
    x["season"]=x["season"].astype(int); x["week"]=x["week"].astype(int)
    if x.duplicated(["season","week","team","stable_id"]).any(): raise RuntimeError("duplicate state")
    return x

def prep_parent(path:Path)->pd.DataFrame:
    # Intentionally excludes N, y, CRPS, AE and names.
    x=readcols(path,"qualified TE parent",[
        "season","week","game_key","team","cell","family","player_id",
        "stage1_p_active","stage1_named_player_supported","authority_q","canonical_zero_share"
    ])
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team)
    x["player_id"]=x["player_id"].map(txt)
    x["game_key"]=x["game_key"].map(txt)
    x["cell"]=x["cell"].astype("string").fillna("").str.strip()
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["stage1_p_active"]=pd.to_numeric(x["stage1_p_active"],errors="coerce")
    x["authority_q"]=pd.to_numeric(x["authority_q"],errors="coerce")
    x["stage1_named_player_supported"]=bseries(x["stage1_named_player_supported"])
    x["canonical_zero_share"]=bseries(x["canonical_zero_share"])
    x=x.loc[
        x["season"].isin(SEASONS)&x["cell"].eq("TE_RECEIVING")&
        x["family"].eq("TE")&x["player_id"].ne("")
    ].copy()
    x["season"]=x["season"].astype(int); x["week"]=x["week"].astype(int)
    if x.duplicated(["season","week","team","player_id"]).any(): raise RuntimeError("duplicate TE parent identity")
    return x.reset_index(drop=True)

def prep_stage1(path:Path)->pd.DataFrame:
    # Intentionally excludes y and all names.
    x=readcols(path,"Stage-1 mechanism",[
        "season","week","team","family","player_id","p_baseline","p_candidate","n_player_prior"
    ])
    x["season"]=pd.to_numeric(x["season"],errors="coerce")
    x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team)
    x["player_id"]=x["player_id"].map(txt)
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    for c in ("p_baseline","p_candidate","n_player_prior"):
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.loc[x["season"].isin(SEASONS)&x["family"].eq("TE")&x["player_id"].ne("")].copy()
    x["season"]=x["season"].astype(int); x["week"]=x["week"].astype(int)
    if x.duplicated(["season","week","team","player_id"]).any(): raise RuntimeError("duplicate Stage-1 TE identity")
    return x

def select_rows(full:pd.DataFrame,eligible:pd.DataFrame,parent:pd.DataFrame)->tuple[pd.DataFrame,dict[str,int]]:
    out=[]; ex={"NO_CANONICAL_TE_RECEIVING_BASELINE":0}
    groups={(int(s),str(pid)):g.sort_values(["week","team"],kind="mergesort")
            for (s,pid),g in full.groupby(["season","stable_id"],sort=False)}
    for r in eligible.itertuples(index=False):
        later=groups.get((int(r.season),str(r.stable_id)))
        next_week=None
        if later is not None:
            qlater=later.loc[later["week"].gt(int(r.week))]
            if len(qlater): next_week=int(qlater["week"].min())
        q=parent.loc[
            parent["season"].eq(int(r.season))&
            parent["player_id"].eq(str(r.stable_id))&
            parent["team"].eq(str(r.team))&
            parent["week"].ge(int(r.week))
        ].copy()
        if next_week is not None: q=q.loc[q["week"].lt(next_week)]
        if q.empty:
            ex["NO_CANONICAL_TE_RECEIVING_BASELINE"]+=1
            continue
        first=q.sort_values(["week","game_key"],kind="mergesort").iloc[0]
        out.append({
            "transition_id":str(r.transition_id),"season":int(r.season),
            "transition_week":int(r.week),"team":str(r.team),"stable_id":str(r.stable_id),
            "transition_kind":str(r.transition_kind),"next_transition_week":next_week,
            "target_week":int(first["week"]),"game_key":str(first["game_key"]),
            "stage1_p_active":float(first["stage1_p_active"]),
            "stage1_named_player_supported":bool(first["stage1_named_player_supported"]),
            "authority_q":float(first["authority_q"]),
            "canonical_zero_share":bool(first["canonical_zero_share"]),
        })
    z=pd.DataFrame(out)
    if len(z):
        if z["transition_id"].duplicated().any(): raise RuntimeError("duplicate selected transition")
        if (z["target_week"]<z["transition_week"]).any(): raise RuntimeError("chronology violation")
        m=z["next_transition_week"].notna()
        if (z.loc[m,"target_week"]>=z.loc[m,"next_transition_week"]).any(): raise RuntimeError("crossed later transition")
    return z,ex

def attach_reset(selected:pd.DataFrame,stage1:pd.DataFrame)->pd.DataFrame:
    s=stage1.rename(columns={
        "week":"target_week","player_id":"stable_id",
        "p_baseline":"stage1_p_baseline","p_candidate":"stage1_p_candidate",
        "n_player_prior":"stage1_player_prior_rows",
    })[["season","target_week","team","stable_id","stage1_p_baseline","stage1_p_candidate","stage1_player_prior_rows"]]
    z=selected.merge(s,on=["season","target_week","team","stable_id"],how="left",validate="many_to_one")
    z["stage1_exact_mechanism_row"]=z["stage1_p_candidate"].notna()
    m=z["stage1_exact_mechanism_row"]
    z["stage1_candidate_matches_current"]=True
    z.loc[m,"stage1_candidate_matches_current"]=(
        z.loc[m,"stage1_p_candidate"]-z.loc[m,"stage1_p_active"]
    ).abs().le(TOL)
    z["stage1_support_flag_matches_exact_row"]=z["stage1_named_player_supported"].eq(z["stage1_exact_mechanism_row"])
    z["reset_p_active"]=z["stage1_p_baseline"].where(m,z["stage1_p_active"])
    z["actionable_stage1_reset"]=(z["reset_p_active"]-z["stage1_p_active"]).abs().gt(TOL)
    return z

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--team-context-events",type=Path,required=True)
    ap.add_argument("--team-context-states",type=Path,required=True)
    ap.add_argument("--te-parent-rows",type=Path,required=True)
    ap.add_argument("--stage1-mechanism-rows",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)

    full,eligible=prep_events(a.team_context_events)
    states=prep_states(a.team_context_states)
    parent=prep_parent(a.te_parent_rows)
    stage1=prep_stage1(a.stage1_mechanism_rows)

    sk=set(map(tuple,states[["season","week","team","stable_id"]].itertuples(index=False,name=None)))
    attempts=len(parent)
    hits=sum((int(s),int(w),str(t),str(pid)) in sk for s,w,t,pid in parent[
        ["season","week","team","player_id"]
    ].itertuples(index=False,name=None))
    rate=hits/attempts if attempts else 0.0

    selected,ex=select_rows(full,eligible,parent)
    rows=attach_reset(selected,stage1)

    per={}
    for season in SEASONS:
        g=rows.loc[rows["season"].eq(season)].copy()
        act=g.loc[g["actionable_stage1_reset"]].copy()
        per[str(season)]={
            "selected_market_rows":int(len(g)),
            "distinct_stable_ids":int(g["stable_id"].nunique()),
            "stage1_exact_mechanism_rows":int(g["stage1_exact_mechanism_row"].sum()),
            "actionable_reset_rows":int(len(act)),
            "actionable_reset_stable_ids":int(act["stable_id"].nunique()),
            "canonical_zero_target_rows":int(g["canonical_zero_share"].sum()),
            "transition_kind_counts":{str(k):int(v) for k,v in g["transition_kind"].value_counts().sort_index().items()},
        }

    gates={
        "immutable_parent_contracts_verified_by_workflow":True,
        "eligible_transition_ids_nonblank":bool(len(eligible) and eligible["stable_id"].ne("").all()),
        "qualified_te_receiving_parent_state_link_rate_ge_099":rate>=0.99,
        "selected_transition_rows_unique":bool(len(rows) and not rows["transition_id"].duplicated().any()),
        "selected_chronology_valid":bool(
            len(rows) and rows["target_week"].ge(rows["transition_week"]).all() and
            (rows["next_transition_week"].isna()|rows["target_week"].lt(rows["next_transition_week"])).all()
        ),
        "stage1_exact_rows_match_parent_p_active":bool(
            len(rows) and rows.loc[rows["stage1_exact_mechanism_row"],"stage1_candidate_matches_current"].all()
        ),
        "stage1_support_flag_matches_exact_row":bool(len(rows) and rows["stage1_support_flag_matches_exact_row"].all()),
        "selected_targets_positive_canonical_entitlement":bool(len(rows) and rows["authority_q"].gt(0).all() and (~rows["canonical_zero_share"]).all()),
        "reset_probability_finite_open_interval_all":bool(
            len(rows) and np.isfinite(rows["reset_p_active"]).all() and
            rows["reset_p_active"].gt(0).all() and rows["reset_p_active"].lt(1).all()
        ),
        "development_2024_selected_rows_ge_30":per["2024"]["selected_market_rows"]>=30,
        "development_2024_distinct_ids_ge_25":per["2024"]["distinct_stable_ids"]>=25,
        "confirmation_2025_selected_rows_ge_30":per["2025"]["selected_market_rows"]>=30,
        "confirmation_2025_distinct_ids_ge_25":per["2025"]["distinct_stable_ids"]>=25,
        "development_2024_actionable_rows_ge_15":per["2024"]["actionable_reset_rows"]>=15,
        "development_2024_actionable_ids_ge_15":per["2024"]["actionable_reset_stable_ids"]>=15,
        "confirmation_2025_actionable_rows_ge_15":per["2025"]["actionable_reset_rows"]>=15,
        "confirmation_2025_actionable_ids_ge_15":per["2025"]["actionable_reset_stable_ids"]>=15,
        "names_read_or_used_zero":True,"predictive_outcomes_read_or_scored_zero":True,
        "target_game_opportunity_counts_read_zero":True,"sportsbook_inputs_zero":True,
        "oddsapi_calls_zero":True,"coefficients_fit_zero":True,"thresholds_searched_zero":True,
        "production_changed_false":True,
    }
    ready=all(gates.values())
    result={
        "version":VERSION,"freeze_commit":FREEZE_COMMIT,
        "disposition":"PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_READY" if ready else "PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_BLOCKED",
        "source_ready":bool(ready),
        "eligible_team_context_transitions":int(len(eligible)),
        "selected_first_market_rows":int(len(rows)),
        "source_exclusions":ex,
        "parent_state_link_hits":int(hits),"parent_state_link_attempts":int(attempts),"parent_state_link_rate":float(rate),
        "per_season":per,"source_gates":gates,
        "names_read_or_used":0,"predictive_outcomes_read_or_scored":0,
        "target_game_opportunity_counts_read":0,"sportsbook_inputs_used":0,"oddsapi_calls":0,
        "coefficients_fit":0,"thresholds_searched":0,"production_changed":False,
    }
    rows.to_csv(a.out_dir/"te_receiving_team_context_stage1_reset_source_rows.csv",index=False)
    (a.out_dir/"te_receiving_team_context_stage1_reset_source_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0

if __name__=="__main__": raise SystemExit(main())
