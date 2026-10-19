#!/usr/bin/env python3
"""Source-only audit for RB rushing team-context Stage-1 participation reset V1."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scripts._opponent_map import canon_team

VERSION="PRISM_RB_RUSHING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE"
FREEZE_COMMIT="7a27318b2a6cf4c4f0b235429e653f63a72714a8"
SEASONS=(2024,2025); TRANSITION_KINDS={"OFFSEASON_TEAM_CHANGE","INSEASON_TEAM_CHANGE"}; TOL=1e-12

def txt(v:Any)->str:
    if v is None or pd.isna(v): return ""
    s=str(v).strip(); return "" if s.upper() in {"","NAN","NONE","<NA>"} else s
def bs(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype("string").str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False}).fillna(False).astype(bool)
def rc(path:Path,label:str,cols:list[str])->pd.DataFrame:
    h=pd.read_csv(path,nrows=0); mp={str(c).strip().lower():c for c in h.columns}
    miss=[c for c in cols if c not in mp]
    if miss: raise RuntimeError(f"{label} missing {miss}")
    x=pd.read_csv(path,usecols=[mp[c] for c in cols],low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def prep_events(path:Path):
    x=rc(path,"events",["season","week","team","stable_id","family","transition_kind","prior_team_support_ge4"])
    x["season"]=pd.to_numeric(x["season"],errors="coerce"); x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team); x["stable_id"]=x["stable_id"].map(txt)
    x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["transition_kind"]=x["transition_kind"].astype("string").fillna("").str.strip()
    x["prior_team_support_ge4"]=bs(x["prior_team_support_ge4"])
    full=x.loc[x["season"].isin(SEASONS)&x["family"].eq("RB")&x["transition_kind"].ne("")&x["stable_id"].ne("")].copy()
    full["season"]=full["season"].astype(int); full["week"]=full["week"].astype(int)
    e=full.loc[full["transition_kind"].isin(TRANSITION_KINDS)&full["prior_team_support_ge4"]].copy().sort_values(["season","stable_id","week","team"],kind="mergesort").reset_index(drop=True)
    e["transition_id"]=[f"{int(s)}:{int(w)}:{t}:{pid}:{kind}" for s,w,t,pid,kind in e[["season","week","team","stable_id","transition_kind"]].itertuples(index=False,name=None)]
    return full,e

def prep_states(path:Path):
    x=rc(path,"states",["season","week","team","stable_id","family"])
    x["season"]=pd.to_numeric(x["season"],errors="coerce");x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team);x["stable_id"]=x["stable_id"].map(txt);x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x=x.loc[x["season"].isin(SEASONS)&x["family"].eq("RB")&x["stable_id"].ne("")].copy()
    x["season"]=x["season"].astype(int);x["week"]=x["week"].astype(int);return x

def prep_parent(path:Path):
    x=rc(path,"parent",["season","week","game_key","team","cell","family","player_id","stage1_p_active","stage1_named_player_supported","authority_q","canonical_zero_share"])
    x["season"]=pd.to_numeric(x["season"],errors="coerce");x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team);x["player_id"]=x["player_id"].map(txt);x["game_key"]=x["game_key"].map(txt)
    x["cell"]=x["cell"].astype("string").fillna("").str.strip();x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    x["stage1_p_active"]=pd.to_numeric(x["stage1_p_active"],errors="coerce");x["authority_q"]=pd.to_numeric(x["authority_q"],errors="coerce")
    x["stage1_named_player_supported"]=bs(x["stage1_named_player_supported"]);x["canonical_zero_share"]=bs(x["canonical_zero_share"])
    x=x.loc[x["season"].isin(SEASONS)&x["cell"].eq("RB_RUSHING")&x["family"].eq("RB")&x["player_id"].ne("")].copy()
    x["season"]=x["season"].astype(int);x["week"]=x["week"].astype(int)
    return x

def prep_stage1(path:Path):
    x=rc(path,"stage1",["season","week","team","family","player_id","p_baseline","p_candidate","n_player_prior"])
    x["season"]=pd.to_numeric(x["season"],errors="coerce");x["week"]=pd.to_numeric(x["week"],errors="coerce")
    x["team"]=x["team"].map(canon_team);x["player_id"]=x["player_id"].map(txt);x["family"]=x["family"].astype("string").fillna("").str.upper().str.strip()
    for c in ("p_baseline","p_candidate","n_player_prior"):x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.loc[x["season"].isin(SEASONS)&x["family"].eq("RB")&x["player_id"].ne("")].copy()
    x["season"]=x["season"].astype(int);x["week"]=x["week"].astype(int);return x

def select(full,e,parent):
    positive=parent.loc[parent["authority_q"].gt(TOL)&(~parent["canonical_zero_share"])].copy()
    groups={(int(s),str(pid)):g.sort_values(["week","team"],kind="mergesort") for (s,pid),g in full.groupby(["season","stable_id"],sort=False)}
    out=[];ex={"NO_POSITIVE_CANONICAL_RB_RUSHING_BASELINE":0}
    for r in e.itertuples(index=False):
        later=groups.get((int(r.season),str(r.stable_id))); nxt=None
        if later is not None:
            ql=later.loc[later["week"].gt(int(r.week))]
            if len(ql):nxt=int(ql["week"].min())
        q=positive.loc[positive["season"].eq(int(r.season))&positive["player_id"].eq(str(r.stable_id))&positive["team"].eq(str(r.team))&positive["week"].ge(int(r.week))].copy()
        if nxt is not None:q=q.loc[q["week"].lt(nxt)]
        if q.empty:ex["NO_POSITIVE_CANONICAL_RB_RUSHING_BASELINE"]+=1;continue
        f=q.sort_values(["week","game_key"],kind="mergesort").iloc[0]
        out.append({"transition_id":str(r.transition_id),"season":int(r.season),"transition_week":int(r.week),"team":str(r.team),"stable_id":str(r.stable_id),"transition_kind":str(r.transition_kind),"next_transition_week":nxt,"target_week":int(f["week"]),"game_key":str(f["game_key"]),"stage1_p_active":float(f["stage1_p_active"]),"stage1_named_player_supported":bool(f["stage1_named_player_supported"]),"authority_q":float(f["authority_q"])})
    return pd.DataFrame(out),ex

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--events",type=Path,required=True);ap.add_argument("--states",type=Path,required=True);ap.add_argument("--parent",type=Path,required=True);ap.add_argument("--stage1",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    full,e=prep_events(a.events);states=prep_states(a.states);parent=prep_parent(a.parent);stage1=prep_stage1(a.stage1)
    sk=set(map(tuple,states[["season","week","team","stable_id"]].itertuples(index=False,name=None)))
    hits=sum((int(s),int(w),str(t),str(pid)) in sk for s,w,t,pid in parent[["season","week","team","player_id"]].itertuples(index=False,name=None));attempts=len(parent);rate=hits/attempts if attempts else 0
    z,ex=select(full,e,parent)
    s=stage1.rename(columns={"week":"target_week","player_id":"stable_id","p_baseline":"stage1_p_baseline","p_candidate":"stage1_p_candidate","n_player_prior":"stage1_player_prior_rows"})[["season","target_week","team","stable_id","stage1_p_baseline","stage1_p_candidate","stage1_player_prior_rows"]]
    z=z.merge(s,on=["season","target_week","team","stable_id"],how="left",validate="many_to_one")
    z["stage1_exact_mechanism_row"]=z["stage1_p_candidate"].notna();m=z["stage1_exact_mechanism_row"]
    z["stage1_candidate_matches_current"]=True;z.loc[m,"stage1_candidate_matches_current"]=(z.loc[m,"stage1_p_candidate"]-z.loc[m,"stage1_p_active"]).abs().le(TOL)
    z["stage1_support_flag_matches_exact_row"]=z["stage1_named_player_supported"].eq(z["stage1_exact_mechanism_row"])
    z["reset_p_active"]=z["stage1_p_baseline"].where(m,z["stage1_p_active"])
    z["actionable_stage1_reset"]=(z["reset_p_active"]-z["stage1_p_active"]).abs().gt(TOL)
    per={}
    for season in SEASONS:
        g=z.loc[z["season"].eq(season)];act=g.loc[g["actionable_stage1_reset"]]
        per[str(season)]={"selected_market_rows":int(len(g)),"distinct_stable_ids":int(g["stable_id"].nunique()),"stage1_exact_mechanism_rows":int(g["stage1_exact_mechanism_row"].sum()),"actionable_reset_rows":int(len(act)),"actionable_reset_stable_ids":int(act["stable_id"].nunique())}
    gates={
      "immutable_parent_contracts_verified_by_workflow":True,"eligible_transition_ids_nonblank":bool(len(e) and e["stable_id"].ne("").all()),
      "qualified_rb_rushing_parent_state_link_rate_ge_099":rate>=.99,
      "selected_rows_unique":bool(len(z) and not z["transition_id"].duplicated().any()),
      "selected_chronology_valid":bool(len(z) and z["target_week"].ge(z["transition_week"]).all() and (z["next_transition_week"].isna()|z["target_week"].lt(z["next_transition_week"])).all()),
      "stage1_exact_rows_match_parent":bool(len(z) and z.loc[z["stage1_exact_mechanism_row"],"stage1_candidate_matches_current"].all()),
      "stage1_support_flag_matches_exact_row":bool(len(z) and z["stage1_support_flag_matches_exact_row"].all()),
      "selected_targets_positive_entitlement":bool(len(z) and z["authority_q"].gt(TOL).all()),
      "reset_probability_open_interval":bool(len(z) and np.isfinite(z["reset_p_active"]).all() and z["reset_p_active"].gt(0).all() and z["reset_p_active"].lt(1).all()),
      "development_2024_selected_rows_ge_30":per["2024"]["selected_market_rows"]>=30,"development_2024_ids_ge_25":per["2024"]["distinct_stable_ids"]>=25,
      "confirmation_2025_selected_rows_ge_30":per["2025"]["selected_market_rows"]>=30,"confirmation_2025_ids_ge_25":per["2025"]["distinct_stable_ids"]>=25,
      "development_2024_actionable_rows_ge_15":per["2024"]["actionable_reset_rows"]>=15,"development_2024_actionable_ids_ge_15":per["2024"]["actionable_reset_stable_ids"]>=15,
      "confirmation_2025_actionable_rows_ge_15":per["2025"]["actionable_reset_rows"]>=15,"confirmation_2025_actionable_ids_ge_15":per["2025"]["actionable_reset_stable_ids"]>=15,
      "names_read_zero":True,"predictive_outcomes_read_zero":True,"target_game_counts_read_zero":True,"sportsbook_inputs_zero":True,"oddsapi_calls_zero":True,"coefficients_fit_zero":True,"thresholds_searched_zero":True,"production_changed_false":True}
    ready=all(gates.values())
    result={"version":VERSION,"freeze_commit":FREEZE_COMMIT,"disposition":"PRISM_RB_RUSHING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_READY" if ready else "PRISM_RB_RUSHING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_BLOCKED","source_ready":ready,"eligible_team_context_transitions":int(len(e)),"selected_first_positive_market_rows":int(len(z)),"source_exclusions":ex,"parent_state_link_hits":int(hits),"parent_state_link_attempts":int(attempts),"parent_state_link_rate":float(rate),"per_season":per,"source_gates":gates,"predictive_outcomes_read":0,"target_game_counts_read":0,"production_changed":False}
    z.to_csv(a.out_dir/"rb_rushing_team_context_stage1_reset_source_rows.csv",index=False);(a.out_dir/"rb_rushing_team_context_stage1_reset_source_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
