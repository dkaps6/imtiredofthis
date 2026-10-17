#!/usr/bin/env python3
"""PRISM Week-4/5 all-row player-first forensic replay V1 (diagnostic only)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

VERSION="PRISM_WEEK4_5_ALL_ROW_PLAYER_FIRST_FORENSIC_REPLAY_V1"
SEASON=2026
RATIO=1.5
CASE_N=10


def read_csv(p:Path,label:str)->pd.DataFrame:
    if not p.exists() or p.stat().st_size<=0: raise RuntimeError(f"missing/empty {label}: {p}")
    x=pd.read_csv(p,low_memory=False); x.columns=[str(c).strip().lower() for c in x.columns]; return x

def num(x): return pd.to_numeric(x,errors="coerce")
def pfam(x):
    p=str(x or "").upper();
    return "RB" if p in {"RB","HB","FB"} else ("WR" if p.startswith("WR") else ("TE" if p.startswith("TE") else ("QB" if p.startswith("QB") else (p or "UNKNOWN"))))
def sha(df): return "sha256:"+hashlib.sha256(df.to_csv(index=False,float_format="%.12f",lineterminator="\n").encode()).hexdigest()
def comp_label(o,e):
    try: o=float(o);e=float(e)
    except Exception:return "UNRESOLVED"
    if not np.isfinite(o) or not np.isfinite(e): return "UNRESOLVED"
    if abs(o)>=RATIO*abs(e): return "OPPORTUNITY_DOMINANT"
    if abs(e)>=RATIO*abs(o): return "EFFICIENCY_DOMINANT"
    return "MIXED_OPPORTUNITY_EFFICIENCY"


def board4(p):
    x=read_csv(p,"W4 board"); x["season"]=SEASON;x["week"]=4;x["position_family"]=x.position.map(pfam)
    x["replay_settlement_status"]=x.settlement_status.astype(str).str.upper();x["model_error"]=num(x.actual)-num(x.model_proj);x["abs_error"]=x.model_error.abs()
    x["board_authority"]="37485600879/11423535484"
    keep=["season","week","event_id","team","opponent","player","player_clean_key","position","position_family","market","model_proj","actual","model_error","abs_error","replay_settlement_status","board_authority","bayes_evidence_state","rules_role","qb_pred_attempts","qb_pred_ypa","identity_status","roster_status","snap_participated"]
    return x[[c for c in keep if c in x.columns]]
def board5(p):
    x=read_csv(p,"W5 board");x=x[(num(x.season)==SEASON)&(num(x.week)==5)].copy();x["position_family"]=x.position.map(pfam)
    x["replay_settlement_status"]=x.football_settlement_status.astype(str).str.upper();x["model_error"]=num(x.actual)-num(x.model_proj);x["abs_error"]=x.model_error.abs();x["board_authority"]="38395384228/11812200405; parent 37860592615/11585973062"
    keep=["season","week","event_id","team","opponent","player","player_clean_key","position","position_family","market","model_proj","actual","model_error","abs_error","replay_settlement_status","board_authority","identity_status","roster_status","snap_participated"]
    return x[[c for c in keep if c in x.columns]]


def prior_table(hist,board):
    h=hist[(num(hist.season)==SEASON)].copy();h["err"]=num(h.actual_output)-num(h.baseline_projection);h["ae"]=h.err.abs();rows=[]
    for r in board[["week","player_clean_key","market"]].drop_duplicates().itertuples(index=False):
        q=h[(h.player_clean_key.astype(str)==str(r.player_clean_key))&(h.market==r.market)&(num(h.week)<int(r.week))].sort_values("week")
        rows.append({"week":int(r.week),"player_clean_key":str(r.player_clean_key),"market":r.market,"prior_games":int(num(q.week).nunique()),"prior_first_week":num(q.week).min(),"prior_last_week":num(q.week).max(),"prior_mae":num(q.ae).mean(),"prior_mean_error":num(q.err).mean(),"prior_mean_actual_opportunity":num(q.actual_opportunities).mean(),"prior_last_actual_opportunity":num(q.actual_opportunities).iloc[-1] if len(q) else np.nan,"prior_mean_actual_efficiency":num(q.actual_efficiency).mean(),"prior_high_side_misses":int((q.err>0).sum()),"prior_low_side_misses":int((q.err<0).sum())})
    return pd.DataFrame(rows)

def w4_components(hist):
    h=hist[(num(hist.season)==SEASON)&(num(hist.week)==4)].copy();h=h.sort_values(["team","player_clean_key","market"]).drop_duplicates(["team","player_clean_key","market"],keep="last")
    pe=num(h.model_effective_efficiency);ae=num(h.actual_efficiency);pn=num(h.predicted_opportunities);an=num(h.actual_opportunities)
    h["opportunity_contribution"]=(an-pn)*(pe+ae)/2;h["efficiency_contribution"]=(ae-pe)*(pn+an)/2;h["component_label"]=[comp_label(o,e) for o,e in zip(h.opportunity_contribution,h.efficiency_contribution)]
    h["component_identity_gap"]=(h.opportunity_contribution+h.efficiency_contribution-(num(h.actual_output)-num(h.baseline_reconstructed))).abs()
    h["component_projection_gap_vs_final_placeholder"]=np.nan
    return h[["team","player_clean_key","market","predicted_opportunities","actual_opportunities","model_effective_efficiency","actual_efficiency","baseline_projection","component_label","opportunity_contribution","efficiency_contribution","component_identity_gap"]]
def w5_state(p):
    s=read_csv(p,"W5 state")
    for c in ["rules_tgt_share","rules_rush_share","rules_ypt","rules_ypc","rules_catch_rate","expected_targets","expected_carries","player_tgt_share","player_rush_share","off_proe","opp_def_pass_epa","opp_def_rush_epa","workload_score"]:
        if c in s:s[c]=num(s[c])
    s["target_room_rank"]=s.groupby("team").expected_targets.rank(method="min",ascending=False);s["carry_room_rank"]=s.groupby("team").expected_carries.rank(method="min",ascending=False)
    if s.duplicated(["team","player_clean_key"]).any():raise RuntimeError("duplicate W5 state player")
    return s
def w5_case(p):
    c=read_csv(p,"W5 casebook")
    keep=["team","player_clean_key","market","pred_opportunity","actual_opportunity","pred_efficiency","actual_efficiency","opportunity_contribution","efficiency_contribution","component_label","case_rank","pred_team_pass_att","pred_team_rush_att"]
    return c[[x for x in keep if x in c.columns]]


def annotate_layers(x):
    x["identity_history_status"]=np.where(x.prior_games.fillna(0)>0,"STRICT_PRIOR_SAME_PLAYER_MARKET_HISTORY_PRESENT","NO_CURRENT_SEASON_SAME_MARKET_HISTORY")
    x["current_role_state_status"]=np.where((x.week==5)&x.get("role",pd.Series(index=x.index,dtype=object)).notna(),"W5_PREGAME_PLAYER_STATE_TRACE_PRESENT","W4_HISTORY_PLUS_FROZEN_BOARD_ONLY")
    x["teammate_competition_status"]=np.where((x.week==5)&(x.get("target_room_rank",pd.Series(index=x.index)).notna()|x.get("carry_room_rank",pd.Series(index=x.index)).notna()),"FINITE_ROOM_RANK_VISIBLE__OFFICIAL_INACTIVE_IDENTITY_NOT_ROW_PRESERVED","ROW_LEVEL_ROOM_AVAILABILITY_SOURCE_NOT_PRESERVED")
    x["opportunity_type_quality_status"]=np.where(x.market=="rush_yards","RUSH_TYPE_KNOWN__HIGH_VALUE_SUBTYPE_NOT_ALL_ROW_PRESERVED",np.where(x.market.isin(["rec_yards","receptions"]),"TARGET_TYPE_KNOWN__ROUTE_DEPTH_HIGH_VALUE_SUBTYPE_NOT_ALL_ROW_PRESERVED",np.where(x.market=="pass_yards","PASS_ATTEMPT_TYPE_KNOWN","COMPOSITE_TYPE_PARTIAL")))
    x["team_game_environment_status"]=np.where((x.week==5)&x.get("off_proe",pd.Series(index=x.index)).notna(),"W5_OFFENSIVE_ENVIRONMENT_STATE_AVAILABLE","ENVIRONMENT_ROW_STATE_NOT_PRESERVED")
    x["player_conditioned_matchup_status"]=np.where((x.week==5)&(x.get("opp_def_pass_epa",pd.Series(index=x.index)).notna()|x.get("opp_def_rush_epa",pd.Series(index=x.index)).notna()),"GENERIC_OPPONENT_STATE_AVAILABLE__PLAYER_CONDITIONED_SOURCE_PARITY_BLOCKED","PLAYER_CONDITIONED_MATCHUP_SOURCE_BLOCKED_OR_UNAVAILABLE")
    x["reliability_status"]=np.where((x.week==5)&(x.position_family=="WR")&(x.prior_games==4)&(x.prior_first_week==1)&(x.prior_last_week==4),"W5_EXACT_4_PRIOR_RULE_ELIGIBLE__NO_SCORE_INVENTED_HERE",np.where((x.week==4)&(x.position_family=="WR"),"W4_EXACT_WR_DUAL_SCORE_FORBIDDEN_BY_CONTRACT",np.where(x.prior_games.fillna(0)>0,"STRICT_PRIOR_PLAYER_ERROR_HISTORY_VISIBLE__TRUST_ONLY","SPARSE_CURRENT_SEASON_HISTORY")))
    return x

def attribute(x):
    prim=[];sec=[]
    for r in x.itertuples(index=False):
        if str(r.replay_settlement_status)!="SETTLED":p="SOURCE_BLOCKED_OR_UNAVAILABLE";s="NONSETTLED_DNP_VOID__NOT_SCORED_AS_PROJECTION_ERROR"
        elif r.component_label=="OPPORTUNITY_DOMINANT":p="OPPORTUNITY_QUANTITY_MISREAD";s="NONE"
        elif r.component_label=="EFFICIENCY_DOMINANT":p="EFFICIENCY_MISREAD";s="NONE"
        elif r.component_label=="MIXED_OPPORTUNITY_EFFICIENCY":p="MIXED";s="OPPORTUNITY_AND_EFFICIENCY_BOTH_MATERIAL"
        else:p="SOURCE_BLOCKED_OR_UNAVAILABLE";s="FINAL_COMPONENT_ATTRIBUTION_NOT_RECOVERABLE_FROM_PRESERVED_AUTHORITY"
        if int(r.week)==5 and r.market in {"rush_yards","rec_yards","receptions"}:
            cur=getattr(r,"player_rush_share",np.nan) if r.market=="rush_yards" else getattr(r,"player_tgt_share",np.nan); rule=getattr(r,"rules_rush_share",np.nan) if r.market=="rush_yards" else getattr(r,"rules_tgt_share",np.nan)
            try:d=float(cur)-float(rule)
            except:d=np.nan
            if np.isfinite(d) and np.isfinite(r.model_error) and ((d>0 and r.model_error>0) or (d<0 and r.model_error<0)):s+=(";" if s!="NONE" else "")+"CURRENT_ROLE_STATE_DIRECTION_AGREES_WITH_MISS"
        prim.append(p);sec.append(s)
    x["primary_attribution"]=prim;x["secondary_attribution"]=sec;return x


def build_cases(x):
    z=x[(x.replay_settlement_status=="SETTLED")&x.abs_error.notna()].sort_values(["week","market","abs_error","player_clean_key"],ascending=[1,1,0,1]);z=z.groupby(["week","market"],group_keys=False).head(CASE_N).copy();z["case_selection_rule"]=f"TOP_{CASE_N}_ABS_ERROR_PER_WEEK_MARKET";return z

def controls(x,cases):
    z=x[(x.replay_settlement_status=="SETTLED")&x.abs_error.notna()].copy();z["proj_pct"]=z.groupby(["week","market"]).model_proj.rank(pct=True);z["prior_pct"]=z.groupby(["week","market"]).prior_mae.rank(pct=True);cut=z.groupby(["week","market"]).abs_error.quantile(.25).rename("cut");z=z.merge(cut,on=["week","market"]);pool=z[z.abs_error<=z.cut];rows=[];used=set()
    for c in cases.itertuples(index=False):
        q=pool[(pool.week==c.week)&(pool.market==c.market)&(pool.position_family==c.position_family)&~((pool.team==c.team)&(pool.player_clean_key.astype(str)==str(c.player_clean_key)))].copy()
        if q.empty:q=pool[(pool.week==c.week)&(pool.market==c.market)].copy()
        if q.empty:continue
        cp=z[(z.week==c.week)&(z.team==c.team)&(z.player_clean_key.astype(str)==str(c.player_clean_key))&(z.market==c.market)].iloc[0]
        q["dist"]=(q.proj_pct-cp.proj_pct).abs().fillna(.5)+(q.prior_pct-cp.prior_pct).abs().fillna(.5);q["used"]=[(int(r.week),str(r.player_clean_key),str(r.market)) in used for r in q.itertuples()];p=q.sort_values(["used","dist","abs_error","player_clean_key"]).iloc[0].copy();used.add((int(p.week),str(p.player_clean_key),str(p.market)));d=p.to_dict();d.update({"matched_case_week":int(c.week),"matched_case_player":c.player,"matched_case_team":c.team,"matched_case_market":c.market,"matched_case_abs_error":float(c.abs_error),"match_basis":"SAME_WEEK_MARKET_POSITION__LOWEST_ERROR_QUARTILE__NEAREST_PROJECTION_AND_PRIOR_MAE_PERCENTILES"});rows.append(d)
    return pd.DataFrame(rows)
def mech(cases,ctrl):
    rows=[]
    for pop,g0 in [("CASE",cases),("CONTROL",ctrl)]:
        for (w,m,a),g in g0.groupby(["week","market","primary_attribution"],dropna=False):rows.append({"population":pop,"week":w,"market":m,"primary_attribution":a,"rows":len(g),"unique_players":g.player_clean_key.nunique(),"median_abs_error":num(g.abs_error).median(),"median_prior_mae":num(g.prior_mae).median(),"role_direction_agreement_rows":g.secondary_attribution.astype(str).str.contains("CURRENT_ROLE_STATE_DIRECTION_AGREES_WITH_MISS",regex=False).sum()})
    return pd.DataFrame(rows)
def qa(x):
    z=x[(x.replay_settlement_status=="SETTLED")&x.model_error.notna()].copy();z["sq"]=z.model_error**2;q=z.groupby(["week","position_family","market"]).agg(rows=("player_clean_key","size"),unique_players=("player_clean_key","nunique"),mae=("abs_error","mean"),mse=("sq","mean"),median_abs_error=("abs_error","median")).reset_index();q["rmse"]=np.sqrt(q.mse);q["qa_boundary"]="DOWNSTREAM_ONLY";return q.drop(columns="mse")

def ledgers(inv):
    need=["feature","landscape_layer","production_consumption_status","historical_validation_status","current_season_validation_status","closed_reopen_rule","notes"]
    a=inv[inv.production_consumption_status.eq("AVAILABLE_BUT_NOT_CONSUMED")][need].copy();a["boundary"]="PREGAME_SIGNAL_EXISTS__NOT_CAUSALLY_ASSIGNED_FROM_W4_W5_OUTCOMES"
    b=inv[inv.production_consumption_status.isin(["SOURCE_PARITY_BLOCKED","NOT_AVAILABLE"])][need].copy();extra=pd.DataFrame([{"feature":"historical_row_level_official_inactive_room_identity","landscape_layer":"ROOM_HIERARCHY_COMPETITION","production_consumption_status":"REPLAY_PRESERVATION_GAP","historical_validation_status":"PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED","current_season_validation_status":"W4_W5_EXACT_ROOM_IDENTITY_NOT_ROW_PRESERVED","closed_reopen_rule":"DO_NOT_INFER_FROM_TARGET_GAME_PARTICIPATION","notes":"Needed for true teammate competition state."},{"feature":"exact_player_conditioned_role_subtype_matchup","landscape_layer":"INDIVIDUAL_MATCHUP_INTERACTION","production_consumption_status":"SOURCE_PARITY_BLOCKED","historical_validation_status":"GENERIC_DVP_FAMILIES_CLOSED_OR_BLOCKED","current_season_validation_status":"GENERIC_OPPONENT_STATE_ONLY","closed_reopen_rule":"NO_GENERIC_DEFENSE_VS_POSITION_PROXY","notes":"Future source must condition on named-player opportunity subtype."}]);b=pd.concat([b,extra],ignore_index=True);b["boundary"]="FAIL_CLOSED__DO_NOT_INVENT_OR_PROXY";return a,b

def hypotheses(cases):
    o=int(cases.primary_attribution.isin(["OPPORTUNITY_QUANTITY_MISREAD","MIXED"]).sum());e=int((cases.primary_attribution=="EFFICIENCY_MISREAD").sum());r=int(cases.secondary_attribution.astype(str).str.contains("CURRENT_ROLE_STATE_DIRECTION_AGREES_WITH_MISS",regex=False).sum())
    return pd.DataFrame([{"priority":1,"question":"Can PRISM arbitrate real named-player current-role regime change without raw recent-share replacement?","support":f"opportunity/mixed cases={o}; role-direction agreement={r}","boundary":"Freeze independently; raw recent-share replacement remains CLOSED."},{"priority":2,"question":"Where does strong upstream named-player role knowledge get attenuated before final market opportunity?","support":"Exact W5 top-40 plus prior Bowers trace show this architecture class.","boundary":"Trace deterministic stages; no W4/W5-fitted coefficient."},{"priority":3,"question":"Can exact pregame teammate availability + finite room competition identify role changes?","support":"Replay preservation gap and Room Competition State V1 source BLOCKED.","boundary":"Source audit first; do not lower gates."},{"priority":4,"question":"Can opponent interaction be conditioned on the named player's actual opportunity subtype?","support":"Generic opponent state exists; player-conditioned parity is blocked.","boundary":"No generic DvP rescue."},{"priority":5,"question":"Which named players warrant trust reduction because failure mechanism changes week to week?","support":f"efficiency-dominant informative cases={e}","boundary":"Trust only; no Week-4 fake WR dual score and no direction from reliability."}])

def main():
    ap=argparse.ArgumentParser();
    for n in ["w4_board","w5_board","w1_w4_components","w5_state","w5_casebook","feature_inventory"]:ap.add_argument("--"+n.replace("_","-"),dest=n,type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    h=read_csv(a.w1_w4_components,"W1-W4 components");w4=board4(a.w4_board);w5=board5(a.w5_board);base=pd.concat([w4,w5],ignore_index=True,sort=False)
    if len(base)!=836:raise RuntimeError(f"expected 836 all-row board rows, got {len(base)}")
    pri=prior_table(h,base);c4=w4_components(h);w4x=w4.merge(c4,on=["team","player_clean_key","market"],how="left",validate="one_to_one");w4x["component_authority"]=np.where(w4x.component_label.notna(),"W4_PRESERVED_COMPONENT_DECOMPOSITION","NO_W4_COMPONENT_AUTHORITY");w4x["component_projection_gap_vs_final"]=(num(w4x.baseline_projection)-num(w4x.model_proj)).abs()
    s=w5_state(a.w5_state);c5=w5_case(a.w5_casebook);w5x=w5.merge(s,on=["team","player_clean_key"],how="left",suffixes=("","_state"),validate="many_to_one").merge(c5,on=["team","player_clean_key","market"],how="left",validate="one_to_one");w5x["component_authority"]=np.where(w5x.component_label.notna(),"W5_EXACT_TOP40_CASEBOOK_COMPONENT_DECOMPOSITION","NO_EXACT_W5_ALL_ROW_COMPONENT_AUTHORITY__PREGAME_PARENT_ARTIFACT_EXPIRED");w5x["state_trace_expected_opportunity"]=np.where(w5x.market=="rush_yards",w5x.expected_carries,np.where(w5x.market.isin(["rec_yards","receptions"]),w5x.expected_targets,np.nan));w5x["state_trace_current_minus_rules_share"]=np.where(w5x.market=="rush_yards",w5x.player_rush_share-w5x.rules_rush_share,np.where(w5x.market.isin(["rec_yards","receptions"]),w5x.player_tgt_share-w5x.rules_tgt_share,np.nan))
    allx=pd.concat([w4x,w5x],ignore_index=True,sort=False);allx=allx.merge(pri,on=["week","player_clean_key","market"],how="left",validate="one_to_one");allx["component_label"]=allx.component_label.fillna("UNRESOLVED");allx["fragility_pct"]=allx.groupby(["week","market"]).prior_mae.rank(pct=True);allx=annotate_layers(allx);allx=attribute(allx);allx["abs_error_pct_within_week_market"]=allx.groupby(["week","market"]).abs_error.rank(pct=True)
    gap=num(allx.loc[(allx.week==4)&allx.component_identity_gap.notna(),"component_identity_gap"]);maxgap=float(gap.max()) if len(gap) else np.nan
    if len(gap) and maxgap>1e-8:raise RuntimeError(f"W4 component identity gap {maxgap}")
    cases=build_cases(allx);ctrl=controls(allx,cases);ms=mech(cases,ctrl);reps=cases.groupby(["player_clean_key","market"],group_keys=False).filter(lambda g:g.week.nunique()>=2).copy();inv=read_csv(a.feature_inventory,"feature inventory");notx,blocked=ledgers(inv);hyp=hypotheses(cases);q=qa(allx)
    outs={"prism_week4_5_all_row_player_first_replay.csv":allx,"prism_week4_5_named_player_informative_casebook.csv":cases,"prism_week4_5_matched_correct_controls.csv":ctrl,"prism_week4_5_repeated_mechanism_summary.csv":ms,"prism_week4_5_repeat_same_player_market_registry.csv":reps,"prism_week4_5_pregame_signals_not_transmitted.csv":notx,"prism_week4_5_source_blocked_player_conditioned_signals.csv":blocked,"prism_week4_5_candidate_football_hypotheses.csv":hyp,"prism_week4_5_aggregate_qa.csv":q}
    res={"version":VERSION,"disposition":"PRISM_WEEK4_5_ALL_ROW_PLAYER_FIRST_REPLAY_COMPLETE__DIAGNOSTIC_ONLY__NO_PROMOTION","all_row_replay_rows":len(allx),"week4_rows":int((allx.week==4).sum()),"week5_rows":int((allx.week==5).sum()),"settled_rows":int((allx.replay_settlement_status=="SETTLED").sum()),"nonsettled_void_dnp_rows":int((allx.replay_settlement_status!="SETTLED").sum()),"unique_named_players":allx.player_clean_key.nunique(),"component_rows":int((allx.component_label!="UNRESOLVED").sum()),"week4_component_rows":int(((allx.week==4)&(allx.component_label!="UNRESOLVED")).sum()),"week5_exact_top40_component_rows":int(((allx.week==5)&(allx.component_label!="UNRESOLVED")).sum()),"informative_case_rows":len(cases),"matched_correct_control_rows":len(ctrl),"repeat_case_rows":len(reps),"repeat_same_player_market_identities":int(reps[["player_clean_key","market"]].drop_duplicates().shape[0]) if len(reps) else 0,"w4_component_max_identity_gap":maxgap,"week5_wr_exact_four_prior_rule_eligible_rows":int(((allx.week==5)&(allx.position_family=="WR")&(allx.prior_games==4)&(allx.prior_first_week==1)&(allx.prior_last_week==4)).sum()),"week4_exact_wr_dual_scores_manufactured":0,"new_model_fit":0,"threshold_searches":0,"sportsbook_inputs_used_upstream":0,"sportsbook_refetch":False,"oddsapi_calls":0,"week6_outcomes_used":0,"week6_locks_changed":False,"production_changed":False,"raw_recent_share_replacement_reopened":False,"generic_dvp_reopened":False,"room_competition_gates_lowered":False,"known_limitation":"Week-5 parent artifact 11585973062 is no longer retrievable; exact W5 final component attribution is therefore limited to frozen top-40 casebook rows. All other W5 rows retain player-state trace without relabeling it as final production components.","scientific_object":"named player x target game x exact market","aggregate_error_role":"downstream QA only"}
    for name,d in outs.items():d.to_csv(a.out_dir/name,index=False,float_format="%.12f");res[name+"_sha256"]=sha(d)
    (a.out_dir/"prism_week4_5_all_row_player_first_replay_result.json").write_text(json.dumps(res,indent=2,sort_keys=True)+"\n");print(json.dumps(res,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
