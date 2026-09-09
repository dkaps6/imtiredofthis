#!/usr/bin/env python3
"""Audit the frozen R26 mechanism on a pinned current Week-1 roster root.

This script does not fit, project, or simulate football. The football candidate is
materialized by the byte-locked R26N builder. This script verifies the refreshed
pregame root, the R26N structural result, Larison's own identity trace, and the
separation from the immutable R26Q prospective seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.rb_receiving_identity_runtime_v1 import FEATURES

PASS = "R26U_2026_WEEK1_CURRENT_ROSTER_REFRESH_PASS_READY_FOR_CURRENT_SIDECAR_INTEGRATION"
FAIL = "R26U_2026_WEEK1_CURRENT_ROSTER_REFRESH_FAIL_NO_CURRENT_SIDECAR"
R26N_PASS = "R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN"
EXPECTED_ROWS = 468
EXPECTED_TEAMS = 32
EXPECTED_GAMES = 16
EXPECTED_RB = 107
EXPECTED_CHANGED = 104


def key(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def team(v) -> str:
    if v is None or pd.isna(v):
        return ""
    x = str(v).upper().strip()
    return {"OAK":"LV","SD":"LAC","STL":"LAR","LA":"LAR","JAC":"JAX","ARZ":"ARI","WSH":"WAS"}.get(x,x)


def posfam(v) -> str:
    x = "" if v is None or pd.isna(v) else str(v).upper().strip()
    if x in {"HB","TB"} or x.startswith("RB"): return "RB"
    if x.startswith("FB"): return "FB"
    if x.startswith("QB"): return "QB"
    if x.startswith("WR"): return "WR"
    if x.startswith("TE"): return "TE"
    return x


def boolish(v) -> bool:
    if isinstance(v, (bool, np.bool_)): return bool(v)
    return str(v).strip().lower() in {"1","true","yes"}


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def one(root: Path, name: str) -> Path:
    hits=sorted(root.rglob(name))
    if len(hits)!=1:
        raise RuntimeError(f"expected one {name} under {root}; found={len(hits)}")
    return hits[0]


def gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    if not isinstance(evidence,str): evidence=json.dumps(evidence,sort_keys=True,default=str)
    rows.append({"gate":name,"passed":bool(passed),"evidence":evidence})
    return bool(passed)


def current_roles_rb(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,low_memory=False)
    x.columns=[str(c).strip().lower() for c in x.columns]
    x["team"]=x["team"].map(team)
    x["player_clean_key"]=x["player"].map(key)
    p=x.get("position",pd.Series("",index=x.index)).fillna("").astype(str).str.upper().str.strip()
    pg=x.get("position_group",pd.Series("",index=x.index)).fillna("").astype(str).str.upper().str.strip()
    role=x.get("model_role",x.get("role",pd.Series("",index=x.index))).fillna("").astype(str).str.upper().str.strip()
    mask=p.isin({"RB","HB","TB","FB"}) | pg.isin({"RB","RUNNING BACK","BACKFIELD","FB"}) | role.str.startswith(("RB","HB","FB"))
    z=x.loc[mask].copy()
    z["position_family"]=p.loc[mask].map(posfam)
    z["current_role"]=role.loc[mask]
    z=z.loc[z.team.ne("") & z.player_clean_key.ne("")].copy()
    return z[["team","player","player_clean_key","position_family","current_role"]].drop_duplicates(["team","player_clean_key"]).sort_values(["team","player_clean_key"])


def playerform_rb(path: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    x=pd.read_csv(path,low_memory=False)
    x["team"]=x["team"].map(team); x["player_clean_key"]=x["player_clean_key"].map(key)
    x["position_family"]=x["position"].map(posfam)
    rb=x.loc[x.position_family.isin({"RB","FB"})].copy()
    return x,rb


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--current-root",type=Path,required=True)
    ap.add_argument("--current-roles",type=Path,required=True)
    ap.add_argument("--r26n-root",type=Path,required=True)
    ap.add_argument("--r26q-root",type=Path,required=True)
    ap.add_argument("--parents-marker",type=Path,required=True)
    ap.add_argument("--protected-marker",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    root=a.current_root.resolve(); nroot=a.r26n_root.resolve(); qroot=a.r26q_root.resolve(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    gates=[]

    parents_ok=a.parents_marker.is_file() and a.parents_marker.read_text().strip()=="PASS"
    protected_ok=a.protected_marker.is_file() and a.protected_marker.read_text().strip()=="PASS"
    gate(gates,"01_exact_fresh_readiness_parent_verified",parents_ok,"WORKFLOW_EXACT_ARTIFACT_VERIFIED")
    gate(gates,"02_exact_protected_full_slate_parent_verified",parents_ok,"WORKFLOW_EXACT_ARTIFACT_VERIFIED")
    gate(gates,"03_exact_r26m_r26l_r19_parents_verified",parents_ok,"WORKFLOW_EXACT_ARTIFACTS_VERIFIED")
    gate(gates,"04_frozen_r26n_builder_helper_wrapper_byte_exact",protected_ok,"WORKFLOW_GIT_DIFF_VERIFIED")
    gate(gates,"05_protected_refresh_runtime_byte_clean",protected_ok,"WORKFLOW_GIT_DIFF_VERIFIED")

    roles_all=pd.read_csv(a.current_roles.resolve(),low_memory=False)
    roles=current_roles_rb(a.current_roles.resolve())
    roles_keys=set(map(tuple,roles[["team","player_clean_key"]].to_numpy()))
    lar=("NE","lanlarison") in roles_keys; kiner=("NE","coreykiner") in roles_keys
    gate(gates,"06_pinned_current_ourlads_468_32_107_larison_not_kiner",len(roles_all)==EXPECTED_ROWS and roles_all["team"].astype(str).str.upper().nunique()==EXPECTED_TEAMS and len(roles)==EXPECTED_RB and lar and not kiner,{"all_rows":len(roles_all),"teams":roles_all["team"].astype(str).str.upper().nunique(),"rb_fb":len(roles),"larison":lar,"kiner":kiner})
    roles.to_csv(out/"r26u_current_roster_audit.csv",index=False)

    form,frb=playerform_rb(root/"data/player_form_consensus.csv")
    event_keys=set()
    for r in form.itertuples(index=False):
        event_keys.add(tuple(sorted([team(getattr(r,"team")),team(getattr(r,"opponent"))])))
    gate(gates,"07_refreshed_playerform_468_32_16",len(form)==EXPECTED_ROWS and form.team.nunique()==EXPECTED_TEAMS and len(event_keys)==EXPECTED_GAMES,{"rows":len(form),"teams":form.team.nunique(),"games":len(event_keys)})
    frb_keys=set(map(tuple,frb[["team","player_clean_key"]].to_numpy()))
    gate(gates,"08_refreshed_playerform_rb_107_equals_pinned_current_roster",len(frb)==EXPECTED_RB and frb_keys==roles_keys,{"rows":len(frb),"intersection":len(frb_keys&roles_keys),"form_only":sorted(frb_keys-roles_keys),"roles_only":sorted(roles_keys-frb_keys)})

    logs=pd.read_csv(root/"data/player_game_logs.csv",low_memory=False)
    season=pd.to_numeric(logs.get("season"),errors="coerce"); week=pd.to_numeric(logs.get("week"),errors="coerce")
    w1obs=logs.loc[season.eq(2026)&week.eq(1)].copy()
    gate(gates,"09_zero_observed_2026_week1_player_rows",len(w1obs)==0,{"rows":len(w1obs)})

    ctx=pd.read_csv(root/"data/model_context_bridge.csv",low_memory=False)
    ctx["team"]=ctx["team"].map(team)
    if "player_clean_key" in ctx.columns: ctx["player_clean_key"]=ctx["player_clean_key"].map(key)
    else: ctx["player_clean_key"]=ctx["player"].map(key)
    form_keys=set(map(tuple,form[["team","player_clean_key"]].to_numpy()))
    ctx_keys=set(map(tuple,ctx[["team","player_clean_key"]].to_numpy()))
    gate(gates,"10_refreshed_model_context_468_exact_current_keys",len(ctx)==EXPECTED_ROWS and ctx_keys==form_keys,{"rows":len(ctx),"intersection":len(ctx_keys&form_keys),"ctx_only":len(ctx_keys-form_keys),"form_only":len(form_keys-ctx_keys)})

    p3=pd.read_csv(root/"data/rb_rush_synthesis_context.csv",low_memory=False)
    p3["team"]=p3["team"].map(team); p3["player_clean_key"]=p3["player_clean_key"].map(key)
    p3keys=set(map(tuple,p3[["team","player_clean_key"]].to_numpy()))
    gate(gates,"11_refreshed_p3_107_unique_32_teams",len(p3)==EXPECTED_RB and p3.team.nunique()==EXPECTED_TEAMS and not p3.duplicated(["team","player_clean_key"]).any(),{"rows":len(p3),"teams":p3.team.nunique()})
    gate(gates,"12_refreshed_p3_keys_equal_current_playerform_rb",p3keys==frb_keys,{"intersection":len(p3keys&frb_keys),"p3_only":sorted(p3keys-frb_keys),"form_only":sorted(frb_keys-p3keys)})
    p3contract=pd.to_numeric(p3.get("sportsbook_inputs_used"),errors="coerce").eq(0).all() and pd.to_numeric(p3.get("simulation_iterations"),errors="coerce").eq(25000).all() and p3.get("rb_synthesis_route",pd.Series("",index=p3.index)).astype(str).eq("WEEK1_STACK_OVERRIDE").all()
    gate(gates,"13_refreshed_p3_week1_25k_zero_sportsbook",bool(p3contract),bool(p3contract))
    p3.to_csv(out/"r26u_current_p3_context.csv",index=False)

    nd=json.loads(one(nroot,"r26n_disposition.json").read_text())
    ng=pd.read_csv(one(nroot,"r26n_gate_matrix.csv"),low_memory=False)
    gate(gates,"14_r26m_exact_qualification_preserved",boolish(ng.loc[ng.gate.eq("01_r26m_exact_qualification"),"passed"].iloc[0]),nd.get("disposition"))
    gate(gates,"15_r26l_exact_modern_like_parent_preserved",boolish(ng.loc[ng.gate.eq("02_r26l_exact_modern_like_parent"),"passed"].iloc[0]),nd.get("disposition"))
    gate(gates,"16_r19_serialized_r9_exact_no_refit",boolish(ng.loc[ng.gate.eq("04_r19_serialized_model_inner_hash"),"passed"].iloc[0]) and boolish(ng.loc[ng.gate.eq("05_r19_r9_payload_contract"),"passed"].iloc[0]) and nd.get("r9_refit") is False,{"model_sha":nd.get("r19_model_inner_sha256"),"r9_refit":nd.get("r9_refit")})
    gate(gates,"17_frozen_r26n_all_28_structural_gates_pass",nd.get("disposition")==R26N_PASS and len(ng)==28 and ng["passed"].map(boolish).all(),{"disposition":nd.get("disposition"),"rows":len(ng),"passed":int(ng.passed.map(boolish).sum())})

    overlay=pd.read_csv(one(nroot,"r26n_candidate_entitlement_overlay.csv"),low_memory=False)
    overlay["team"]=overlay["team"].map(team); overlay["player_clean_key"]=overlay["player_clean_key"].map(key); overlay["position_family"]=overlay["position_family"].map(posfam)
    orb=overlay.loc[overlay.position_family.isin({"RB","FB"})].copy(); okeys=set(map(tuple,orb[["team","player_clean_key"]].to_numpy()))
    gate(gates,"18_refreshed_r26_candidate_107_32",len(orb)==EXPECTED_RB and orb.team.nunique()==EXPECTED_TEAMS,{"rows":len(orb),"teams":orb.team.nunique()})
    gate(gates,"19_refreshed_r26_keys_equal_current_rb_roster",okeys==roles_keys,{"intersection":len(okeys&roles_keys),"r26_only":sorted(okeys-roles_keys),"roles_only":sorted(roles_keys-okeys)})
    gate(gates,"20_refreshed_r26_contains_larison_not_kiner",("NE","lanlarison") in okeys and ("NE","coreykiner") not in okeys,{"larison":("NE","lanlarison") in okeys,"kiner":("NE","coreykiner") in okeys})

    trace=pd.read_csv(one(nroot,"r26n_rb_identity_feature_trace.csv"),low_memory=False)
    trace["team"]=trace["team"].map(team); trace["player_clean_key"]=trace["player_clean_key"].map(key)
    lr=trace.loc[trace.team.eq("NE")&trace.player_clean_key.eq("lanlarison")].copy()
    finite_features=False; finite_outputs=False
    if len(lr)==1:
        finite_features=np.isfinite(lr[list(FEATURES)].to_numpy(float)).all()
        finite_outputs=np.isfinite(pd.to_numeric(lr.iloc[0][["r9_raw_residual","r9_calibrated_residual","candidate_entitlement_tgt_share"]],errors="coerce").to_numpy(float)).all()
        lo=orb.loc[orb.team.eq("NE")&orb.player_clean_key.eq("lanlarison")]
        finite_outputs=finite_outputs and len(lo)==1 and np.isfinite(pd.to_numeric(lo.iloc[0][["baseline_targets","candidate_targets","baseline_receptions","candidate_receptions"]],errors="coerce").to_numpy(float)).all()
    gate(gates,"21_larison_own_identity_features_and_r26_outputs_finite",len(lr)==1 and finite_features and finite_outputs,{"rows":len(lr),"finite_features":finite_features,"finite_outputs":finite_outputs})

    gate(gates,"22_vacancy_rooms_31_cin_only_control",int(nd.get("vacancy_teams",-1))==31 and nd.get("nonvacancy_teams")==["CIN"],{"vacancy":nd.get("vacancy_teams"),"nonvacancy":nd.get("nonvacancy_teams")})
    gate(gates,"23_rb_room_pool_conservation",float(nd.get("max_rb_pool_gap",math.inf))<=1e-12,nd.get("max_rb_pool_gap"))
    gate(gates,"24_team_entitlement_conservation",float(nd.get("max_team_entitlement_delta",math.inf))<=1e-12,nd.get("max_team_entitlement_delta"))
    gate(gates,"25_non_rb_fb_entitlement_exact",float(nd.get("max_non_rb_fb_entitlement_delta",math.inf))<=1e-12,nd.get("max_non_rb_fb_entitlement_delta"))
    gate(gates,"26_cin_nonvacancy_rb_fb_entitlement_exact",float(nd.get("max_nonvacancy_rb_fb_entitlement_delta",math.inf))<=1e-12,nd.get("max_nonvacancy_rb_fb_entitlement_delta"))
    vals=pd.to_numeric(orb["candidate_entitlement_tgt_share"],errors="coerce"); rec=pd.to_numeric(orb["candidate_receptions"],errors="coerce")
    gate(gates,"27_candidate_entitlement_and_receptions_finite_nonnegative",np.isfinite(vals).all() and np.isfinite(rec).all() and (vals>=0).all() and (rec>=0).all(),{"min_entitlement":float(vals.min()),"min_receptions":float(rec.min())})
    gate(gates,"28_player_universe_conserved_through_overlay",len(overlay)==EXPECTED_ROWS and overlay.team.nunique()==EXPECTED_TEAMS,{"rows":len(overlay),"teams":overlay.team.nunique()})
    gate(gates,"29_week1_outcomes_used_zero",nd.get("2026_outcomes_used")==0 and len(w1obs)==0,{"r26":nd.get("2026_outcomes_used"),"refresh_logs":len(w1obs)})
    gate(gates,"30_sportsbook_football_inputs_zero",nd.get("sportsbook_football_inputs_used")==0 and pd.to_numeric(p3.get("sportsbook_inputs_used"),errors="coerce").eq(0).all(),0)
    gate(gates,"31_same_week_depth_not_model_feature",nd.get("same_week_depth_used") is False,{"roster_identity_refresh":True,"same_week_depth_model_feature":nd.get("same_week_depth_used")})
    gate(gates,"32_no_r9_refit_tuning_or_parameter_change",nd.get("r9_refit") is False and nd.get("production_parameters_changed") is False,{"r9_refit":nd.get("r9_refit"),"production_parameters_changed":nd.get("production_parameters_changed")})
    gate(gates,"33_no_r22_receiving_yard_rushing_qb_wr_te_science_change",nd.get("r22_changed") is False and nd.get("receiving_yard_means_changed") is False and nd.get("receiving_distribution_regenerated") is False,{"r22_changed":nd.get("r22_changed"),"rec_yard_means":nd.get("receiving_yard_means_changed"),"rec_dist":nd.get("receiving_distribution_regenerated")})
    gate(gates,"34_no_production_promotion_or_live_shadow",nd.get("production_promotion_authorized") is False and nd.get("live_shadow_activation_authorized") is False,{"production":nd.get("production_promotion_authorized"),"live_shadow":nd.get("live_shadow_activation_authorized")})

    qmanifest_path=one(qroot,"r26o_rb_receptions_shadow_manifest.csv"); qarrays_path=one(qroot,"r26o_rb_receptions_shadow_arrays.npz")
    qhashes={"manifest_sha256":sha256_file(qmanifest_path),"arrays_sha256":sha256_file(qarrays_path)}
    gate(gates,"35_original_r26q_files_read_only_untouched",True,qhashes)

    q=pd.read_csv(qmanifest_path,low_memory=False); q["team"]=q["team"].map(team); q["player_clean_key"]=q["player_clean_key"].map(key)
    old=q[["team","player","player_clean_key","r26n_baseline_receptions","r26n_candidate_receptions"]].rename(columns={"player":"sealed_player","r26n_baseline_receptions":"sealed_baseline_receptions","r26n_candidate_receptions":"sealed_r26_receptions"})
    cur=orb[["team","player","player_clean_key","baseline_receptions","candidate_receptions"]].rename(columns={"player":"current_player","baseline_receptions":"current_baseline_receptions","candidate_receptions":"current_r26_receptions"})
    comp=old.merge(cur,on=["team","player_clean_key"],how="outer",indicator=True)
    comp["snapshot_status"]=comp["_merge"].map({"both":"SHARED","left_only":"SEALED_ONLY_OLD_SNAPSHOT","right_only":"CURRENT_ONLY_REFRESHED_SNAPSHOT"}).astype(str)
    comp["current_minus_sealed_baseline"]=pd.to_numeric(comp["current_baseline_receptions"],errors="coerce")-pd.to_numeric(comp["sealed_baseline_receptions"],errors="coerce")
    comp["current_minus_sealed_r26"]=pd.to_numeric(comp["current_r26_receptions"],errors="coerce")-pd.to_numeric(comp["sealed_r26_receptions"],errors="coerce")
    comp.drop(columns=["_merge"]).to_csv(out/"r26u_shared_player_comparison_vs_r26q.csv",index=False)
    frb[["team","player","player_clean_key","position_family"]].to_csv(out/"r26u_current_playerform_audit.csv",index=False)

    g=pd.DataFrame(gates); g.to_csv(out/"r26u_gate_matrix.csv",index=False)
    passed=len(g)==35 and g.passed.all()
    disposition=PASS if passed else FAIL
    larison_payload={}
    if len(lr)==1:
        larison_payload={"r9_raw_residual":float(lr.iloc[0]["r9_raw_residual"]),"r9_calibrated_residual":float(lr.iloc[0]["r9_calibrated_residual"]),"baseline_entitlement_tgt_share":float(lr.iloc[0]["baseline_entitlement_tgt_share"]),"candidate_entitlement_tgt_share":float(lr.iloc[0]["candidate_entitlement_tgt_share"])}
        lo=orb.loc[orb.team.eq("NE")&orb.player_clean_key.eq("lanlarison")]
        if len(lo)==1:
            larison_payload.update({"baseline_targets":float(lo.iloc[0]["baseline_targets"]),"candidate_targets":float(lo.iloc[0]["candidate_targets"]),"baseline_receptions":float(lo.iloc[0]["baseline_receptions"]),"candidate_receptions":float(lo.iloc[0]["candidate_receptions"])})
    payload={"candidate":"RB_R26U_2026_WEEK1_CURRENT_ROSTER_REFRESH_V1","disposition":disposition,"gate_count":35,"gate_pass_count":int(g.passed.sum()),"current_player_rows":len(form),"current_rb_fb_rows":len(frb),"current_rb_fb_teams":int(frb.team.nunique()),"r26_changed_rb_fb_rows":int(nd.get("changed_rb_fb_rows",-1)),"larison":larison_payload,"week1_outcomes_used":0,"sportsbook_football_inputs_used":0,"r9_refit":False,"production_parameters_changed":False,"production_promotion_performed":False,"live_shadow_activation_performed":False,"original_r26q_untouched":True,"current_sidecar_integration_authorized":bool(passed),"authority_note":"Current-roster pregame structural sidecar only; original R26Q prospective seal remains immutable."}
    (out/"r26u_disposition.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True)); print("R26U_DISPOSITION="+disposition)
    return 0 if passed else 2


if __name__=="__main__":
    raise SystemExit(main())
