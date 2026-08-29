#!/usr/bin/env python3
"""Migration 69 — QB game-script forensic atlas.

M69 is a discovery/diagnostic migration, not a predictive model migration.
Postgame information is intentionally allowed to explain *what happened*.
Any statement about what was knowable before kickoff is restricted to columns
that were already present pregame (M65/M68 feature tables). No M69 output is
eligible for direct production promotion.

Questions:
1) Which football mechanisms create Raw QB passing-yards misses, especially 75+/100+?
2) When actual opening behavior deviates from the verified playcaller's prior
   opening tendency, do pregame defensive profiles explain the deviation?
3) How stable are pregame defensive scheme summaries versus the defense actually
   played in the target game?

Frozen mechanism priority (first applicable wins for volume-driven misses):
  role/participation -> planned opening pass/run deviation -> forced trailing/
  leading occupancy -> possession explosion/collapse -> attempt-conversion loss
  -> run-efficiency takeover/failure -> other volume.
Efficiency-dominant misses are labeled YPA explosion/collapse separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team


def num(v): return pd.to_numeric(v, errors="coerce")
def lower(df):
    x=df.copy(); x.columns=[str(c).strip().lower() for c in x.columns]; return x
def to_pd(o):
    if isinstance(o,pd.DataFrame): return o.copy()
    if hasattr(o,"to_pandas"): return o.to_pandas()
    return pd.DataFrame(o)
def canon(v):
    t=canon_team(v); return "WAS" if t=="WSH" else t

def safe_corr(a,b):
    z=pd.DataFrame({"a":num(a),"b":num(b)}).dropna()
    return float(z.a.corr(z.b)) if len(z)>=3 and z.a.nunique()>1 and z.b.nunique()>1 else np.nan

def load_sources(seasons):
    import nflreadpy as nfl
    pbps=[]; parts=[]; manifest=[]
    for s in sorted(set(map(int,seasons))):
        try:
            q=lower(to_pd(nfl.load_pbp(seasons=[s])))
            if len(q): pbps.append(q)
            manifest.append({"season":s,"family":"pbp","status":"recovered" if len(q) else "empty"})
        except Exception as exc:
            manifest.append({"season":s,"family":"pbp","status":f"failed:{type(exc).__name__}"})
        try:
            p=lower(to_pd(nfl.load_participation(seasons=[s])))
            if len(p): parts.append(p)
            manifest.append({"season":s,"family":"participation_realized_scheme","status":"recovered" if len(p) else "empty"})
        except Exception as exc:
            manifest.append({"season":s,"family":"participation_realized_scheme","status":f"failed:{type(exc).__name__}"})
    pbp=pd.concat(pbps,ignore_index=True,sort=False) if pbps else pd.DataFrame()
    part=pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()
    if pbp.empty: raise RuntimeError("M69 requires target-season PBP to reconstruct actual game script")
    return pbp,part,pd.DataFrame(manifest)

def regular(x):
    q=x.copy(); c="season_type" if "season_type" in q else "game_type" if "game_type" in q else None
    if c:
        s=q[c].astype(str).str.upper(); r=q[s.isin(["REG","REGULAR","RS",""])].copy()
        if len(r): q=r
    return q

def pbp_team_games(pbp):
    x=regular(pbp)
    for c in ["season","week","game_id","play_id","posteam","defteam","qb_dropback","pass_attempt","rush_attempt","down","ydstogo","score_differential","qtr","drive","epa","success","sack","qb_scramble","turnover","interception","fumble_lost","touchdown","return_touchdown","special_teams_play"]:
        if c not in x: x[c]=np.nan
    x["team"]=x.posteam.map(canon); x["opponent"]=x.defteam.map(canon)
    x=x[x.team.ne("") & x.opponent.ne("")].copy()
    x["_db"]=num(x.qb_dropback).fillna(0).clip(0,1)
    x["_pa"]=num(x.pass_attempt).fillna(0).eq(1)
    x["_ra"]=num(x.rush_attempt).fillna(0).eq(1)
    x["_scr"]=(x._db.eq(1)|x._pa|x._ra)
    x=x[x._scr].copy()
    x["_play"]=num(x.play_id); x["_score"]=num(x.score_differential); x["_qtr"]=num(x.qtr); x["_drive"]=num(x.drive)
    x["_epa"]=num(x.epa); x["_success"]=num(x.success)
    x["_sack"]=num(x.sack).fillna(0).clip(0,1); x["_scramble"]=num(x.qb_scramble).fillna(0).clip(0,1)
    x["_turnover"]=(num(x.interception).fillna(0).eq(1)|num(x.fumble_lost).fillna(0).eq(1)|num(x.turnover).fillna(0).eq(1)).astype(int)
    x=x.sort_values(["season","week","game_id","team","_play"],kind="mergesort")
    x["_n"]=x.groupby(["season","week","game_id","team"]).cumcount()+1
    rows=[]
    for (s,w,gid,t,o),g in x.groupby(["season","week","game_id","team","opponent"],sort=True):
        def dbr(mask):
            z=g[mask]; return float(z._db.mean()) if len(z) else np.nan
        def mean(mask,col):
            z=num(g.loc[mask,col]).dropna(); return float(z.mean()) if len(z) else np.nan
        f10=g._n.le(10); f15=g._n.le(15); rest=g._n.gt(15)
        drives=[v for v in g._drive.dropna().unique().tolist()]
        f1=g._drive.eq(drives[0]) if drives else f10
        f2=g._drive.isin(drives[:2]) if drives else g._n.le(20)
        neutral=g._score.abs().le(7); trailing8=g._score.le(-8); leading8=g._score.ge(8)
        q1=g._qtr.eq(1); fh=g._qtr.le(2)
        rush=g._ra; pas=g._db.eq(1)
        known=neutral|trailing8|leading8
        den=max(int(known.sum()),1)
        rec={
            "season":int(s),"week":int(w),"game_id":str(gid),"team":canon(t),"opponent":canon(o),
            "actual_drives":float(g._drive.nunique()),"actual_scrimmage_plays":float(len(g)),
            "actual_plays_per_drive":float(len(g)/g._drive.nunique()) if g._drive.nunique() else np.nan,
            "actual_dropbacks":float(g._db.sum()),"actual_pbp_pass_attempts":float(g._pa.sum()),
            "actual_dropback_rate":float(g._db.mean()),
            "actual_attempt_conversion":float(g._pa.sum()/g._db.sum()) if g._db.sum() else np.nan,
            "actual_first10_dbr":dbr(f10),"actual_first15_dbr":dbr(f15),"actual_first_drive_dbr":dbr(f1),
            "actual_first2drives_dbr":dbr(f2),"actual_q1_dbr":dbr(q1),"actual_first_half_dbr":dbr(fh),
            "actual_first15_vs_rest_dbr":dbr(f15)-dbr(rest) if np.isfinite(dbr(f15)) and np.isfinite(dbr(rest)) else np.nan,
            "actual_neutral_share":float(neutral.sum()/den),"actual_trailing8_share":float(trailing8.sum()/den),"actual_leading8_share":float(leading8.sum()/den),
            "actual_neutral_dbr":dbr(neutral),"actual_trailing8_dbr":dbr(trailing8),"actual_leading8_dbr":dbr(leading8),
            "actual_rush_epa":mean(rush,"_epa"),"actual_pass_epa":mean(pas,"_epa"),
            "actual_rush_success":mean(rush,"_success"),"actual_pass_success":mean(pas,"_success"),
            "actual_sack_rate":float(g._sack.sum()/g._db.sum()) if g._db.sum() else np.nan,
            "actual_scramble_rate":float(g._scramble.sum()/g._db.sum()) if g._db.sum() else np.nan,
            "actual_turnovers":float(g._turnover.sum()),
        }
        rows.append(rec)
    return pd.DataFrame(rows)

def shared_keys(part,pbp):
    for keys in (["nflverse_game_id","play_id"],["old_game_id","play_id"],["game_id","play_id"]):
        if all(k in part and k in pbp for k in keys): return list(keys)
    return []

def realized_defense(part,pbp):
    if part.empty: return pd.DataFrame()
    keys=shared_keys(part,pbp)
    if not keys: return pd.DataFrame()
    need=keys+[c for c in ["season","week","posteam","defteam","qb_dropback","pass_attempt","rush_attempt"] if c in pbp]
    r=pbp[need].drop_duplicates(keys)
    x=part.merge(r,on=keys,how="inner",suffixes=("","_pbp"))
    if x.empty: return pd.DataFrame()
    teamcol="posteam" if "posteam" in x else "possession_team" if "possession_team" in x else None
    defcol="defteam" if "defteam" in x else None
    if not teamcol or not defcol: return pd.DataFrame()
    x["team"]=x[teamcol].map(canon); x["opponent"]=x[defcol].map(canon)
    db=num(x.get("qb_dropback",0)).fillna(0).eq(1); pa=num(x.get("pass_attempt",0)).fillna(0).eq(1); ra=num(x.get("rush_attempt",0)).fillna(0).eq(1)
    x=x[db|pa|ra].copy()
    man=x.get("defense_man_zone_type",pd.Series("",index=x.index)).astype(str).str.upper()
    cov=x.get("defense_coverage_type",pd.Series("",index=x.index)).astype(str).str.upper()
    x["_man"]=man.str.contains("MAN",na=False).astype(float); x["_zone"]=man.str.contains("ZONE",na=False).astype(float)
    x["_box"]=num(x.get("defenders_in_box",np.nan)); x["_rushers"]=num(x.get("number_of_pass_rushers",np.nan)); x["_pressure"]=num(x.get("was_pressure",np.nan))
    for n in [0,1,2,3,4,6]: x[f"_cover{n}"]=cov.str.contains(rf"(?:^|\D){n}(?:\D|$)",regex=True,na=False).astype(float)
    rows=[]
    for (s,w,t,o),g in x.groupby(["season","week","team","opponent"],sort=True):
        rec={"season":int(s),"week":int(w),"team":canon(t),"opponent":canon(o),
             "realized_def_man_rate":float(g._man.mean()),"realized_def_zone_rate":float(g._zone.mean()),
             "realized_def_box_mean":float(g._box.mean()) if g._box.notna().any() else np.nan,
             "realized_def_heavy_box_rate":float(g._box.ge(8).mean()) if g._box.notna().any() else np.nan,
             "realized_def_light_box_rate":float(g._box.le(6).mean()) if g._box.notna().any() else np.nan,
             "realized_def_pass_rushers_mean":float(g._rushers.mean()) if g._rushers.notna().any() else np.nan,
             "realized_def_pressure_rate":float(g._pressure.mean()) if g._pressure.notna().any() else np.nan}
        for n in [0,1,2,3,4,6]: rec[f"realized_cover{n}_rate"]=float(g[f"_cover{n}"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)

def choose_baseline(row,*cols):
    for c in cols:
        v=num(pd.Series([row.get(c,np.nan)])).iloc[0]
        if np.isfinite(v): return float(v)
    return np.nan

def classify(row):
    # Error component dominance first.
    att_res=row["attempt_residual_actual_minus_pred"]; ypa_res=row["ypa_residual_actual_minus_pred"]
    vol=row["volume_yard_contribution"]; eff=row["efficiency_yard_contribution"]; inter=row["interaction_yard_contribution"]
    if np.isfinite(eff) and abs(eff) > abs(vol) + 5 and abs(eff) >= abs(inter):
        return "ypa_explosion" if eff>0 else "ypa_collapse"
    # Participation/role mismatch stays first when present in canonical rows.
    share=choose_baseline(row,"actual_attempt_share","actual_qb_attempt_share")
    if np.isfinite(share) and share < .80: return "role_or_participation"
    op=row["opening_deviation_vs_playcaller"]
    if np.isfinite(att_res) and att_res>=4 and np.isfinite(op) and op>=.12: return "planned_pass_heavy_opening"
    if np.isfinite(att_res) and att_res<=-4 and np.isfinite(op) and op<=-.12: return "planned_run_heavy_opening"
    tr=row["trailing_share_surprise"]; ld=row["leading_share_surprise"]
    if np.isfinite(att_res) and att_res>=4 and np.isfinite(tr) and tr>=.15: return "forced_trailing_volume"
    if np.isfinite(att_res) and att_res<=-4 and np.isfinite(ld) and ld>=.15: return "leading_suppression"
    dr=row["drive_residual_actual_minus_pred"]
    if np.isfinite(att_res) and att_res>=4 and np.isfinite(dr) and dr>=2: return "possession_explosion"
    if np.isfinite(att_res) and att_res<=-4 and np.isfinite(dr) and dr<=-2: return "possession_collapse"
    ac=row["attempt_conversion_residual"]
    if np.isfinite(att_res) and att_res<=-4 and np.isfinite(ac) and ac<=-.08: return "dropback_to_attempt_conversion_loss"
    re=row["actual_rush_epa"]
    if np.isfinite(att_res) and att_res<=-4 and np.isfinite(re) and re>=.08: return "run_game_takeover"
    if np.isfinite(att_res) and att_res>=4 and np.isfinite(re) and re<=-.08: return "run_game_failure_pass_pivot"
    return "other_volume_or_mixed"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--m65-game-level",type=Path,required=True)
    ap.add_argument("--m68-features",type=Path,required=True)
    ap.add_argument("--seasons",default="2024,2025")
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args(); seasons=[int(x) for x in a.seasons.split(",") if x.strip()]
    base=lower(pd.read_csv(a.m65_game_level)); new=lower(pd.read_csv(a.m68_features))
    for x in [base,new]:
        x["season"]=num(x.season).astype(int); x["week"]=num(x.week).astype(int); x["team"]=x.team.map(canon)
        if "opponent" in x: x["opponent"]=x.opponent.map(canon)
    base=base[base.season.isin(seasons)].copy()
    keep=[c for c in new.columns if c not in {"opponent"}]
    base=base.merge(new[keep],on=["season","week","team"],how="left",validate="many_to_one")
    pbp,part,manifest=load_sources(seasons); pbg=pbp_team_games(pbp); rdef=realized_defense(part,pbp)
    x=base.merge(pbg,on=["season","week","team","opponent"],how="left",suffixes=("","_pbp"),validate="many_to_one")
    if not rdef.empty: x=x.merge(rdef,on=["season","week","team","opponent"],how="left",validate="many_to_one")
    if x.actual_dropback_rate.isna().any(): raise RuntimeError(f"M69 failed to attach PBP actual script to {int(x.actual_dropback_rate.isna().sum())} target rows")

    x["pred_pass_yards"]=num(x.get("m64_pass_raw_reference",x.get("raw_pass_yards",np.nan)))
    x["actual_pass_yards"]=num(x.get("actual",np.nan)); x["pred_attempts"]=num(x.get("attempts_raw",np.nan)); x["actual_attempts"]=num(x.get("actual_pass_att",np.nan))
    x["pred_ypa"]=num(x.get("ypa_contextual",np.nan)); x["actual_ypa"]=np.where(x.actual_attempts.gt(0),x.actual_pass_yards/x.actual_attempts,np.nan)
    x["pass_residual_actual_minus_pred"]=x.actual_pass_yards-x.pred_pass_yards
    x["attempt_residual_actual_minus_pred"]=x.actual_attempts-x.pred_attempts
    x["ypa_residual_actual_minus_pred"]=x.actual_ypa-x.pred_ypa
    x["volume_yard_contribution"]=(x.actual_attempts-x.pred_attempts)*x.pred_ypa
    x["efficiency_yard_contribution"]=x.pred_attempts*(x.actual_ypa-x.pred_ypa)
    x["interaction_yard_contribution"]=(x.actual_attempts-x.pred_attempts)*(x.actual_ypa-x.pred_ypa)
    x["abs_pass_error"]=x.pass_residual_actual_minus_pred.abs(); x["cat75"]=x.abs_pass_error.ge(75); x["cat100"]=x.abs_pass_error.ge(100)
    x["dbr_residual_actual_minus_pred"]=x.actual_dropback_rate-num(x.get("m64_pred_dropback_rate_neutral",np.nan))
    x["drive_residual_actual_minus_pred"]=x.actual_drives-num(x.get("m64_pred_drives",np.nan))
    x["plays_per_drive_residual"]=x.actual_plays_per_drive-num(x.get("m64_pred_plays_per_drive",np.nan))
    x["attempt_conversion_residual"]=x.actual_attempt_conversion-num(x.get("m64_pred_attempt_conversion",np.nan))
    x["trailing_share_surprise"]=x.actual_trailing8_share-num(x.get("m65_pred_trailing_share",np.nan))
    x["leading_share_surprise"]=x.actual_leading8_share-num(x.get("m65_pred_leading_share",np.nan))
    x["opening_baseline_playcaller"]=x.apply(lambda r: choose_baseline(r,"playcaller_opening_first15_dbr_mean8","opening_first15_dbr_mean8","playcaller_opening_q1_dbr_mean8","opening_q1_dbr_mean8"),axis=1)
    x["opening_deviation_vs_playcaller"]=x.actual_first15_dbr-x.opening_baseline_playcaller
    x["mechanism"]=x.apply(classify,axis=1)
    x["recoverability"] = x.mechanism.map({
        "planned_pass_heavy_opening":"pregame_candidate","planned_run_heavy_opening":"pregame_candidate",
        "run_game_takeover":"partially_in_game","run_game_failure_pass_pivot":"partially_in_game",
        "forced_trailing_volume":"partially_in_game","leading_suppression":"partially_in_game",
        "possession_explosion":"mostly_in_game","possession_collapse":"mostly_in_game",
        "dropback_to_attempt_conversion_loss":"partially_in_game","ypa_explosion":"separate_efficiency_problem","ypa_collapse":"separate_efficiency_problem",
        "role_or_participation":"pregame_candidate","other_volume_or_mixed":"unresolved"}).fillna("unresolved")

    # Mechanism summaries: all, 75+, 100+, and by season.
    sums=[]
    for label,q in [("all",x),("75plus",x[x.cat75]),("100plus",x[x.cat100])]:
        for (season,mech),g in q.groupby(["season","mechanism"],dropna=False):
            sums.append({"slice":label,"season":int(season),"mechanism":mech,"n":len(g),"share":len(g)/len(q[q.season.eq(season)]) if len(q[q.season.eq(season)]) else np.nan,
                         "mean_abs_pass_error":float(g.abs_pass_error.mean()),"mean_attempt_residual":float(g.attempt_residual_actual_minus_pred.mean()),"mean_ypa_residual":float(g.ypa_residual_actual_minus_pred.mean())})
        for mech,g in q.groupby("mechanism",dropna=False):
            sums.append({"slice":label,"season":"combined","mechanism":mech,"n":len(g),"share":len(g)/len(q) if len(q) else np.nan,
                         "mean_abs_pass_error":float(g.abs_pass_error.mean()),"mean_attempt_residual":float(g.attempt_residual_actual_minus_pred.mean()),"mean_ypa_residual":float(g.ypa_residual_actual_minus_pred.mean())})
    summary=pd.DataFrame(sums)

    # Frozen pregame defensive-profile screen against opening deviation. This is
    # descriptive discovery only; no fitted model and no feature promotion.
    scheme_candidates=[
        "opp_coverage_man_rate","opp_coverage_zone_rate","opp_pressure_rate_generated",
        "opp_def_pass_epa","opp_success_rate_def","opponent_force_pass",
        "opp_explosive_play_rate_allowed","market_abs_spread","market_total",
    ]
    screen=[]
    for f in scheme_candidates:
        if f not in x: continue
        c24=safe_corr(x.loc[x.season.eq(2024),f],x.loc[x.season.eq(2024),"opening_deviation_vs_playcaller"])
        c25=safe_corr(x.loc[x.season.eq(2025),f],x.loc[x.season.eq(2025),"opening_deviation_vs_playcaller"])
        cc=safe_corr(x[f],x.opening_deviation_vs_playcaller)
        strong=bool(np.isfinite(c24) and np.isfinite(c25) and np.sign(c24)==np.sign(c25) and abs(c24)>=.10 and abs(c25)>=.10 and abs(cc)>=.15)
        screen.append({"pregame_defense_feature":f,"target":"opening_deviation_vs_playcaller","corr_2024":c24,"corr_2025":c25,"corr_combined":cc,"strong_replicated_descriptive":strong})
    scheme_screen=pd.DataFrame(screen)

    # Realized defense vs pregame scheme stability. Names are intentionally explicit.
    stability=[]
    pairs=[("opp_coverage_man_rate","realized_def_man_rate"),("opp_coverage_zone_rate","realized_def_zone_rate"),("opp_pressure_rate_generated","realized_def_pressure_rate")]
    for pre,act in pairs:
        if pre not in x or act not in x: continue
        for sl,q in [("2024",x[x.season.eq(2024)]),("2025",x[x.season.eq(2025)]),("combined",x)]:
            z=pd.DataFrame({"p":num(q[pre]),"a":num(q[act])}).dropna()
            stability.append({"season":sl,"pregame_feature":pre,"realized_feature":act,"n":len(z),"corr":safe_corr(z.p,z.a),"mae":float((z.p-z.a).abs().mean()) if len(z) else np.nan,"bias":float((z.p-z.a).mean()) if len(z) else np.nan})
    stability=pd.DataFrame(stability)

    # Playcaller/opening planned-deviation cohorts against actual realized scheme.
    x["opening_regime"]=pd.cut(x.opening_deviation_vs_playcaller,[-np.inf,-.12,.12,np.inf],labels=["run_heavier_than_caller","near_caller_baseline","pass_heavier_than_caller"])
    realized_cols=[c for c in x if c.startswith("realized_def_") or c.startswith("realized_cover")]
    cohort=[]
    for (season,reg),g in x.groupby(["season","opening_regime"],observed=True):
        rec={"season":int(season),"opening_regime":str(reg),"n":len(g),"mean_opening_deviation":float(g.opening_deviation_vs_playcaller.mean())}
        for c in realized_cols: rec[c]=float(num(g[c]).mean()) if num(g[c]).notna().any() else np.nan
        cohort.append(rec)
    cohort=pd.DataFrame(cohort)

    # Recovery ceiling diagnostic: how much catastrophic error belongs to each broad class.
    rec=[]
    for sl,q in [("75plus",x[x.cat75]),("100plus",x[x.cat100])]:
        total=float(q.abs_pass_error.sum())
        for r,g in q.groupby("recoverability"):
            rec.append({"slice":sl,"recoverability":r,"n":len(g),"game_share":len(g)/len(q) if len(q) else np.nan,"error_share":float(g.abs_pass_error.sum()/total) if total else np.nan,"mean_abs_error":float(g.abs_pass_error.mean())})
    recovery=pd.DataFrame(rec)

    # Frozen interpretation is descriptive, not a production gate.
    strong_scheme=int(scheme_screen.strong_replicated_descriptive.sum()) if len(scheme_screen) else 0
    planned100=int(x[x.cat100 & x.mechanism.isin(["planned_pass_heavy_opening","planned_run_heavy_opening"])].shape[0])
    interp="m69_matchup_conditioning_supported_for_m70_hypothesis" if strong_scheme>0 else "m69_opening_signal_not_explained_by_current_pregame_defense_profiles"
    interpretation=pd.DataFrame([{"target_games":len(x),"cat75_games":int(x.cat75.sum()),"cat100_games":int(x.cat100.sum()),"planned_opening_cat100_games":planned100,"strong_replicated_pregame_scheme_pairs":strong_scheme,"m69_interpretation":interp,"production_actionable":False}])

    a.out_dir.mkdir(parents=True,exist_ok=True)
    x.to_csv(a.out_dir/"m69_game_forensic_atlas.csv",index=False)
    summary.to_csv(a.out_dir/"m69_mechanism_summary.csv",index=False)
    scheme_screen.to_csv(a.out_dir/"m69_pregame_defense_opening_deviation_screen.csv",index=False)
    stability.to_csv(a.out_dir/"m69_defensive_scheme_stability.csv",index=False)
    cohort.to_csv(a.out_dir/"m69_realized_defense_by_opening_regime.csv",index=False)
    recovery.to_csv(a.out_dir/"m69_recoverability_summary.csv",index=False)
    interpretation.to_csv(a.out_dir/"m69_precommitted_interpretation.csv",index=False)
    manifest.to_csv(a.out_dir/"m69_source_manifest.csv",index=False)
    print("=== M69 INTERPRETATION ==="); print(interpretation.to_string(index=False))
    print("=== M69 100+ MECHANISMS ==="); print(summary[(summary.slice.eq("100plus")) & summary.season.astype(str).eq("combined")].sort_values("n",ascending=False).to_string(index=False))
    print("=== M69 RECOVERABILITY ==="); print(recovery.to_string(index=False))
    print("=== M69 PREGAME SCHEME -> OPENING DEVIATION ==="); print(scheme_screen.to_string(index=False))
    print("=== M69 SCHEME STABILITY ==="); print(stability.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
