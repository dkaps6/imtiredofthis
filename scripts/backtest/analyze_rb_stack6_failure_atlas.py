#!/usr/bin/env python3
"""RB STACK6 failure atlas after the frozen primary role arm failed.

Diagnostic only. No model fitting, no threshold/weight selection, no production
change. 2025 is exposed development evidence. Sportsbook is downstream only.
"""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd
TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
def lower(x): x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def num(s): return pd.to_numeric(s,errors="coerce")
def tm(v):
    s=str(v).strip().upper() if not pd.isna(v) else ""; return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v): return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def one(root,name):
    h=list(root.rglob(name));
    if len(h)!=1: raise RuntimeError(f"expected one {name}, found {len(h)}")
    return lower(pd.read_csv(h[0],low_memory=False))
def mae(a,p):
    q=pd.DataFrame({"a":num(a),"p":num(p)}).dropna(); return float((q.p-q.a).abs().mean()) if len(q) else np.nan

def prep(x):
    z=x.copy();z["season"]=num(z.season).astype(int);z["week"]=num(z.week).astype(int);z["team"]=z.team.map(tm)
    z["join_key"]=z.get("join_key",z.get("player_clean_key",z.player)).astype(str).map(nk)
    actual_den=z.groupby(["season","week","team"])["actual_carries"].transform("sum")
    z["actual_share"]=np.where(num(actual_den).gt(0),num(z.actual_carries)/num(actual_den),0.0)
    z["needed_share_delta"]=num(z.actual_share)-num(z.baseline_share)
    z["role_share_delta"]=num(z.role_share_normalized)-num(z.baseline_share)
    z["primary_share_delta"]=num(z.primary_share)-num(z.baseline_share)
    z["parent_carry_abs_err"]=(num(z.parent_att)-num(z.actual_carries)).abs()
    z["primary_carry_abs_err"]=(num(z.primary_att)-num(z.actual_carries)).abs()
    z["carry_improvement"]=z.parent_carry_abs_err-z.primary_carry_abs_err
    z["parent_yard_abs_err"]=(num(z.parent_yards)-num(z.actual_rush_yards)).abs()
    z["primary_yard_abs_err"]=(num(z.primary_yards)-num(z.actual_rush_yards)).abs()
    z["yard_improvement"]=z.parent_yard_abs_err-z.primary_yard_abs_err
    z["role_direction_correct"]=(np.sign(z.role_share_delta)==np.sign(z.needed_share_delta)).astype(float)
    z.loc[z.needed_share_delta.abs().lt(.02),"role_direction_correct"]=np.nan
    z["snap_trend_1v3"]=num(z.get("prior1_snap_pct",pd.Series(np.nan,index=z.index)))-num(z.get("prior3_snap_pct",pd.Series(np.nan,index=z.index)))
    z["share_trend_1v3"]=num(z.get("prior1_rb_share",pd.Series(np.nan,index=z.index)))-num(z.get("prior3_rb_share",pd.Series(np.nan,index=z.index)))
    return z

def summarize(z,label,mask):
    q=z.loc[mask].copy()
    if q.empty:return None
    corr=q[["role_share_delta","needed_share_delta"]].corr().iloc[0,1] if len(q)>=3 else np.nan
    return {"stratum":label,"n":len(q),"parent_carry_mae":mae(q.actual_carries,q.parent_att),"primary_carry_mae":mae(q.actual_carries,q.primary_att),"carry_mae_gain":float(q.carry_improvement.mean()),"parent_yard_mae":mae(q.actual_rush_yards,q.parent_yards),"primary_yard_mae":mae(q.actual_rush_yards,q.primary_yards),"yard_mae_gain":float(q.yard_improvement.mean()),"role_needed_delta_corr":float(corr) if pd.notna(corr) else np.nan,"role_direction_correct":float(q.role_direction_correct.mean()),"mean_role_share_delta":float(q.role_share_delta.mean()),"mean_needed_share_delta":float(q.needed_share_delta.mean()),"mean_abs_role_delta":float(q.role_share_delta.abs().mean()),"mean_abs_needed_delta":float(q.needed_share_delta.abs().mean())}
def atlas(z):
    sec=z.secondary_defined.eq(1)&z.state_m95f_risk.eq(0)&z.week.ge(2)
    avail=num(z.role_raw_pred).notna()
    strata={
        "secondary_nonrisk_all":sec,
        "secondary_nonrisk_role_available":sec&avail,
        "role_says_lower":sec&avail&z.role_share_delta.lt(0),
        "role_says_raise":sec&avail&z.role_share_delta.gt(0),
        "role_abs_delta_ge_03":sec&avail&z.role_share_delta.abs().ge(.03),
        "role_abs_delta_ge_05":sec&avail&z.role_share_delta.abs().ge(.05),
        "role_abs_delta_ge_10":sec&avail&z.role_share_delta.abs().ge(.10),
        "snap_trend_down_10pt":sec&z.snap_trend_1v3.le(-.10),
        "snap_trend_up_10pt":sec&z.snap_trend_1v3.ge(.10),
        "share_trend_down_10pt":sec&z.share_trend_1v3.le(-.10),
        "share_trend_up_10pt":sec&z.share_trend_1v3.ge(.10),
    }
    optional={
        "injury_reported":"injury_reported","player_out_doubtful":"injury_out_doubtful","player_questionable":"injury_questionable",
        "practice_dnp":"practice_dnp","practice_limited":"practice_limited","rookie":"rookie_flag",
    }
    for label,col in optional.items():
        if col in z.columns:strata[label]=sec&num(z[col]).fillna(0).gt(0)
    if "credible_competitors" in z.columns:
        strata["two_plus_credible_competitors"]=sec&num(z.credible_competitors).ge(2)
        strata["one_credible_competitor"]=sec&num(z.credible_competitors).eq(1)
    if "prior_backfield_hhi" in z.columns:
        strata["low_concentration_hhi_lt55"]=sec&num(z.prior_backfield_hhi).lt(.55)
        strata["high_concentration_hhi_ge70"]=sec&num(z.prior_backfield_hhi).ge(.70)
    if "injured_comp_count" in z.columns:
        strata["injured_competitor_present"]=sec&num(z.injured_comp_count).fillna(0).gt(0)
    if "injured_comp_prior_share" in z.columns:
        strata["meaningful_injured_comp_share_ge20"]=sec&num(z.injured_comp_prior_share).fillna(0).ge(.20)
    rows=[summarize(z,k,v) for k,v in strata.items()];return pd.DataFrame([r for r in rows if r])
def top_cases(z):
    sec=z.secondary_defined.eq(1)&z.state_m95f_risk.eq(0)&z.week.ge(2)&num(z.role_raw_pred).notna()
    keep=[c for c in ["season","week","team","player","depth_rank","actual_carries","parent_att","primary_att","actual_rush_yards","parent_yards","primary_yards","baseline_share","role_share_normalized","actual_share","needed_share_delta","role_share_delta","carry_improvement","yard_improvement","prior1_snap_pct","prior3_snap_pct","prior1_rb_share","prior3_rb_share","credible_competitors","prior_backfield_hhi","injury_reported","injury_out_doubtful","injury_questionable","practice_dnp","practice_limited","rookie_flag"] if c in z.columns]
    q=z.loc[sec,keep].copy();return q.nlargest(40,"yard_improvement"),q.nsmallest(40,"yard_improvement")
def market_atlas(z,cb):
    c=cb.copy();c["week"]=num(c.week).astype(int);c["team"]=c.team.map(tm);c["join_key"]=c.player.astype(str).map(nk)
    q=z.merge(c[["week","team","join_key","consensus_line"]].drop_duplicates(["week","team","join_key"]),on=["week","team","join_key"],how="inner",validate="one_to_one")
    q["parent_market_edge"]=num(q.parent_yards)-num(q.consensus_line);q["primary_market_edge"]=num(q.primary_yards)-num(q.consensus_line)
    q["vegas_adv_parent"]=(num(q.parent_yards)-num(q.actual_rush_yards)).abs()-(num(q.consensus_line)-num(q.actual_rush_yards)).abs()
    sec=q.secondary_defined.eq(1)&q.state_m95f_risk.eq(0)&q.week.ge(2)
    rows=[]
    for label,mask in {"all_market":pd.Series(True,index=q.index),"secondary_nonrisk_market":sec,"secondary_nonrisk_parent_10plus_above":sec&q.parent_market_edge.ge(10),"secondary_nonrisk_parent_10plus_below":sec&q.parent_market_edge.le(-10),"secondary_role_says_lower":sec&q.role_share_delta.lt(0),"secondary_role_says_raise":sec&q.role_share_delta.gt(0)}.items():
        g=q.loc[mask]
        if g.empty:continue
        rows.append({"stratum":label,"n":len(g),"parent_mae":mae(g.actual_rush_yards,g.parent_yards),"primary_mae":mae(g.actual_rush_yards,g.primary_yards),"vegas_mae":mae(g.actual_rush_yards,g.consensus_line),"parent_minus_vegas":mae(g.actual_rush_yards,g.parent_yards)-mae(g.actual_rush_yards,g.consensus_line),"primary_minus_vegas":mae(g.actual_rush_yards,g.primary_yards)-mae(g.actual_rush_yards,g.consensus_line),"mean_parent_edge":float(g.parent_market_edge.mean()),"mean_role_delta":float(g.role_share_delta.mean())})
    return pd.DataFrame(rows)
def conclusion(a):
    # This is descriptive routing of the next research question, not a point-model selector.
    q=a.set_index("stratum") if len(a) else pd.DataFrame()
    sec=q.loc["secondary_nonrisk_role_available"] if "secondary_nonrisk_role_available" in q.index else pd.Series(dtype=float)
    lower=q.loc["role_says_lower"] if "role_says_lower" in q.index else pd.Series(dtype=float)
    raise_=q.loc["role_says_raise"] if "role_says_raise" in q.index else pd.Series(dtype=float)
    return pd.DataFrame([{"disposition":"STACK6_FAILURE_ATLAS_COMPLETE_NO_RETUNE","secondary_role_delta_needed_corr":sec.get("role_needed_delta_corr",np.nan),"secondary_role_direction_correct":sec.get("role_direction_correct",np.nan),"lower_signal_yard_gain":lower.get("yard_mae_gain",np.nan),"raise_signal_yard_gain":raise_.get("yard_mae_gain",np.nan),"next":"AUDIT_CURRENT_WEEK_AVAILABILITY_AND_ROLE_TRANSITION_STATE_IF_ATLAS_SUPPORTS; OTHERWISE NEW NONREDUNDANT_OPPORTUNITY_MECHANISM","model_fit":0,"threshold_search":0,"weight_search":0,"sportsbook_upstream":0}])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stack6-root",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    z=prep(one(a.stack6_root,"stack6_2025_casebook.csv")); at=atlas(z);good,bad=top_cases(z);ma=market_atlas(z,one(a.market_root,"rb_market_casebook.csv"));co=conclusion(at)
    at.to_csv(a.out_dir/"stack6_failure_atlas.csv",index=False);good.to_csv(a.out_dir/"stack6_role_best_cases.csv",index=False);bad.to_csv(a.out_dir/"stack6_role_worst_cases.csv",index=False);ma.to_csv(a.out_dir/"stack6_market_failure_atlas_DOWNSTREAM.csv",index=False);co.to_csv(a.out_dir/"stack6_failure_atlas_disposition.csv",index=False)
    print("=== FAILURE ATLAS ===");print(at.to_string(index=False));print("=== MARKET DOWNSTREAM ===");print(ma.to_string(index=False));print("=== CONCLUSION ===");print(co.to_string(index=False))
if __name__=="__main__":main()
