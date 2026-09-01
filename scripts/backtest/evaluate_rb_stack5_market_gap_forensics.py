#!/usr/bin/env python3
"""RB-STACK5 no-fit forensic audit of the remaining 899-game market gap."""
from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd
TEAM={"OAK":"LV","SD":"LAC","STL":"LA","LAR":"LA","JAX":"JAC","ARZ":"ARI","WSH":"WAS"}
def tm(v):
 s=str(v).strip().upper() if not pd.isna(v) else "";return TEAM.get(s,s) if s not in {"","NAN","NONE","<NA>"} else ""
def nk(v):return re.sub(r"[^a-z0-9]","",str(v or "").lower())
def lower(x):x=x.copy();x.columns=[str(c).strip().lower() for c in x.columns];return x
def one(root,name):
 h=list(root.rglob(name));
 if len(h)!=1:raise RuntimeError(f"expected one {name} under {root}; found {len(h)}")
 return lower(pd.read_csv(h[0],low_memory=False))
def num(s):return pd.to_numeric(s,errors="coerce")
def prep(x):
 z=x.copy();z["week"]=num(z.week).astype(int);z["team"]=z.team.map(tm)
 if "join_key" in z.columns:z["jk"]=z.join_key.astype(str).map(nk)
 elif "player_clean_key" in z.columns:z["jk"]=z.player_clean_key.astype(str).map(nk)
 elif "name_key" in z.columns:z["jk"]=z.name_key.astype(str).map(nk)
 else:z["jk"]=z.player.map(nk)
 return z
def summary(g,name):
 if g.empty:return {"stratum":name,"n":0}
 model_err=np.abs(num(g.parent_yards)-num(g.actual_rush_yards));vegas_err=np.abs(num(g.consensus_line)-num(g.actual_rush_yards))
 return {"stratum":name,"n":len(g),"model_mae":float(model_err.mean()),"vegas_mae":float(vegas_err.mean()),"model_minus_vegas_mae":float(model_err.mean()-vegas_err.mean()),"model_closer_rate":float((model_err<vegas_err).mean()),"mean_edge":float(num(g.edge).mean()),"actual_yards_mean":float(num(g.actual_rush_yards).mean()),"model_yards_mean":float(num(g.parent_yards).mean()),"vegas_yards_mean":float(num(g.consensus_line).mean()),"actual_carries_mean":float(num(g.actual_rush_att).mean()),"pred_carries_mean":float(num(g.parent_att).mean()),"carry_error_mean":float(num(g.carry_error).mean()),"carry_abs_error_mean":float(num(g.carry_error).abs().mean()),"actual_ypc_mean":float(num(g.actual_ypc).mean()),"pred_ypc_mean":float(num(g.parent_ypc).mean()),"ypc_error_mean":float(num(g.ypc_error).mean()),"ypc_abs_error_mean":float(num(g.ypc_error).abs().mean()),"opportunity_oracle_recovery":float(num(g.opp_recovery).mean()),"efficiency_oracle_recovery":float(num(g.eff_recovery).mean()),"opp_dominant_rate":float(num(g.opp_recovery).gt(num(g.eff_recovery)).mean()),"week1_rate":float(num(g.week).eq(1).mean()),"m95f_risk_rate":float(num(g.get("state_m95f_risk",0)).fillna(0).eq(1).mean()),"injury_report_rate":float(num(g.get("injury_reported",0)).fillna(0).gt(0).mean()),"rookie_rate":float(num(g.get("rookie_flag",0)).fillna(0).gt(0).mean()),"committee_rate":float(num(g.get("credible_competitors",0)).fillna(0).ge(1).mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stack4-root",type=Path,required=True);ap.add_argument("--market-root",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
 x=prep(one(a.stack4_root,"stack4_2025_casebook.csv"));cb=prep(one(a.market_root,"rb_market_casebook.csv"));keys=["week","team","jk"]
 q=x.merge(cb[keys+["consensus_line"]].drop_duplicates(keys),on=keys,how="inner",validate="one_to_one")
 if len(q)!=899:raise RuntimeError(f"expected exact 899 market rows; found {len(q)}")
 w=num(q.week);q["parent_yards"]=num(q.p3_parent);q["parent_att"]=np.where(w.eq(1),num(q.stack_att),num(q.enriched_att));q["parent_ypc"]=np.where(num(q.parent_att).abs().gt(.20),num(q.parent_yards)/num(q.parent_att),np.nan)
 q["actual_rush_att"]=num(q.actual_rush_att);q["actual_rush_yards"]=num(q.actual_rush_yards);q["actual_ypc"]=np.where(q.actual_rush_att.gt(0),q.actual_rush_yards/q.actual_rush_att,np.nan)
 q["edge"]=q.parent_yards-num(q.consensus_line);q["abs_edge"]=q.edge.abs();q["model_abs_err"]=(q.parent_yards-q.actual_rush_yards).abs();q["vegas_abs_err"]=(num(q.consensus_line)-q.actual_rush_yards).abs();q["vegas_advantage"]=q.model_abs_err-q.vegas_abs_err;q["model_closer"]=(q.model_abs_err<q.vegas_abs_err).astype(int)
 q["carry_error"]=q.parent_att-q.actual_rush_att;q["ypc_error"]=q.parent_ypc-q.actual_ypc
 q["opp_oracle_yards"]=q.actual_rush_att*q.parent_ypc;q["eff_oracle_yards"]=q.parent_att*q.actual_ypc
 q["opp_recovery"]=q.model_abs_err-(q.opp_oracle_yards-q.actual_rush_yards).abs();q["eff_recovery"]=q.model_abs_err-(q.eff_oracle_yards-q.actual_rush_yards).abs()
 q["mechanism"]=np.where(q.opp_recovery>q.eff_recovery,"opportunity","efficiency")
 strata={"all_899":pd.Series(True,index=q.index),"within_5":q.abs_edge.lt(5),"model_above_5_10":q.edge.between(5,10,inclusive="left"),"model_below_5_10":q.edge.between(-10,-5,inclusive="right"),"model_above_10plus":q.edge.ge(10),"model_below_10plus":q.edge.le(-10),"week1":w.eq(1),"weeks2_18":w.ge(2),"m95f_risk":num(q.get("state_m95f_risk",0)).fillna(0).eq(1),"m95f_nonrisk":num(q.get("state_m95f_risk",0)).fillna(0).eq(0),"injury_reported":num(q.get("injury_reported",0)).fillna(0).gt(0),"rookie":num(q.get("rookie_flag",0)).fillna(0).gt(0),"committee":num(q.get("credible_competitors",0)).fillna(0).ge(1),"concentrated":num(q.get("prior_backfield_hhi",0)).fillna(0).ge(.65),"depth_rank1":num(q.get("depth_rank",np.nan)).eq(1),"depth_rank2":num(q.get("depth_rank",np.nan)).eq(2),"depth_rank3plus":num(q.get("depth_rank",np.nan)).ge(3)}
 out=pd.DataFrame([summary(q.loc[m],n) for n,m in strata.items()])
 # signed disagreement x mechanism table
 mech=[]
 for n,m in {k:v for k,v in strata.items() if k in ["within_5","model_above_5_10","model_below_5_10","model_above_10plus","model_below_10plus"]}.items():
  g=q.loc[m]
  for typ in ["opportunity","efficiency"]:mech.append({"stratum":n,"mechanism":typ,"n":int(g.mechanism.eq(typ).sum()),"rate":float(g.mechanism.eq(typ).mean()) if len(g) else np.nan,"mean_vegas_advantage":float(num(g.loc[g.mechanism.eq(typ),"vegas_advantage"]).mean()) if g.mechanism.eq(typ).any() else np.nan})
 mech=pd.DataFrame(mech)
 keep=[c for c in ["week","team","player","position","parent_yards","consensus_line","actual_rush_yards","edge","model_abs_err","vegas_abs_err","vegas_advantage","parent_att","actual_rush_att","carry_error","parent_ypc","actual_ypc","ypc_error","opp_recovery","eff_recovery","mechanism","depth_rank","depth_slot","prior1_snap_pct","prior3_snap_pct","prior3_rb_share","credible_competitors","prior_backfield_hhi","injury_reported","injury_out_doubtful","injury_questionable","rookie_flag","state_m95f_risk"] if c in q.columns]
 false_high=q.loc[q.edge.ge(10),keep].sort_values("vegas_advantage",ascending=False);false_low=q.loc[q.edge.le(-10),keep].sort_values("vegas_advantage",ascending=False);vw=q[keep].assign(vegas_advantage=q.vegas_advantage).sort_values("vegas_advantage",ascending=False).head(75);mw=q[keep].assign(vegas_advantage=q.vegas_advantage).sort_values("vegas_advantage",ascending=True).head(75)
 # Evidence-backed missing-information ledger from current data only.
 above=q.loc[q.edge.ge(10)];below=q.loc[q.edge.le(-10)]
 ledger=pd.DataFrame([
  {"candidate":"current_role_and_backfield_allocation","evidence":"opportunity-oracle recovery on 10+ disagreements","model_above_10_opp_recovery":float(num(above.opp_recovery).mean()),"model_below_10_opp_recovery":float(num(below.opp_recovery).mean()),"status":"INVESTIGATE_IF_LARGE"},
  {"candidate":"runner_efficiency_matchup","evidence":"efficiency-oracle recovery on 10+ disagreements","model_above_10_eff_recovery":float(num(above.eff_recovery).mean()),"model_below_10_eff_recovery":float(num(below.eff_recovery).mean()),"status":"INVESTIGATE_IF_LARGE"},
  {"candidate":"week1_role_initialization","evidence":"Week1 rate among 10+ disagreements vs all","model_above_10_rate":float(w.loc[above.index].eq(1).mean()),"model_below_10_rate":float(w.loc[below.index].eq(1).mean()),"all_rate":float(w.eq(1).mean()),"status":"AUDIT"},
  {"candidate":"committee_substitution_role","evidence":"committee rate among 10+ disagreements","model_above_10_rate":float(num(above.get("credible_competitors",0)).fillna(0).ge(1).mean()),"model_below_10_rate":float(num(below.get("credible_competitors",0)).fillna(0).ge(1).mean()),"all_rate":float(num(q.get("credible_competitors",0)).fillna(0).ge(1).mean()),"status":"AUDIT"},
  {"candidate":"injury_availability_precision","evidence":"injury-report rate among 10+ disagreements","model_above_10_rate":float(num(above.get("injury_reported",0)).fillna(0).gt(0).mean()),"model_below_10_rate":float(num(below.get("injury_reported",0)).fillna(0).gt(0).mean()),"all_rate":float(num(q.get("injury_reported",0)).fillna(0).gt(0).mean()),"status":"AUDIT"},
 ])
 for n,d in [("stack5_stratum_summary.csv",out),("stack5_mechanism_summary.csv",mech),("stack5_false_high_casebook.csv",false_high),("stack5_false_low_casebook.csv",false_low),("stack5_vegas_biggest_wins.csv",vw),("stack5_model_biggest_wins.csv",mw),("stack5_missing_information_ledger.csv",ledger),("stack5_899_casebook.csv",q)]:d.to_csv(a.out_dir/n,index=False)
 print("=== strata ===");print(out.to_string(index=False));print("=== mechanisms ===");print(mech.to_string(index=False));print("=== ledger ===");print(ledger.to_string(index=False));print("=== false-high top20 ===");print(false_high.head(20).to_string(index=False));print("=== false-low top20 ===");print(false_low.head(20).to_string(index=False))
if __name__=="__main__":main()
