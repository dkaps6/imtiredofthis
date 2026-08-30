#!/usr/bin/env python3
"""Migration 79: one predictive test of corrected official game-day inactives.

2024 is development/training. 2025 is untouched evaluation. The frozen
canonical-v3 football-only baseline is corrected only through official inactive
identity/position plus strictly-prior snap-role/pass-rush quality. No market,
generic injury feed, depth-chart-discontinuity remix, or target-game outcomes
are features. One model family only; no post-result retuning.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.backtest import audit_qb_40s_information_frontier as m76
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import test_qb_exact_personnel_discontinuity as m77

CANON_SHA="c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
INACTIVE_SHA="d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
RIDGE_ALPHA=10.0; DEV_SPLIT_WEEK=9; BOOT_N=2000; BOOT_SEED=79
PROMOTE_MAE=1.50; PROMOTE_CORR=.02; MIN_ID=.90; MIN_PASS_PROB=.80
MIN_COMP_PROB=.80; MIN_ATT_GAIN=.10; MIN_YPA_GAIN=.03
FEATURES=[
 "off_ol_count","off_ol_role","off_ol_high","off_skill_count","off_skill_role","off_skill_high",
 "def_db_count","def_db_role","def_db_high","def_rush_count","def_rush_role","def_rush_high",
 "def_rush_pressures","def_rush_sacks"]
OL={"OL","OT","T","LT","RT","OG","G","LG","RG","C"}
SKILL={"RB","FB","HB","WR","TE"}; DB={"CB","DB","S","FS","SS"}
RUSH={"DL","DT","NT","DE","EDGE","OLB","LB","ILB","MLB"}
MARKET=("market","spread","sportsbook","moneyline","vegas","game_total","team_total")

def num(x): return pd.to_numeric(x,errors="coerce")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mae(a,p): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(p,float))))
def corr(a,p):
 a=np.asarray(a,float); p=np.asarray(p,float)
 return float(np.corrcoef(a,p)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(p)>0 else np.nan

def load_inputs(cp,ip):
 if sha(cp)!=CANON_SHA: raise RuntimeError("canonical SHA drift")
 if sha(ip)!=INACTIVE_SHA: raise RuntimeError("corrected M78 inactive SHA drift")
 b=pd.read_csv(cp,low_memory=False); b.columns=[str(c).lower() for c in b.columns]
 if len(b)!=884 or {int(k):int(v) for k,v in num(b.season).value_counts().to_dict().items()}!={2024:444,2025:440}: raise RuntimeError("canonical row drift")
 if any(any(t in c for t in MARKET) for c in b.columns): raise RuntimeError("market boundary violated")
 b["season"]=num(b.season).astype(int); b["week"]=num(b.week).astype(int)
 b["team"]=b.team.map(m76.team_value); b["opponent"]=b.opponent.map(m76.team_value)
 b["base_ypa"]=num(b.pred_pass_yards)/num(b.pred_attempts).replace(0,np.nan)
 b["actual_ypa_calc"]=num(b.actual_pass_yards)/num(b.actual_attempts).replace(0,np.nan)
 i=pd.read_csv(ip,low_memory=False); i.columns=[str(c).lower() for c in i.columns]
 if len(i)!=1088 or {int(k):int(v) for k,v in num(i.season).value_counts().to_dict().items()}!={2024:544,2025:544}: raise RuntimeError("inactive row drift")
 if i.duplicated(["season","week","team"]).any() or i.inactive_tokens.isna().any(): raise RuntimeError("inactive contract drift")
 i["season"]=num(i.season).astype(int); i["week"]=num(i.week).astype(int); i["team"]=i.team.map(m76.team_value)
 return b,i

def roster_map(df):
 if df.empty:return {}
 tc=m76.first_col(df,["team","club_code"]); wc=m76.first_col(df,["week"]); ic=m76.first_col(df,["gsis_id"])
 ncs=[c for c in ["full_name","player_name","display_name","football_name"] if c in df.columns]
 if not tc or not wc or not ic or not ncs:return {}
 out={}
 for r in df.to_dict("records"):
  try:w=int(float(r[wc]))
  except:continue
  team=m76.team_value(r[tc]); pid=str(r.get(ic,"")).strip()
  if not pid or pid.lower() in {"nan","none"}:continue
  for nc in ncs:
   k=m78.norm_name(r.get(nc,""))
   if k:out.setdefault((w,team,k),pid)
 return out

def parse_tokens(s):
 out=[]
 for tok in str(s).split("|"):
  p=tok.rsplit(":",2)
  if len(p)==3:out.append((p[0],p[1].upper()))
 return out

def ingroup(pos,grp): return bool(set(str(pos).split("/"))&grp)

def prior_sources(out):
 asof=datetime.now(timezone.utc).isoformat(); meta=[]
 rms={s:roster_map(m78.load_weekly_rosters(s,meta)) for s in [2024,2025]}
 snaps=[m76.download_table("pfr_snap_counts",s,m76.release_urls("snaps",s),meta,asof) for s in [2023,2024,2025]]
 pfrs=[m76.download_table("pfr_individual_pass_rush",s,m76.release_urls("pfr",s),meta,asof) for s in [2023,2024,2025]]
 sr=pd.concat([x for x in snaps if not x.empty],ignore_index=True) if any(not x.empty for x in snaps) else pd.DataFrame()
 pr=pd.concat([x for x in pfrs if not x.empty],ignore_index=True) if any(not x.empty for x in pfrs) else pd.DataFrame()
 sr,pr,bridge=m76.build_id_bridge(sr,pr,meta,asof)
 sh=m77.prepare_snap_history(sr); ph=m77.prepare_pfr_history(pr)
 pd.DataFrame(meta).to_csv(out/"m79_source_snapshot_hashes.csv",index=False)
 (out/"m79_source_contract.json").write_text(json.dumps({"as_of_utc":asof,"canonical_sha256":CANON_SHA,"inactive_sha256":INACTIVE_SHA,"snap_rows":len(sh),"pfr_rows":len(ph),"bridge":bridge,"sportsbook_used":False,"generic_injury_feed_used":False,"depth_chart_discontinuity_features_used":False,"target_game_outcome_features_used":False},indent=2)+"\n")
 return rms,sh,ph

def grp(players,allowed,side,s,w,team,rm,si,pi=None):
 sel=[x for x in players if ingroup(x[1],allowed)]; roles=[]; mapped=0; press=0.; sacks=0.
 for name,_ in sel:
  pid=rm.get((w,team,name)); role=0.
  if pid:
   mapped+=1; role=m77.prior_role(si,pid,s,w,side)
   if pi is not None:
    press+=m77.prior_pfr(pi,pid,s,w,"pressures"); sacks+=m77.prior_pfr(pi,pid,s,w,"sacks")
  roles.append(role)
 return len(sel),float(np.sum(roles)),float(sum(x>=.5 for x in roles)),press,sacks,mapped

def make_features(base,inact,rms,sh,ph):
 si=m77.history_index(sh); pi=m77.history_index(ph)
 ix={(int(r.season),int(r.week),str(r.team)):parse_tokens(r.inactive_tokens) for r in inact.itertuples(index=False)}
 rows=[]; mapped=total=0
 for r in base.itertuples(index=False):
  s,w,t,o=int(r.season),int(r.week),str(r.team),str(r.opponent); rm=rms.get(s,{})
  own=ix[(s,w,t)]; opp=ix[(s,w,o)]
  a=grp(own,OL,"off",s,w,t,rm,si); b=grp(own,SKILL,"off",s,w,t,rm,si)
  c=grp(opp,DB,"def",s,w,o,rm,si); d=grp(opp,RUSH,"def",s,w,o,rm,si,pi)
  total+=a[0]+b[0]+c[0]+d[0]; mapped+=a[5]+b[5]+c[5]+d[5]
  rows.append({"season":s,"week":w,"team":t,"opponent":o,
   "off_ol_count":a[0],"off_ol_role":a[1],"off_ol_high":a[2],"off_skill_count":b[0],"off_skill_role":b[1],"off_skill_high":b[2],
   "def_db_count":c[0],"def_db_role":c[1],"def_db_high":c[2],"def_rush_count":d[0],"def_rush_role":d[1],"def_rush_high":d[2],"def_rush_pressures":d[3],"def_rush_sacks":d[4]})
 f=pd.DataFrame(rows); f.attrs["id_rate"]=mapped/max(total,1); return f

def fit(train,test,target):
 sc=StandardScaler(); x=sc.fit_transform(train[FEATURES].astype(float)); z=sc.transform(test[FEATURES].astype(float))
 m=Ridge(alpha=RIDGE_ALPHA,fit_intercept=False).fit(x,train[target].astype(float))
 return m.predict(z),pd.DataFrame({"feature":FEATURES,"coefficient":m.coef_,"target":target})

def evaluate(df,ac,yc):
 ay=df.actual_pass_yards.to_numpy(float); ba=df.pred_pass_yards.to_numpy(float)
 aa=df.actual_attempts.to_numpy(float); pa=df.pred_attempts.to_numpy(float); by=df.base_ypa.to_numpy(float); ya=df.actual_ypa_calc.to_numpy(float)
 na=np.maximum(1.,pa+ac); ny=np.maximum(.1,by+yc); py=na*ny
 return {"n":len(df),"baseline_mae":mae(ay,ba),"corrected_mae":mae(ay,py),"mae_gain":mae(ay,ba)-mae(ay,py),"baseline_corr":corr(ay,ba),"corrected_corr":corr(ay,py),"corr_gain":corr(ay,py)-corr(ay,ba),"baseline_tails_100":int(np.sum(np.abs(ba-ay)>=100)),"corrected_tails_100":int(np.sum(np.abs(py-ay)>=100)),"attempt_baseline_mae":mae(aa,pa),"attempt_corrected_mae":mae(aa,na),"attempt_mae_gain":mae(aa,pa)-mae(aa,na),"ypa_baseline_mae":mae(ya,by),"ypa_corrected_mae":mae(ya,ny),"ypa_mae_gain":mae(ya,by)-mae(ya,ny),"attempt_only_pass_mae":mae(ay,na*by),"ypa_only_pass_mae":mae(ay,pa*ny)}

def boot(df,ac,yc):
 rng=np.random.default_rng(BOOT_SEED); ay=df.actual_pass_yards.to_numpy(float); ba=df.pred_pass_yards.to_numpy(float); aa=df.actual_attempts.to_numpy(float); pa=df.pred_attempts.to_numpy(float); by=df.base_ypa.to_numpy(float); ya=df.actual_ypa_calc.to_numpy(float); na=np.maximum(1.,pa+ac); ny=np.maximum(.1,by+yc); py=na*ny; n=len(df); gy=[]; ga=[]; gp=[]
 for _ in range(BOOT_N):
  q=rng.integers(0,n,n); gy.append(mae(ay[q],ba[q])-mae(ay[q],py[q])); ga.append(mae(aa[q],pa[q])-mae(aa[q],na[q])); gp.append(mae(ya[q],by[q])-mae(ya[q],ny[q]))
 return {"pass_gain_prob_gt0":float(np.mean(np.array(gy)>0)),"attempt_gain_prob_gt0":float(np.mean(np.array(ga)>0)),"ypa_gain_prob_gt0":float(np.mean(np.array(gp)>0)),"pass_gain_p10":float(np.quantile(gy,.1)),"pass_gain_p50":float(np.quantile(gy,.5)),"pass_gain_p90":float(np.quantile(gy,.9))}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--canonical",required=True); ap.add_argument("--inactives",required=True); ap.add_argument("--out-dir",required=True); a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 base,ina=load_inputs(Path(a.canonical),Path(a.inactives)); rms,sh,ph=prior_sources(out); f=make_features(base,ina,rms,sh,ph); idrate=float(f.attrs["id_rate"])
 d=base.merge(f,on=["season","week","team","opponent"],validate="one_to_one"); d["att_resid"]=num(d.actual_attempts)-num(d.pred_attempts); d["ypa_resid"]=num(d.actual_ypa_calc)-num(d.base_ypa)
 if d[FEATURES].isna().any().any(): raise RuntimeError("feature null")
 devtr=d[(d.season==2024)&(d.week<=DEV_SPLIT_WEEK)]; devte=d[(d.season==2024)&(d.week>DEV_SPLIT_WEEK)]; tr=d[d.season==2024]; te=d[d.season==2025]
 da,_=fit(devtr,devte,"att_resid"); dy,_=fit(devtr,devte,"ypa_resid"); dm=evaluate(devte,da,dy); dm["split"]="2024_w10_18_holdout"
 ta,ca=fit(tr,te,"att_resid"); ty,cy=fit(tr,te,"ypa_resid"); tm=evaluate(te,ta,ty); tm["split"]="2025_untouched_eval"; bt=boot(te,ta,ty)
 comp=(tm["attempt_mae_gain"]>=MIN_ATT_GAIN and bt["attempt_gain_prob_gt0"]>=MIN_COMP_PROB) or (tm["ypa_mae_gain"]>=MIN_YPA_GAIN and bt["ypa_gain_prob_gt0"]>=MIN_COMP_PROB)
 gates=[("inactive_id_bridge_rate",idrate,f">={MIN_ID}",idrate>=MIN_ID),("2024_dev_mae_direction",dm["mae_gain"],">=0",dm["mae_gain"]>=0),("2025_pass_mae_gain",tm["mae_gain"],f">={PROMOTE_MAE}",tm["mae_gain"]>=PROMOTE_MAE),("2025_corr_gain",tm["corr_gain"],f">={PROMOTE_CORR}",tm["corr_gain"]>=PROMOTE_CORR),("2025_100plus_tails_nonincrease",tm["corrected_tails_100"]-tm["baseline_tails_100"],"<=0",tm["corrected_tails_100"]<=tm["baseline_tails_100"]),("2025_pass_bootstrap_support",bt["pass_gain_prob_gt0"],f">={MIN_PASS_PROB}",bt["pass_gain_prob_gt0"]>=MIN_PASS_PROB),("2025_component_signal",1 if comp else 0,"==1",comp)]
 gd=pd.DataFrame(gates,columns=["gate","value","threshold","passed"]); promoted=bool(gd.passed.all())
 pred=te[["season","week","team","opponent","player_clean_key","actual_pass_yards","pred_pass_yards"]].copy(); pred["att_correction"]=ta; pred["ypa_correction"]=ty; pred["corrected_attempts"]=np.maximum(1.,te.pred_attempts.to_numpy(float)+ta); pred["corrected_ypa"]=np.maximum(.1,te.base_ypa.to_numpy(float)+ty); pred["corrected_pass_yards"]=pred.corrected_attempts*pred.corrected_ypa
 interp=pd.DataFrame([{"migration":"M79","status":"PROMOTED_CANDIDATE" if promoted else "REJECTED_FOR_PROMOTION","production_actionable":False,"canonical_rows":len(base),"inactive_team_weeks":len(ina),"id_bridge_rate":idrate,"baseline_2025_mae":tm["baseline_mae"],"m79_2025_mae":tm["corrected_mae"],"mae_gain":tm["mae_gain"],"corr_gain":tm["corr_gain"],"attempt_mae_gain":tm["attempt_mae_gain"],"ypa_mae_gain":tm["ypa_mae_gain"],"baseline_tails_100":tm["baseline_tails_100"],"m79_tails_100":tm["corrected_tails_100"],"pass_gain_prob_gt0":bt["pass_gain_prob_gt0"],"component_supported":comp,"predictive_model_fit":True,"sportsbook_used":False,"next_step":"freeze_and_shadow_integrate" if promoted else "reject_official_inactive_identity_family_no_retest_without_new_information"}])
 d.to_csv(out/"m79_feature_matrix.csv",index=False); pred.to_csv(out/"m79_2025_predictions.csv",index=False); pd.DataFrame([dm,tm]).to_csv(out/"m79_metrics.csv",index=False); pd.concat([ca,cy],ignore_index=True).to_csv(out/"m79_coefficients.csv",index=False); pd.DataFrame([bt]).to_csv(out/"m79_bootstrap.csv",index=False); gd.to_csv(out/"m79_promotion_gates.csv",index=False); interp.to_csv(out/"m79_interpretation.csv",index=False)
 (out/"m79_manifest.json").write_text(json.dumps({"as_of_utc":datetime.now(timezone.utc).isoformat(),"canonical_sha256":CANON_SHA,"inactive_snapshot_sha256":INACTIVE_SHA,"ridge_alpha":RIDGE_ALPHA,"features":FEATURES,"development_boundary":"2024 weeks 1-9 train / 10-18 holdout","final_boundary":"2024 full train / 2025 untouched eval","promotion_gates_frozen_before_first_result":True,"no_post_result_retuning":True,"sportsbook_used":False},indent=2)+"\n")
 print("=== M79 INTERPRETATION ==="); print(interp.to_string(index=False)); print("\n=== METRICS ==="); print(pd.DataFrame([dm,tm]).to_string(index=False)); print("\n=== GATES ==="); print(gd.to_string(index=False)); print("\n=== BOOTSTRAP ==="); print(pd.DataFrame([bt]).to_string(index=False)); return 0
if __name__=="__main__": raise SystemExit(main())
