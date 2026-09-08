#!/usr/bin/env python3
"""RB-R9: training-derived reliability shrinkage for persistent RB receiving identity.

R8 remains a scientific FAIL. Its forensic audit showed that strict-prior receiving
identity carried useful signal, while the largest full-strength redistributions could
overcorrect. R9 is a new frozen hypothesis, not a rescue/tuning of R8 holdout results.

Frozen mechanism
----------------
- preserve the canonical M38 explicit team/RB target pool exactly;
- preserve all non-RB entitlement exactly;
- use the exact R8 receiving-identity feature set and Ridge model;
- estimate ONE reliability multiplier from training data only using rolling-origin
  out-of-fold predictions of the R8 residual model;
- force the reliability multiplier to [0, 1], so baseline remains the anchor;
- apply: calibrated_residual = reliability * raw_R8_residual;
- renormalize only inside the fixed RB/FB room;
- zero sportsbook inputs; zero same/future outcome features.

Freshness governance
--------------------
R9 was designed after observing R8 2018/2019 holdouts. Those seasons are therefore
not valid fresh confirmation for R9. We use:
    train 2015 -> fresh historical confirmation 2016
    train 2016 -> supporting 2017 replication
2017 was previously used as R8 training, so it is supportive rather than pristine.
A PASS authorizes a separate modern-era/prospective confirmation and integration
step; it does not promote R9 by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import metric, read
from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6
from scripts.backtest import evaluate_rb_r8_receiving_identity_v1 as r8
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R9_IDENTITY_SHRINKAGE"
FEATURES = r8.FEATURES
PRED_CLIP = r8.PRED_CLIP

# Frozen before observing 2016/2017 R9 results.
MIN_FRESH_TARGET_MAE_GAIN = 0.02
MIN_FRESH_REC_YARDS_MAE_GAIN = 0.10
MIN_COMBINED_TARGET_MAE_GAIN = 0.02
MIN_COMBINED_REC_YARDS_MAE_GAIN = 0.10
MIN_FRESH_TOP20_REC_YARDS_GAIN = 0.10
MAX_FRESH_REST_REC_YARDS_WORSEN = 0.10
MIN_FRESH_NONWORSE_PHASES = 3
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65
MAX_TAIL_RATE_WORSEN = 0.0025


def _fit_reliability(train: pd.DataFrame):
    """Fit full R8 model plus a rolling-origin, training-only calibration slope."""
    full_model = r8._fit(train)
    oof_parts = []
    # Strict forward blocks: no future training week contributes to its calibration prediction.
    for start, end in ((5, 8), (9, 12), (13, 17)):
        tr = train.loc[pd.to_numeric(train.week, errors="coerce").lt(start)].copy()
        va = train.loc[pd.to_numeric(train.week, errors="coerce").between(start, end)].copy()
        if tr.empty or va.empty:
            continue
        m = r8._fit(tr)
        p = np.clip(m.predict(va[FEATURES]), -PRED_CLIP, PRED_CLIP)
        y = pd.to_numeric(va.within_residual_target, errors="coerce").to_numpy(float)
        oof_parts.append(pd.DataFrame({"week": va.week.to_numpy(), "raw_pred": p, "actual_residual": y}))
    if not oof_parts:
        raise RuntimeError("R9 reliability calibration produced zero rolling-origin rows")
    oof = pd.concat(oof_parts, ignore_index=True)
    p = oof.raw_pred.to_numpy(float); y = oof.actual_residual.to_numpy(float)
    den = float(np.dot(p, p))
    slope_raw = float(np.dot(p, y) / den) if den > 1e-12 else 0.0
    reliability = float(np.clip(slope_raw, 0.0, 1.0))
    oof["shrunk_pred"] = reliability * oof.raw_pred
    cal = {
        "oof_rows": int(len(oof)),
        "raw_slope": slope_raw,
        "reliability": reliability,
        "raw_residual_mae": float(np.mean(np.abs(oof.raw_pred - oof.actual_residual))),
        "shrunk_residual_mae": float(np.mean(np.abs(oof.shrunk_pred - oof.actual_residual))),
        "raw_corr": float(np.corrcoef(oof.raw_pred, oof.actual_residual)[0,1]) if len(oof) > 1 else float("nan"),
    }
    return full_model, reliability, oof, cal


def _apply(baseline: pd.DataFrame, *, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame, model, reliability: float):
    out = baseline.copy()
    x = baseline.copy().reset_index(drop=False).rename(columns={"index":"_row_index"})
    x["position_family"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip().replace({"HB":"RB", "TB":"RB"})
    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x.entitlement_tgt_share, errors="coerce").fillna(0.0)
    rb = x.loc[x.position_family.isin({"RB", "FB"})].copy()
    if rb.empty:
        raise RuntimeError("R9 confirmation week has zero RB rows")
    rb["b0_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    rb["b0_rb_within_share"] = np.where(rb.b0_rb_pool.gt(0), rb.baseline_entitlement_tgt_share / rb.b0_rb_pool, 0.0)
    rb = r8._attach_identity(rb, season, week, states, prev)
    raw = np.clip(model.predict(rb[FEATURES]), -PRED_CLIP, PRED_CLIP)
    rb["r9_raw_r8_residual"] = raw
    rb["r9_reliability"] = reliability
    rb["r9_calibrated_residual"] = reliability * raw
    rb["r9_score"] = np.log(rb.b0_rb_within_share.clip(lower=0.0) + r8.EPS) + rb.r9_calibrated_residual

    audits=[]
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        source_idx = rb.loc[idx, "_row_index"].astype(int).tolist()
        team_idx = out.index[out.event_id.astype(str).eq(str(event_id)) & out.team.astype(str).eq(str(team))]
        base_rb = float(pd.to_numeric(out.loc[source_idx, "entitlement_tgt_share"], errors="raise").sum())
        team_before = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="raise").sum())
        non_idx=[i for i in team_idx if i not in source_idx]
        non_before=pd.to_numeric(out.loc[non_idx,"entitlement_tgt_share"],errors="raise").to_numpy(float) if non_idx else np.array([])
        score=rb.loc[idx,"r9_score"].to_numpy(float)
        w=np.exp(score-np.max(score)); share=w/w.sum(); cand=base_rb*share
        if len(cand): cand[int(np.argmax(share))]+=base_rb-float(cand.sum())
        out.loc[source_idx,"entitlement_tgt_share"]=cand
        rb.loc[idx,"r9_entitlement_tgt_share"]=cand
        rb_after=float(pd.to_numeric(out.loc[source_idx,"entitlement_tgt_share"],errors="raise").sum())
        team_after=float(pd.to_numeric(out.loc[team_idx,"entitlement_tgt_share"],errors="raise").sum())
        non_after=pd.to_numeric(out.loc[non_idx,"entitlement_tgt_share"],errors="raise").to_numpy(float) if non_idx else np.array([])
        audits.append({"season":season,"week":week,"event_id":str(event_id),"team":str(team),
                       "rb_pool_gap":rb_after-base_rb,"team_player_mass_gap":team_after-team_before,
                       "max_non_rb_entitlement_delta":float(np.max(np.abs(non_after-non_before))) if len(non_before) else 0.0,
                       "sportsbook_inputs_used":0,"current_future_outcomes_used_in_features":0,
                       "reliability":reliability})
    return out, rb, pd.DataFrame(audits)


def _fold(*, train_season:int, test_season:int, train_dir:Path, test_dir:Path,
          train_logs:pd.DataFrame, test_logs:pd.DataFrame, history_start:int, iterations:int):
    states, prev = r8._identity_atlas(history_start, test_season)
    train = r8._training_cases(season=train_season, data_dir=train_dir, logs=train_logs, states=states, prev=prev)
    model, reliability, oof, cal = _fit_reliability(train)
    pred_parts=[]; audit_parts=[]
    for week in r8._weeks(test_season):
        baseline=r6._build_bundle_frame(season=test_season,week=week,prior_season=test_season-1,data_dir=test_dir,logs=test_logs)
        candidate,rbfeat,audit=_apply(baseline,season=test_season,week=week,states=states,prev=prev,model=model,reliability=reliability)
        audit_parts.append(audit)
        seed=909000+test_season*100+week
        bsim=explicit_simulate(baseline,iterations=iterations,seed=seed)
        csim=explicit_simulate(candidate,iterations=iterations,seed=seed)
        at=r6._actual_target_frame(test_logs,test_season,week); ay=r6._actual_yards_frame(test_logs,test_season,week)
        identity=rbfeat[["team","player_clean_key","prior_rb_room_share","r9_raw_r8_residual","r9_calibrated_residual","r9_reliability"]].copy()
        for variant,frame,sim in ((BASE,baseline,bsim),(CAND,candidate,csim)):
            p=r6._prediction_rows(frame,sim,variant,test_season,week)
            p=p.merge(at,on=["team","player_clean_key"],how="inner").merge(ay,on=["team","player_clean_key"],how="inner")
            p=p.merge(identity,on=["team","player_clean_key"],how="left")
            p["train_season"]=train_season; pred_parts.append(p)
        print(f"[rb-r9] train={train_season} test={test_season} week={week:02d} reliability={reliability:.6f}")
    sc=model.named_steps["standardscaler"]; rg=model.named_steps["ridge"]
    coef=pd.DataFrame({"train_season":train_season,"test_season":test_season,"feature":FEATURES,
                       "scaler_mean":sc.mean_,"scaler_scale":sc.scale_,"ridge_coef":rg.coef_,"ridge_intercept":float(rg.intercept_),
                       "reliability":reliability})
    oof["train_season"]=train_season; oof["test_season"]=test_season
    return pd.concat(pred_parts,ignore_index=True),pd.concat(audit_parts,ignore_index=True),coef,oof,cal,len(train)


def _summaries(pred:pd.DataFrame):
    x=pred.loc[pred.position_family.isin({"RB","FB"})].copy()
    x["abs_err"]=(pd.to_numeric(x.mc_rec_yards,errors="coerce")-pd.to_numeric(x.actual_rec_yards,errors="coerce")).abs()
    x["phase"]=pd.cut(x.week,[0,4,9,13,18],labels=["W1-4","W5-9","W10-13","W14-18"])
    x["identity_pct"]=x.groupby(["variant","season","week"])["prior_rb_room_share"].rank(pct=True,method="average")
    x["identity_bucket"]=np.where(x.identity_pct.gt(.80),"TOP20","REST80")
    rows=[]; phases=[]; buckets=[]
    for bucket,g0 in [("COMBINED",x)]+[(str(s),g) for s,g in x.groupby("season")]:
        for variant,g in g0.groupby("variant"):
            for market,ac,pc in (("targets","actual_targets","pred_targets"),("rec_yards","actual_rec_yards","mc_rec_yards")):
                r={"season_bucket":bucket,"variant":variant,"market":market,**metric(g[ac],g[pc])}
                if market=="rec_yards":
                    r["miss_30_plus_rate"]=float(g.abs_err.ge(30).mean()); r["miss_50_plus_rate"]=float(g.abs_err.ge(50).mean())
                rows.append(r)
    for s,sg in x.groupby("season"):
        for (v,p),g in sg.groupby(["variant","phase"],observed=False):
            if len(g): phases.append({"season":int(s),"phase":str(p),"variant":v,**metric(g.actual_rec_yards,g.mc_rec_yards)})
        for (v,b),g in sg.groupby(["variant","identity_bucket"]):
            r={"season":int(s),"variant":v,"identity_bucket":b,**metric(g.actual_rec_yards,g.mc_rec_yards)}
            r["miss_30_plus_rate"]=float(g.abs_err.ge(30).mean()); r["miss_50_plus_rate"]=float(g.abs_err.ge(50).mean())
            buckets.append(r)
    return pd.DataFrame(rows),pd.DataFrame(phases),pd.DataFrame(buckets),x


def _bootstrap(x:pd.DataFrame,season:int,reps:int,seed:int=90909)->float:
    keys=["season","week","team","player_clean_key"]
    b=x.loc[x.season.eq(season)&x.variant.eq(BASE),keys+["actual_rec_yards","mc_rec_yards"]]
    c=x.loc[x.season.eq(season)&x.variant.eq(CAND),keys+["actual_rec_yards","mc_rec_yards"]]
    z=b.merge(c,on=keys,suffixes=("_b","_c"),validate="one_to_one")
    eb=(z.mc_rec_yards_b-z.actual_rec_yards_b).abs().to_numpy(float); ec=(z.mc_rec_yards_c-z.actual_rec_yards_c).abs().to_numpy(float)
    rng=np.random.default_rng(seed); wins=0
    for _ in range(reps):
        ii=rng.integers(0,len(z),len(z)); wins+=int(ec[ii].mean()<eb[ii].mean())
    return wins/reps


def main()->int:
    ap=argparse.ArgumentParser()
    for s in (2015,2016,2017):
        ap.add_argument(f"--data-{s}",dest=f"data_{s}",type=Path,required=True)
        ap.add_argument(f"--logs-{s}",dest=f"logs_{s}",type=Path,required=True)
    ap.add_argument("--history-start",type=int,default=2013)
    ap.add_argument("--iterations",type=int,default=2000); ap.add_argument("--bootstrap-reps",type=int,default=2000)
    ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_r9_receiving_identity_shrinkage_v1"))
    a=ap.parse_args(); data={s:getattr(a,f"data_{s}") for s in (2015,2016,2017)}; logs={s:read(getattr(a,f"logs_{s}")) for s in (2015,2016,2017)}
    pp=[]; aa=[]; cc=[]; oo=[]; meta=[]
    for tr,te,label in ((2015,2016,"fresh_historical_confirmation"),(2016,2017,"supporting_replication")):
        p,au,co,oof,cal,n=_fold(train_season=tr,test_season=te,train_dir=data[tr],test_dir=data[te],train_logs=logs[tr],test_logs=logs[te],history_start=a.history_start,iterations=a.iterations)
        pp.append(p); aa.append(au); cc.append(co); oo.append(oof); meta.append({"train_season":tr,"test_season":te,"role":label,"train_rows":n,**cal})
    pred=pd.concat(pp,ignore_index=True); audit=pd.concat(aa,ignore_index=True); coef=pd.concat(cc,ignore_index=True); oof=pd.concat(oo,ignore_index=True)
    summary,phases,buckets,x=_summaries(pred)
    def sr(s,v,m): return summary.loc[summary.season_bucket.eq(str(s))&summary.variant.eq(v)&summary.market.eq(m)].iloc[0]
    def br(s,v,b): return buckets.loc[buckets.season.eq(s)&buckets.variant.eq(v)&buckets.identity_bucket.eq(b)].iloc[0]
    fbt,fct=sr(2016,BASE,"targets"),sr(2016,CAND,"targets"); fby,fcy=sr(2016,BASE,"rec_yards"),sr(2016,CAND,"rec_yards")
    rbt,rct=sr(2017,BASE,"targets"),sr(2017,CAND,"targets"); rby,rcy=sr(2017,BASE,"rec_yards"),sr(2017,CAND,"rec_yards")
    cbt,cct=sr("COMBINED",BASE,"targets"),sr("COMBINED",CAND,"targets"); cby,ccy=sr("COMBINED",BASE,"rec_yards"),sr("COMBINED",CAND,"rec_yards")
    topb,topc=br(2016,BASE,"TOP20"),br(2016,CAND,"TOP20"); restb,restc=br(2016,BASE,"REST80"),br(2016,CAND,"REST80")
    fresh_nonworse=0
    for ph in phases.loc[phases.season.eq(2016),"phase"].unique():
        b=phases.loc[phases.season.eq(2016)&phases.phase.eq(ph)&phases.variant.eq(BASE)]; c=phases.loc[phases.season.eq(2016)&phases.phase.eq(ph)&phases.variant.eq(CAND)]
        if len(b) and len(c) and float(c.iloc[0].mae)<=float(b.iloc[0].mae)+1e-12: fresh_nonworse+=1
    boot=_bootstrap(x,2016,a.bootstrap_reps)
    gates={
        "fresh_target_gain":float(fbt.mae-fct.mae)>=MIN_FRESH_TARGET_MAE_GAIN,
        "fresh_rec_yards_gain":float(fby.mae-fcy.mae)>=MIN_FRESH_REC_YARDS_MAE_GAIN,
        "fresh_p90_nonworse":float(fcy.p90_abs_error)<=float(fby.p90_abs_error)+1e-12,
        "fresh_cat30_guard":float(fcy.miss_30_plus_rate)<=float(fby.miss_30_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_cat50_guard":float(fcy.miss_50_plus_rate)<=float(fby.miss_50_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_bias_abs_nonworse":abs(float(fcy.bias))<=abs(float(fby.bias))+1e-12,
        "fresh_phase_nonworse":fresh_nonworse>=MIN_FRESH_NONWORSE_PHASES,
        "fresh_bootstrap":boot>=MIN_BOOTSTRAP_IMPROVE_PROB,
        "fresh_top20_gain":float(topb.mae-topc.mae)>=MIN_FRESH_TOP20_REC_YARDS_GAIN,
        "fresh_top20_p90_nonworse":float(topc.p90_abs_error)<=float(topb.p90_abs_error)+1e-12,
        "fresh_top20_cat30_guard":float(topc.miss_30_plus_rate)<=float(topb.miss_30_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_top20_cat50_guard":float(topc.miss_50_plus_rate)<=float(topb.miss_50_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_rest80_guard":float(restc.mae)<=float(restb.mae)+MAX_FRESH_REST_REC_YARDS_WORSEN,
        "rep_target_nonworse":float(rct.mae)<=float(rbt.mae)+1e-12,
        "rep_rec_yards_nonworse":float(rcy.mae)<=float(rby.mae)+1e-12,
        "combined_target_gain":float(cbt.mae-cct.mae)>=MIN_COMBINED_TARGET_MAE_GAIN,
        "combined_rec_yards_gain":float(cby.mae-ccy.mae)>=MIN_COMBINED_REC_YARDS_MAE_GAIN,
        "rb_pool_conservation":float(audit.rb_pool_gap.abs().max())<=1e-12,
        "team_mass_conservation":float(audit.team_player_mass_gap.abs().max())<=1e-12,
        "non_rb_exact":float(audit.max_non_rb_entitlement_delta.max())<=1e-12,
        "sportsbook_zero":int(audit.sportsbook_inputs_used.max())==0,
        "future_feature_zero":int(audit.current_future_outcomes_used_in_features.max())==0,
        "reliability_bounded":bool(((audit.reliability>=0)&(audit.reliability<=1)).all()),
    }
    passed=all(gates.values())
    result={"candidate":"RB_R9_RECEIVING_IDENTITY_SHRINKAGE_V1","disposition":"RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS" if passed else "RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_FAIL",
            "science_pass":passed,"fresh_confirmation_season":2016,"replication_season":2017,"replication_note":"2017 was previously used as R8 training; supportive, not pristine",
            "r8_status":"RB_R8_RECEIVING_IDENTITY_OOS_FAIL_UNCHANGED","history_start":a.history_start,"sportsbook_inputs_used":0,
            "feature_contract":"exact R8 strict-prior receiving identity features; training-only rolling-origin reliability slope",
            "features":FEATURES,"folds":meta,"fresh_bootstrap_improve_probability":boot,"fresh_nonworse_phases":fresh_nonworse,
            "fresh_top20_baseline_mae":float(topb.mae),"fresh_top20_candidate_mae":float(topc.mae),
            "fresh_rest80_baseline_mae":float(restb.mae),"fresh_rest80_candidate_mae":float(restc.mae),"gates":gates}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    pred.to_csv(a.out_dir/"rb_r9_predictions.csv",index=False); audit.to_csv(a.out_dir/"rb_r9_conservation_audit.csv",index=False)
    coef.to_csv(a.out_dir/"rb_r9_coefficients.csv",index=False); oof.to_csv(a.out_dir/"rb_r9_reliability_oof.csv",index=False)
    summary.to_csv(a.out_dir/"rb_r9_market_summary.csv",index=False); phases.to_csv(a.out_dir/"rb_r9_phase_summary.csv",index=False); buckets.to_csv(a.out_dir/"rb_r9_identity_bucket_summary.csv",index=False)
    (a.out_dir/"rb_r9_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print("\n=== market summary ===\n",summary.to_string(index=False)); print("\n=== identity buckets ===\n",buckets.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())