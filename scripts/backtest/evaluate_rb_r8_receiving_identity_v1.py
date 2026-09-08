#!/usr/bin/env python3
"""RB-R8: persistent receiving identity within a fixed canonical RB target room.

Motivation is frozen from RB_RECEIVING_IDENTITY_DIAGNOSTIC_V1: receiving work is
not a universal RB trait. Strict-prior target/reception/RB-room history strongly
concentrated future 5+/7+ target games and 2025 catastrophic receiving-yard misses.
R7 showed generic offensive-snap participation was not sufficient.

Frozen candidate
----------------
- start from canonical M38 explicit target entitlement;
- preserve each team-game RB/FB target pool exactly;
- preserve every non-RB entitlement and total team player mass exactly;
- redistribute only within RB/FB using strict-prior receiving-identity history;
- use all pre-specified non-route history features from the diagnostic, plus
  history-count availability terms;
- StandardScaler + Ridge(alpha=20) on the same log-share residual target used by
  R6/R7; train clip [-2,2], inference clip [-1,1];
- no sportsbook inputs and no same/future outcomes in confirmation features.

Freshness governance
--------------------
The identity hypothesis was discovered on 2022-2025 outcomes. R7 already examined
2020/2021. R8 therefore uses older outcomes that were not inspected by this RB
receiving research family as reverse-time holdouts:
    train 2017 -> untouched confirmation 2018
    train 2018 -> untouched replication 2019
These are historical OOS tests, not prospective evidence. A PASS only authorizes a
separate modern-era replication/refit/integration step; it does not promote R8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import metric, read
from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6
from scripts.backtest import audit_rb_receiving_identity_v1 as ident
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R8_RECEIVING_IDENTITY"
EPS = r6.EPS
ALPHA = 20.0
TRAIN_CLIP = 2.0
PRED_CLIP = 1.0

FEATURES = [
    "prior_targets_pg", "prior_receptions_pg", "prior_target_share", "prior_rb_room_share",
    "prior_5plus_target_rate", "prior_7plus_target_rate",
    "last8_targets_pg", "last8_receptions_pg", "last8_target_share", "last8_rb_room_share",
    "prev_season_targets_pg", "prev_season_receptions_pg", "prev_season_target_share", "prev_season_rb_room_share",
    "same_team_prior_targets_pg", "same_team_prior_rb_room_share",
    "log1p_prior_games", "log1p_same_team_prior_games", "prev_season_available",
]

# Frozen before observing 2018/2019 R8 outcomes.
MIN_FRESH_TARGET_MAE_GAIN = 0.02
MIN_FRESH_REC_YARDS_MAE_GAIN = 0.10
MIN_COMBINED_TARGET_MAE_GAIN = 0.02
MIN_COMBINED_REC_YARDS_MAE_GAIN = 0.10
MIN_FRESH_TOP20_REC_YARDS_GAIN = 0.25
MAX_FRESH_REST_REC_YARDS_WORSEN = 0.10
MIN_FRESH_NONWORSE_PHASES = 3
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65
MAX_TAIL_RATE_WORSEN = 0.0025


def _weeks(season: int) -> range:
    return range(1, 18 if season <= 2020 else 19)


def _identity_atlas(history_start: int, through_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = ident._add_rb_room_share(ident._load_logs(list(range(history_start, through_season + 1))))
    rb = logs.loc[logs.position_family.isin({"RB", "FB"})].copy()
    states = ident._build_states(rb)
    prev = ident._previous_season_features(rb)
    return states, prev


def _attach_identity(rb: pd.DataFrame, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    q = rb[["player_clean_key", "team"]].copy()
    q["season"] = int(season); q["week"] = int(week)
    feat = ident._snapshot_queries(q, states, prev)
    keep = ["player_clean_key", "team", "season", "week"] + [c for c in feat.columns if c in set(FEATURES) | {
        "prior_games", "same_team_prior_games", "prev_season_games", "prior_rb_room_share"
    }]
    feat = feat[keep].copy()
    feat["log1p_prior_games"] = np.log1p(pd.to_numeric(feat.get("prior_games", 0), errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_same_team_prior_games"] = np.log1p(pd.to_numeric(feat.get("same_team_prior_games", 0), errors="coerce").fillna(0).clip(lower=0))
    feat["prev_season_available"] = pd.to_numeric(feat.get("prev_season_games", np.nan), errors="coerce").notna().astype(float)
    for c in FEATURES:
        if c not in feat.columns:
            feat[c] = 0.0
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    out = rb.merge(feat[["player_clean_key", "team", *FEATURES]], on=["player_clean_key", "team"], how="left", validate="one_to_one")
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _training_cases(*, season: int, data_dir: Path, logs: pd.DataFrame, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for week in _weeks(season):
        baseline = r6._build_bundle_frame(season=season, week=week, prior_season=season - 1, data_dir=data_dir, logs=logs)
        x = baseline.copy().reset_index(drop=False).rename(columns={"index": "_row_index"})
        x["position_family"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip().replace({"HB":"RB", "TB":"RB"})
        x["baseline_entitlement_tgt_share"] = pd.to_numeric(x.entitlement_tgt_share, errors="coerce").fillna(0.0)
        rb = x.loc[x.position_family.isin({"RB", "FB"})].copy()
        rb["b0_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
        rb["b0_rb_within_share"] = np.where(rb.b0_rb_pool.gt(0), rb.baseline_entitlement_tgt_share / rb.b0_rb_pool, 0.0)
        rb = _attach_identity(rb, season, week, states, prev)
        at = r6._actual_target_frame(logs, season, week)
        rb = rb.merge(at, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        rb["actual_targets"] = pd.to_numeric(rb.actual_targets, errors="coerce").fillna(0.0)
        rb["actual_rb_targets"] = rb.groupby(["event_id", "team"])["actual_targets"].transform("sum")
        rb["actual_rb_within_share"] = np.where(rb.actual_rb_targets.gt(0), rb.actual_targets / rb.actual_rb_targets, 0.0)
        rb["within_residual_target"] = (
            np.log(rb.actual_rb_within_share.clip(lower=0.0) + EPS)
            - np.log(rb.b0_rb_within_share.clip(lower=0.0) + EPS)
        ).clip(-TRAIN_CLIP, TRAIN_CLIP)
        rb["season"] = season; rb["week"] = week
        parts.append(rb.loc[rb.actual_rb_targets.gt(0) & rb.b0_rb_pool.gt(0)].copy())
        print(f"[rb-r8] training season={season} week={week:02d} rows={len(rb)}")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        raise RuntimeError(f"R8 empty training casebook season={season}")
    return out


def _fit(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(train[FEATURES], train["within_residual_target"])
    return model


def _apply(baseline: pd.DataFrame, *, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame, model):
    out = baseline.copy()
    x = baseline.copy().reset_index(drop=False).rename(columns={"index":"_row_index"})
    x["position_family"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip().replace({"HB":"RB", "TB":"RB"})
    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x.entitlement_tgt_share, errors="coerce").fillna(0.0)
    rb = x.loc[x.position_family.isin({"RB", "FB"})].copy()
    if rb.empty:
        raise RuntimeError("R8 confirmation week has zero RB rows")
    rb["b0_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    rb["b0_rb_within_share"] = np.where(rb.b0_rb_pool.gt(0), rb.baseline_entitlement_tgt_share / rb.b0_rb_pool, 0.0)
    rb = _attach_identity(rb, season, week, states, prev)
    pred = np.clip(model.predict(rb[FEATURES]), -PRED_CLIP, PRED_CLIP)
    rb["r8_predicted_residual"] = pred
    rb["r8_score"] = np.log(rb.b0_rb_within_share.clip(lower=0.0) + EPS) + pred

    audits = []
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        source_idx = rb.loc[idx, "_row_index"].astype(int).tolist()
        team_idx = out.index[out.event_id.astype(str).eq(str(event_id)) & out.team.astype(str).eq(str(team))]
        base_rb = float(pd.to_numeric(out.loc[source_idx, "entitlement_tgt_share"], errors="raise").sum())
        team_before = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="raise").sum())
        non_idx = [i for i in team_idx if i not in source_idx]
        non_before = pd.to_numeric(out.loc[non_idx, "entitlement_tgt_share"], errors="raise").to_numpy(float) if non_idx else np.array([])
        score = rb.loc[idx, "r8_score"].to_numpy(float)
        w = np.exp(score - np.max(score)); share = w / w.sum()
        cand = base_rb * share
        if len(cand): cand[int(np.argmax(share))] += base_rb - float(cand.sum())
        out.loc[source_idx, "entitlement_tgt_share"] = cand
        rb.loc[idx, "r8_entitlement_tgt_share"] = cand
        rb_after = float(pd.to_numeric(out.loc[source_idx, "entitlement_tgt_share"], errors="raise").sum())
        team_after = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="raise").sum())
        non_after = pd.to_numeric(out.loc[non_idx, "entitlement_tgt_share"], errors="raise").to_numpy(float) if non_idx else np.array([])
        audits.append({
            "season": season, "week": week, "event_id": str(event_id), "team": str(team),
            "rb_pool_gap": rb_after - base_rb, "team_player_mass_gap": team_after - team_before,
            "max_non_rb_entitlement_delta": float(np.max(np.abs(non_after-non_before))) if len(non_before) else 0.0,
            "sportsbook_inputs_used": 0, "current_future_outcomes_used_in_features": 0,
        })
    return out, rb, pd.DataFrame(audits)


def _fold(*, train_season: int, test_season: int, train_dir: Path, test_dir: Path, train_logs: pd.DataFrame, test_logs: pd.DataFrame,
          history_start: int, iterations: int):
    states, prev = _identity_atlas(history_start, test_season)
    train = _training_cases(season=train_season, data_dir=train_dir, logs=train_logs, states=states, prev=prev)
    model = _fit(train)
    pred_parts, audit_parts = [], []
    for week in _weeks(test_season):
        baseline = r6._build_bundle_frame(season=test_season, week=week, prior_season=test_season-1, data_dir=test_dir, logs=test_logs)
        candidate, rbfeat, audit = _apply(baseline, season=test_season, week=week, states=states, prev=prev, model=model)
        audit_parts.append(audit)
        seed = 808000 + test_season*100 + week
        bsim = explicit_simulate(baseline, iterations=iterations, seed=seed)
        csim = explicit_simulate(candidate, iterations=iterations, seed=seed)
        at = r6._actual_target_frame(test_logs, test_season, week)
        ay = r6._actual_yards_frame(test_logs, test_season, week)
        identity = rbfeat[["team", "player_clean_key", "prior_rb_room_share"]].copy()
        for variant, frame, sim in ((BASE, baseline, bsim), (CAND, candidate, csim)):
            p = r6._prediction_rows(frame, sim, variant, test_season, week)
            p = p.merge(at, on=["team", "player_clean_key"], how="inner").merge(ay, on=["team", "player_clean_key"], how="inner")
            p = p.merge(identity, on=["team", "player_clean_key"], how="left")
            p["train_season"] = train_season
            pred_parts.append(p)
        print(f"[rb-r8] confirm train={train_season} test={test_season} week={week:02d}")
    sc = model.named_steps["standardscaler"]; rg = model.named_steps["ridge"]
    coef = pd.DataFrame({"train_season":train_season, "test_season":test_season, "feature":FEATURES,
                         "scaler_mean":sc.mean_, "scaler_scale":sc.scale_, "ridge_coef":rg.coef_, "ridge_intercept":float(rg.intercept_)})
    return pd.concat(pred_parts, ignore_index=True), pd.concat(audit_parts, ignore_index=True), coef, len(train)


def _summaries(pred: pd.DataFrame):
    x = pred.loc[pred.position_family.isin({"RB", "FB"})].copy()
    x["abs_err"] = (pd.to_numeric(x.mc_rec_yards, errors="coerce") - pd.to_numeric(x.actual_rec_yards, errors="coerce")).abs()
    x["phase"] = pd.cut(x.week, [0,4,9,13,18], labels=["W1-4","W5-9","W10-13","W14-18"])
    # Identity top 20% is computed within each season-week from strict-prior RB-room share.
    x["identity_pct"] = x.groupby(["variant","season","week"])["prior_rb_room_share"].rank(pct=True, method="average")
    x["identity_bucket"] = np.where(x.identity_pct.gt(.80), "TOP20", "REST80")
    rows=[]
    for bucket,g0 in [("COMBINED",x)] + [(str(s),g) for s,g in x.groupby("season")]:
        for variant,g in g0.groupby("variant"):
            for market,ac,pc in (("targets","actual_targets","pred_targets"),("rec_yards","actual_rec_yards","mc_rec_yards")):
                r={"season_bucket":bucket,"variant":variant,"market":market,**metric(g[ac],g[pc])}
                if market=="rec_yards":
                    r["miss_30_plus_rate"]=float(g.abs_err.ge(30).mean()); r["miss_50_plus_rate"]=float(g.abs_err.ge(50).mean())
                rows.append(r)
    phases=[]
    for s,sg in x.groupby("season"):
        for (v,p),g in sg.groupby(["variant","phase"],observed=False):
            if len(g): phases.append({"season":int(s),"phase":str(p),"variant":v,**metric(g.actual_rec_yards,g.mc_rec_yards)})
    buckets=[]
    for s,sg in x.groupby("season"):
        for (v,b),g in sg.groupby(["variant","identity_bucket"]):
            buckets.append({"season":int(s),"variant":v,"identity_bucket":b,**metric(g.actual_rec_yards,g.mc_rec_yards),
                            "bias":float((g.mc_rec_yards-g.actual_rec_yards).mean())})
    return pd.DataFrame(rows),pd.DataFrame(phases),pd.DataFrame(buckets),x


def _bootstrap(x: pd.DataFrame, season: int, reps: int, seed: int=80808) -> float:
    keys=["season","week","team","player_clean_key"]
    b=x.loc[x.season.eq(season)&x.variant.eq(BASE),keys+["actual_rec_yards","mc_rec_yards"]]
    c=x.loc[x.season.eq(season)&x.variant.eq(CAND),keys+["actual_rec_yards","mc_rec_yards"]]
    z=b.merge(c,on=keys,suffixes=("_b","_c"),validate="one_to_one")
    eb=(z.mc_rec_yards_b-z.actual_rec_yards_b).abs().to_numpy(float); ec=(z.mc_rec_yards_c-z.actual_rec_yards_c).abs().to_numpy(float)
    rng=np.random.default_rng(seed); wins=0
    for _ in range(reps):
        ii=rng.integers(0,len(z),len(z)); wins += int(ec[ii].mean() < eb[ii].mean())
    return wins/reps


def main() -> int:
    ap=argparse.ArgumentParser()
    for s in (2017,2018,2019):
        ap.add_argument(f"--data-{s}",dest=f"data_{s}",type=Path,required=True)
        ap.add_argument(f"--logs-{s}",dest=f"logs_{s}",type=Path,required=True)
    ap.add_argument("--history-start",type=int,default=2015)
    ap.add_argument("--iterations",type=int,default=2000)
    ap.add_argument("--bootstrap-reps",type=int,default=2000)
    ap.add_argument("--out-dir",type=Path,default=Path("data/backtests/rb_r8_receiving_identity_v1"))
    a=ap.parse_args()
    data={s:getattr(a,f"data_{s}") for s in (2017,2018,2019)}
    logs={s:read(getattr(a,f"logs_{s}")) for s in (2017,2018,2019)}
    pp=[]; aa=[]; cc=[]; meta=[]
    for tr,te in ((2017,2018),(2018,2019)):
        p,au,co,n=_fold(train_season=tr,test_season=te,train_dir=data[tr],test_dir=data[te],train_logs=logs[tr],test_logs=logs[te],history_start=a.history_start,iterations=a.iterations)
        pp.append(p); aa.append(au); cc.append(co); meta.append({"train_season":tr,"test_season":te,"train_rows":n,"untouched_holdout":True})
    pred=pd.concat(pp,ignore_index=True); audit=pd.concat(aa,ignore_index=True); coef=pd.concat(cc,ignore_index=True)
    summary,phases,buckets,x=_summaries(pred)
    def sr(s,v,m): return summary.loc[summary.season_bucket.eq(str(s))&summary.variant.eq(v)&summary.market.eq(m)].iloc[0]
    fbt,fct=sr(2018,BASE,"targets"),sr(2018,CAND,"targets"); fby,fcy=sr(2018,BASE,"rec_yards"),sr(2018,CAND,"rec_yards")
    rbt,rct=sr(2019,BASE,"targets"),sr(2019,CAND,"targets"); rby,rcy=sr(2019,BASE,"rec_yards"),sr(2019,CAND,"rec_yards")
    cbt,cct=sr("COMBINED",BASE,"targets"),sr("COMBINED",CAND,"targets"); cby,ccy=sr("COMBINED",BASE,"rec_yards"),sr("COMBINED",CAND,"rec_yards")
    def br(s,v,b): return buckets.loc[buckets.season.eq(s)&buckets.variant.eq(v)&buckets.identity_bucket.eq(b)].iloc[0]
    topb,topc=br(2018,BASE,"TOP20"),br(2018,CAND,"TOP20"); restb,restc=br(2018,BASE,"REST80"),br(2018,CAND,"REST80")
    fresh_nonworse=0
    for phase in phases.loc[phases.season.eq(2018),"phase"].unique():
        b=phases.loc[phases.season.eq(2018)&phases.phase.eq(phase)&phases.variant.eq(BASE)]
        c=phases.loc[phases.season.eq(2018)&phases.phase.eq(phase)&phases.variant.eq(CAND)]
        if len(b) and len(c) and float(c.iloc[0].mae)<=float(b.iloc[0].mae)+1e-12: fresh_nonworse+=1
    boot=_bootstrap(x,2018,a.bootstrap_reps)
    gates={
        "fresh_target_gain":float(fbt.mae-fct.mae)>=MIN_FRESH_TARGET_MAE_GAIN,
        "fresh_rec_yards_gain":float(fby.mae-fcy.mae)>=MIN_FRESH_REC_YARDS_MAE_GAIN,
        "fresh_p90_nonworse":float(fcy.p90_abs_error)<=float(fby.p90_abs_error)+1e-12,
        "fresh_cat30_guard":float(fcy.miss_30_plus_rate)<=float(fby.miss_30_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_cat50_guard":float(fcy.miss_50_plus_rate)<=float(fby.miss_50_plus_rate)+MAX_TAIL_RATE_WORSEN,
        "fresh_phase_nonworse":fresh_nonworse>=MIN_FRESH_NONWORSE_PHASES,
        "fresh_bootstrap":boot>=MIN_BOOTSTRAP_IMPROVE_PROB,
        "fresh_top20_gain":float(topb.mae-topc.mae)>=MIN_FRESH_TOP20_REC_YARDS_GAIN,
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
    }
    passed=all(gates.values())
    result={"candidate":"RB_R8_RECEIVING_IDENTITY_V1","disposition":"RB_R8_RECEIVING_IDENTITY_OOS_PASS" if passed else "RB_R8_RECEIVING_IDENTITY_OOS_FAIL",
            "science_pass":passed,"fresh_confirmation_season":2018,"replication_season":2019,"history_start":a.history_start,
            "diagnostic_discovery_seasons":[2022,2023,2024,2025],"sportsbook_inputs_used":0,"feature_contract":"strict-prior completed player receiving history only",
            "features":FEATURES,"alpha":ALPHA,"pred_clip":PRED_CLIP,"train_clip":TRAIN_CLIP,"folds":meta,
            "fresh_bootstrap_improve_probability":boot,"fresh_nonworse_phases":fresh_nonworse,
            "fresh_top20_baseline_mae":float(topb.mae),"fresh_top20_candidate_mae":float(topc.mae),
            "fresh_rest80_baseline_mae":float(restb.mae),"fresh_rest80_candidate_mae":float(restc.mae),"gates":gates}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    pred.to_csv(a.out_dir/"rb_r8_predictions.csv",index=False); audit.to_csv(a.out_dir/"rb_r8_conservation_audit.csv",index=False)
    coef.to_csv(a.out_dir/"rb_r8_coefficients.csv",index=False); summary.to_csv(a.out_dir/"rb_r8_market_summary.csv",index=False)
    phases.to_csv(a.out_dir/"rb_r8_phase_summary.csv",index=False); buckets.to_csv(a.out_dir/"rb_r8_identity_bucket_summary.csv",index=False)
    (a.out_dir/"rb_r8_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); print("\n=== market summary ===\n",summary.to_string(index=False)); print("\n=== identity buckets ===\n",buckets.to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
