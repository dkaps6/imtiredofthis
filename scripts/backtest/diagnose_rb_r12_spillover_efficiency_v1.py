#!/usr/bin/env python3
"""Diagnostic-only R12 conservation-spillover and receiving-efficiency decomposition.

Consumes the mechanically corrected RB-only R12 artifact. No model is refit and no
candidate thresholds are changed. Two questions are isolated:

1. Did exact RB-pool conservation/rescaling create the small target-MAE regression
   versus R9 by moving otherwise-good REST80 rows?
2. After R12 improved TOP20 target MAE, is the remaining receiving-yard error now
   dominated by the frozen yards-per-target mapping rather than target entitlement?

Postgame actual targets/YPT are oracle diagnostics only and are never deployable
features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _m(a, p):
    a = pd.to_numeric(a, errors="coerce").to_numpy(float)
    p = pd.to_numeric(p, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90": np.nan}
    e = p - a
    ae = np.abs(e)
    return {"n": int(len(a)), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(e*e))), "bias": float(e.mean()), "p90": float(np.quantile(ae, .90))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    x = pd.read_csv(a.predictions, low_memory=False)
    req = {
        "season","week","team","player_clean_key","identity_bucket","actual_targets","actual_rec_yards",
        "baseline_pred_targets","r9_pred_targets","r12_targets_preconserve","r12_pred_targets",
        "baseline_pred_rec_yards","r9_pred_rec_yards","r12_pred_rec_yards","frozen_ypt",
        "baseline_team_rb_targets","r12_team_preconserve","team_pool_gap","state_probability","blend_alpha",
    }
    missing = sorted(req - set(x.columns))
    if missing:
        raise RuntimeError(f"corrected R12 predictions missing columns: {missing}")

    for c in req - {"team","player_clean_key","identity_bucket"}:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x["target_state"] = pd.cut(x.actual_targets, [-np.inf,2,4,6,np.inf], labels=["0-2","3-4","5-6","7+"])
    x["preconserve_rec_yards"] = x.r12_targets_preconserve * x.frozen_ypt
    x["conservation_target_shift"] = x.r12_pred_targets - x.r12_targets_preconserve
    x["conservation_rec_shift"] = x.r12_pred_rec_yards - x.preconserve_rec_yards
    x["team_scale"] = np.where(x.r12_team_preconserve.abs().gt(1e-12), x.baseline_team_rb_targets / x.r12_team_preconserve, 1.0)

    # Oracle decompositions. Actual-target + frozen YPT isolates efficiency error.
    x["actual_target_frozen_ypt_oracle"] = x.actual_targets * x.frozen_ypt
    x["actual_ypt"] = np.where(x.actual_targets.gt(0), x.actual_rec_yards / x.actual_targets, np.nan)
    x["r12_target_actual_ypt_oracle"] = x.r12_pred_targets * x.actual_ypt
    x["baseline_target_actual_ypt_oracle"] = x.baseline_pred_targets * x.actual_ypt

    summary_rows = []
    for pop, g0 in [("ALL", x), ("TOP20", x.loc[x.identity_bucket.eq("TOP20")]), ("REST80", x.loc[x.identity_bucket.eq("REST80")])]:
        for stage, tc, rc in [
            ("BASELINE", "baseline_pred_targets", "baseline_pred_rec_yards"),
            ("R9", "r9_pred_targets", "r9_pred_rec_yards"),
            ("R12_PRECONSERVE", "r12_targets_preconserve", "preconserve_rec_yards"),
            ("R12_POSTCONSERVE", "r12_pred_targets", "r12_pred_rec_yards"),
        ]:
            tm = _m(g0.actual_targets, g0[tc]); rm = _m(g0.actual_rec_yards, g0[rc])
            summary_rows.append({"population":pop,"stage":stage,"n":len(g0),"target_mae":tm["mae"],"target_bias":tm["bias"],"rec_mae":rm["mae"],"rec_rmse":rm["rmse"],"rec_bias":rm["bias"],"rec_p90":rm["p90"]})
    summary = pd.DataFrame(summary_rows)

    oracle_rows = []
    for pop, g0 in [("ALL",x),("TOP20",x.loc[x.identity_bucket.eq("TOP20")]),("REST80",x.loc[x.identity_bucket.eq("REST80")])]:
        for state, g in [("ALL_STATES",g0)] + [(str(s),h) for s,h in g0.groupby("target_state", observed=False) if len(h)]:
            r12 = _m(g.actual_rec_yards, g.r12_pred_rec_yards)
            eff = _m(g.actual_rec_yards, g.actual_target_frozen_ypt_oracle)
            pos = g.loc[g.actual_targets.gt(0)].copy()
            target_oracle = _m(pos.actual_rec_yards, pos.r12_target_actual_ypt_oracle) if len(pos) else {"mae":np.nan,"bias":np.nan,"p90":np.nan}
            base_target_oracle = _m(pos.actual_rec_yards, pos.baseline_target_actual_ypt_oracle) if len(pos) else {"mae":np.nan}
            actual_ypt = pd.to_numeric(pos.actual_ypt, errors="coerce")
            frozen_ypt = pd.to_numeric(pos.frozen_ypt, errors="coerce")
            ok = actual_ypt.notna() & frozen_ypt.notna()
            corr = float(np.corrcoef(actual_ypt[ok], frozen_ypt[ok])[0,1]) if ok.sum() > 1 and actual_ypt[ok].std() > 0 and frozen_ypt[ok].std() > 0 else np.nan
            oracle_rows.append({
                "population":pop,"target_state":state,"n":len(g),"positive_target_rows":len(pos),
                "r12_rec_mae":r12["mae"],
                "actual_targets_frozen_ypt_mae_efficiency_floor":eff["mae"],
                "r12_targets_actual_ypt_mae_target_floor_positive_targets":target_oracle["mae"],
                "baseline_targets_actual_ypt_mae_target_floor_positive_targets":base_target_oracle["mae"],
                "mean_frozen_ypt":float(frozen_ypt[ok].mean()) if ok.any() else np.nan,
                "mean_actual_ypt":float(actual_ypt[ok].mean()) if ok.any() else np.nan,
                "ypt_bias_frozen_minus_actual":float((frozen_ypt[ok]-actual_ypt[ok]).mean()) if ok.any() else np.nan,
                "frozen_vs_actual_ypt_corr":corr,
            })
    oracle = pd.DataFrame(oracle_rows)

    # Team-game spillover audit: one row per team-game.
    tg = x.groupby(["season","week","team"], as_index=False).agg(
        baseline_pool=("baseline_team_rb_targets","first"), preconserve_pool=("r12_team_preconserve","first"),
        team_scale=("team_scale","first"), max_abs_pool_gap=("team_pool_gap",lambda s: float(pd.to_numeric(s,errors="coerce").abs().max())),
        mean_abs_player_target_shift=("conservation_target_shift",lambda s: float(pd.to_numeric(s,errors="coerce").abs().mean())),
        max_abs_player_target_shift=("conservation_target_shift",lambda s: float(pd.to_numeric(s,errors="coerce").abs().max())),
    )
    tg["preconserve_pool_gap"] = tg.preconserve_pool - tg.baseline_pool

    def srow(pop, stage):
        return summary.loc[summary.population.eq(pop)&summary.stage.eq(stage)].iloc[0]
    top_base, top_post = srow("TOP20","BASELINE"), srow("TOP20","R12_POSTCONSERVE")
    top_pre = srow("TOP20","R12_PRECONSERVE")
    rest_r9, rest_pre, rest_post = srow("REST80","R9"), srow("REST80","R12_PRECONSERVE"), srow("REST80","R12_POSTCONSERVE")
    top_oracle = oracle.loc[(oracle.population.eq("TOP20"))&(oracle.target_state.eq("ALL_STATES"))].iloc[0]

    result = {
        "diagnostic":"RB_R12_SPILLOVER_EFFICIENCY_V1",
        "disposition":"DIAGNOSTIC_ONLY_R12_REMAINS_FAIL",
        "r12_status":"RB_R12_STATE_GATED_MODERN_STABILITY_FAIL_DIAGNOSTIC_ONLY_UNCHANGED",
        "top20_preconserve_target_mae":float(top_pre.target_mae),
        "top20_postconserve_target_mae":float(top_post.target_mae),
        "top20_baseline_target_mae":float(top_base.target_mae),
        "top20_preconserve_rec_mae":float(top_pre.rec_mae),
        "top20_postconserve_rec_mae":float(top_post.rec_mae),
        "top20_baseline_rec_mae":float(top_base.rec_mae),
        "rest80_r9_target_mae":float(rest_r9.target_mae),
        "rest80_preconserve_target_mae":float(rest_pre.target_mae),
        "rest80_postconserve_target_mae":float(rest_post.target_mae),
        "rest80_conservation_target_mae_damage_vs_pre":float(rest_post.target_mae-rest_pre.target_mae),
        "rest80_conservation_rec_mae_damage_vs_pre":float(rest_post.rec_mae-rest_pre.rec_mae),
        "top20_actual_targets_frozen_ypt_efficiency_floor_mae":float(top_oracle.actual_targets_frozen_ypt_mae_efficiency_floor),
        "top20_r12_targets_actual_ypt_target_floor_mae_positive_targets":float(top_oracle.r12_targets_actual_ypt_mae_target_floor_positive_targets),
        "top20_frozen_vs_actual_ypt_corr":float(top_oracle.frozen_vs_actual_ypt_corr) if pd.notna(top_oracle.frozen_vs_actual_ypt_corr) else None,
        "top20_mean_frozen_ypt":float(top_oracle.mean_frozen_ypt),
        "top20_mean_actual_ypt":float(top_oracle.mean_actual_ypt),
        "team_scale_mean":float(tg.team_scale.mean()),
        "team_scale_p10":float(tg.team_scale.quantile(.10)),
        "team_scale_p90":float(tg.team_scale.quantile(.90)),
        "max_abs_conserved_pool_gap":float(tg.max_abs_pool_gap.max()),
        "sportsbook_inputs_added":0,
        "model_parameters_changed":0,
        "oracle_note":"actual targets and actual YPT are postgame localization only and cannot be deployable features",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir/"rb_r12_spillover_efficiency_casebook.csv",index=False)
    summary.to_csv(a.out_dir/"rb_r12_spillover_stage_summary.csv",index=False)
    oracle.to_csv(a.out_dir/"rb_r12_efficiency_oracle_summary.csv",index=False)
    tg.to_csv(a.out_dir/"rb_r12_team_conservation_audit.csv",index=False)
    (a.out_dir/"rb_r12_spillover_efficiency_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    print("\n=== stage summary ===\n",summary.to_string(index=False))
    print("\n=== oracle summary ===\n",oracle.to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
