#!/usr/bin/env python3
"""R27 V1: translate exact frozen R26 opportunity into RB receiving-yard means.

This evaluator does not fit a new efficiency model.  It consumes the exact R26
player-level fold output, rebuilds the same pregame production context, and asks
whether replacing baseline RB target allocation with R26 target allocation while
holding production YPT fixed improves receiving-yard point means.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import component_predictions as cp
from scripts.backtest import evaluate_rb_r26_vacancy_gated_r9_v1 as r26

BASE = "baseline"
CAND = "candidate"
RB_POS = {"RB", "FB", "HB", "TB"}


def _num(v, default=np.nan) -> float:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) and np.isfinite(float(x)) else float(default)


def _metric(actual: pd.Series, pred: pd.Series, market: str) -> dict[str, float | int]:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "median_abs_error": np.nan, "p75_abs_error": np.nan,
            "p90_abs_error": np.nan, "pearson": np.nan, "spearman": np.nan,
            "large_error_30_rate": np.nan,
        }
    err = p - a
    ae = np.abs(err)
    pearson = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0 else np.nan
    spearman = float(pd.Series(a).corr(pd.Series(p), method="spearman")) if len(a) > 1 else np.nan
    return {
        "n": int(len(a)),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "median_abs_error": float(np.quantile(ae, 0.50)),
        "p75_abs_error": float(np.quantile(ae, 0.75)),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
        "pearson": pearson,
        "spearman": spearman,
        "large_error_30_rate": float(np.mean(ae >= 30.0)) if market == "rec_yards" else np.nan,
    }


def _efficiency_frame(test_dir: Path, season: int, week: int, inputs: dict) -> pd.DataFrame:
    base = r26.build_base(test_dir, inputs, season, week).copy()
    pos = base.get("position", pd.Series("", index=base.index)).fillna("").astype(str).str.upper().str.strip()
    base = base.loc[pos.isin(RB_POS)].copy()
    if base.empty:
        raise RuntimeError(f"R27 no RB/FB production context for {season} W{week}")
    if "player_clean_key" not in base.columns:
        base["player_clean_key"] = base.get("player", "").map(r26.name_key)
    base["player_clean_key"] = base["player_clean_key"].fillna("").astype(str)
    base["event_id"] = base["event_id"].astype(str)
    base["team"] = base["team"].astype(str)

    # Match the frozen R26/production fallback hierarchy exactly.
    prior_catch = r26._catch_prior(inputs["logs"], season, week)
    ypt = []
    catch = []
    for _, row in base.iterrows():
        cr = r26.finite(row.get("rules_catch_rate"), r26.finite(row.get("bayes_receptions_per_target"), prior_catch))
        cr = float(np.clip(cr, 0.35, 0.95))
        yy = max(r26.finite(row.get("rules_ypt"), r26.finite(row.get("bayes_ypt"), 0.0)), 0.0)
        catch.append(cr)
        ypt.append(yy)
    base["production_catch_rate"] = catch
    base["production_ypt"] = ypt
    keep = ["event_id", "team", "player_clean_key", "production_catch_rate", "production_ypt"]
    out = base[keep].drop_duplicates(["event_id", "team", "player_clean_key"])
    if out.duplicated(["event_id", "team", "player_clean_key"]).any():
        raise RuntimeError(f"R27 duplicate efficiency identities for {season} W{week}")
    return out


def _actual_rec_yards(inputs: dict, season: int, week: int) -> pd.DataFrame:
    actual = cp.build_actual_rows(inputs["logs"], season, week)
    if actual.empty:
        return pd.DataFrame(columns=["team", "player_clean_key", "actual_rec_yards"])
    x = actual.loc[actual["market"].astype(str).eq("rec_yards"), ["team", "player_clean_key", "actual"]].copy()
    x.rename(columns={"actual": "actual_rec_yards"}, inplace=True)
    x["team"] = x["team"].astype(str)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    return x.drop_duplicates(["team", "player_clean_key"])


def _cohorts(pred: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ALL": pd.Series(True, index=pred.index),
        "VACANCY_ACTIVE": pred["vacancy_active"].eq(1),
        "VACANCY_INCUMBENT": pred["vacancy_incumbent"].eq(1),
        "VACANCY_RB1_INCUMBENT": pred["vacancy_incumbent"].eq(1) & pred["role"].eq("RB1"),
        "VACANCY_RB2PLUS_INCUMBENT": pred["vacancy_incumbent"].eq(1) & pred["role"].eq("RB2+"),
        "VACANCY_NEW_VETERAN": pred["vacancy_new_veteran"].eq(1),
        "VACANCY_NO_PRIOR_NFL": pred["vacancy_no_prior_nfl"].eq(1),
        "WEEK1": pred["week"].eq(1),
        "WEEKS2PLUS": pred["week"].ge(2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--r26-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    r26_pred_path = args.r26_dir / "r26_predictions.csv"
    r26_audit_path = args.r26_dir / "r26_structural_audit.csv"
    r26_fit_path = args.r26_dir / "r26_fit_metadata.json"
    for p in (r26_pred_path, r26_audit_path, r26_fit_path):
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"R27 missing exact R26 parent output: {p}")

    pred = pd.read_csv(r26_pred_path, low_memory=False)
    audit = pd.read_csv(r26_audit_path, low_memory=False)
    fit = json.loads(r26_fit_path.read_text(encoding="utf-8"))
    if int(args.test_season) != int(fit.get("test_season", -1)):
        raise RuntimeError("R27 R26 fold metadata/test season mismatch")
    if fit.get("candidate") != "RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1":
        raise RuntimeError("R27 received noncanonical R26 candidate output")
    if fit.get("strict_prior_fit") is not True:
        raise RuntimeError("R27 requires strict-prior R26 fit")
    if int(fit.get("sportsbook_inputs_used", 1)) != 0 or int(fit.get("future_outcomes_used_in_features", 1)) != 0:
        raise RuntimeError("R27 rejected R26 output with leakage flag")

    inputs = r26.load_bundle_inputs(args.test_dir)
    parts = []
    for week in r26.weeks(args.test_season):
        p = pred.loc[pd.to_numeric(pred["week"], errors="coerce").eq(week)].copy()
        if p.empty:
            raise RuntimeError(f"R27 missing R26 prediction rows for {args.test_season} W{week}")
        p["event_id"] = p["event_id"].astype(str)
        p["team"] = p["team"].astype(str)
        p["player_clean_key"] = p["player_clean_key"].astype(str)
        eff = _efficiency_frame(args.test_dir, args.test_season, week, inputs)
        p = p.merge(eff, on=["event_id", "team", "player_clean_key"], how="left", validate="one_to_one")
        if p[["production_ypt", "production_catch_rate"]].isna().any().any():
            bad = p.loc[p["production_ypt"].isna() | p["production_catch_rate"].isna(), ["team", "player", "player_clean_key"]]
            raise RuntimeError(f"R27 production efficiency join incomplete {args.test_season} W{week}: {bad.head(10).to_dict('records')}")
        actual = _actual_rec_yards(inputs, args.test_season, week)
        p = p.merge(actual, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        parts.append(p)

    out = pd.concat(parts, ignore_index=True)
    for c in ("baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions", "production_ypt", "production_catch_rate", "actual_targets", "actual_receptions", "actual_rec_yards"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Frozen V1 receiving-yard candidate: R26 opportunity, identical production efficiency.
    out["r27_baseline_rec_yards"] = out["baseline_targets"] * out["production_ypt"]
    out["r27_candidate_rec_yards"] = out["candidate_targets"] * out["production_ypt"]
    out["implied_production_ypr"] = np.where(out["production_catch_rate"].gt(0), out["production_ypt"] / out["production_catch_rate"], np.nan)
    out["r27_reception_bridge_rec_yards"] = out["candidate_receptions"] * out["implied_production_ypr"]
    out["r26_baseline_mean_parity_gap"] = out["r27_baseline_rec_yards"] - pd.to_numeric(out["baseline_rec_yards"], errors="coerce")
    out["reception_bridge_gap"] = out["r27_reception_bridge_rec_yards"] - out["r27_candidate_rec_yards"]

    # Diagnostics only: actual efficiency never enters either prediction.
    out["actual_ypt"] = np.where(out["actual_targets"].gt(0), out["actual_rec_yards"] / out["actual_targets"], np.nan)
    out["actual_ypr"] = np.where(out["actual_receptions"].gt(0), out["actual_rec_yards"] / out["actual_receptions"], np.nan)
    out["baseline_rec_yards_error"] = out["r27_baseline_rec_yards"] - out["actual_rec_yards"]
    out["candidate_rec_yards_error"] = out["r27_candidate_rec_yards"] - out["actual_rec_yards"]
    out["r26_target_delta"] = out["candidate_targets"] - out["baseline_targets"]
    out["r26_reception_delta"] = out["candidate_receptions"] - out["baseline_receptions"]
    out["r26_rec_yard_delta"] = out["r27_candidate_rec_yards"] - out["r27_baseline_rec_yards"]
    out["r26_moved_toward_actual_rec_yards"] = (
        out["candidate_rec_yards_error"].abs() < out["baseline_rec_yards_error"].abs()
    ).where(out["actual_rec_yards"].notna())

    mrows = []
    for cohort, mask in _cohorts(out).items():
        g = out.loc[mask].copy()
        for variant in (BASE, CAND):
            pred_cols = {
                "targets": f"{variant}_targets",
                "receptions": f"{variant}_receptions",
                "rec_yards": "r27_baseline_rec_yards" if variant == BASE else "r27_candidate_rec_yards",
            }
            actual_cols = {"targets": "actual_targets", "receptions": "actual_receptions", "rec_yards": "actual_rec_yards"}
            for market in ("targets", "receptions", "rec_yards"):
                rec = {
                    "season": int(args.test_season),
                    "cohort": cohort,
                    "variant": variant,
                    "market": market,
                }
                rec.update(_metric(g[actual_cols[market]], g[pred_cols[market]], market))
                mrows.append(rec)
    metrics = pd.DataFrame(mrows)

    max_room_gap = float(pd.to_numeric(audit.get("room_mass_gap"), errors="coerce").abs().max())
    max_nonrb = float(pd.to_numeric(audit.get("max_non_rb_entitlement_delta"), errors="coerce").abs().max())
    stable = out.loc[out["vacancy_active"].eq(0)]
    stable_target_gap = float((stable["candidate_targets"] - stable["baseline_targets"]).abs().max()) if len(stable) else 0.0
    bridge_gap = float(pd.to_numeric(out["reception_bridge_gap"], errors="coerce").abs().max())
    baseline_parity_gap = float(pd.to_numeric(out["r26_baseline_mean_parity_gap"], errors="coerce").abs().max())
    structural = {
        "season": int(args.test_season),
        "train_season": int(fit.get("train_season", args.test_season - 1)),
        "prediction_rows": int(len(out)),
        "vacancy_rows": int(out["vacancy_active"].eq(1).sum()),
        "strict_prior_fit": bool(fit.get("strict_prior_fit")),
        "sportsbook_inputs_upstream": int(max(int(fit.get("sportsbook_inputs_used", 0)), int(pd.to_numeric(audit.get("sportsbook_inputs_used"), errors="coerce").fillna(0).max()))),
        "future_outcomes_used_in_features": int(max(int(fit.get("future_outcomes_used_in_features", 0)), int(pd.to_numeric(audit.get("future_outcomes_used"), errors="coerce").fillna(0).max()))),
        "vacancy_gate_exact_room_exits_ge_1": True,
        "max_rb_room_mass_gap": max_room_gap,
        "max_non_rb_entitlement_delta": max_nonrb,
        "max_nonvacancy_target_delta": stable_target_gap,
        "max_reception_bridge_gap": bridge_gap,
        "max_r26_baseline_mean_parity_gap": baseline_parity_gap,
        "actual_efficiency_used_in_prediction": False,
        "r22_used_as_upstream_mean_correction": False,
        "production_files_changed": False,
        "r26_candidate": str(fit.get("candidate")),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "r27_player_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "r27_metrics_by_cohort.csv", index=False)
    out[[
        "train_season", "season", "week", "event_id", "team", "player", "player_clean_key", "role",
        "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl",
        "actual_targets", "actual_receptions", "actual_rec_yards", "baseline_targets", "candidate_targets",
        "baseline_receptions", "candidate_receptions", "production_catch_rate", "production_ypt",
        "actual_ypt", "actual_ypr", "r27_baseline_rec_yards", "r27_candidate_rec_yards",
        "r27_reception_bridge_rec_yards", "r26_target_delta", "r26_reception_delta", "r26_rec_yard_delta",
        "baseline_rec_yards_error", "candidate_rec_yards_error", "r26_moved_toward_actual_rec_yards",
    ]].to_csv(args.out_dir / "r27_opportunity_efficiency_decomposition.csv", index=False)
    pd.DataFrame([structural]).to_csv(args.out_dir / "r27_structural_audit.csv", index=False)
    (args.out_dir / "r27_fold_metadata.json").write_text(json.dumps(structural, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(structural, indent=2, sort_keys=True))
    print(metrics.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
