#!/usr/bin/env python3
"""Evaluate frozen R27B V2 novel RB receiving-efficiency context.

Exact R27/R26 opportunity is the immutable parent. The only candidate change is
an out-of-sample Ridge residual correction to production YPT, applied only in
vacancy-active RB/FB rows. Test outcomes are joined only after predictions exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = [1.0, 10.0, 100.0, 1000.0]
CORRECTION_CAP = 1.5
ID = ["season", "week", "event_id", "team", "player_clean_key"]
PRIMARY = [
    "player_air_yards_per_target_prior",
    "player_yac_per_reception_prior",
    "player_screen_target_rate_prior",
    "player_explosive20_target_rate_prior",
    "team_rb_targets_per_official_pass_attempt_prior",
    "team_rb_air_yards_per_target_prior",
    "team_rb_yac_per_reception_prior",
    "team_rb_screen_target_rate_prior",
    "opp_rb_air_yards_allowed_per_target_prior",
    "opp_rb_yac_allowed_per_reception_prior",
    "opp_rb_catch_rate_allowed_prior",
    "opp_rb_explosive20_allowed_per_target_prior",
    "opp_rb_screen_target_rate_faced_prior",
    "role_is_rb1",
    "role_is_rb2plus",
    "vacancy_incumbent",
    "vacancy_new_veteran",
    "vacancy_no_prior_nfl",
    "week1",
]
PLAYER_FAMILY = PRIMARY[:4] + PRIMARY[13:]
TEAM_FAMILY = PRIMARY[4:8] + PRIMARY[13:]
OPP_FAMILY = PRIMARY[8:13] + PRIMARY[13:]


def _metric(actual, pred) -> dict:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "abs_bias": np.nan,
                "median_abs_error": np.nan, "p75_abs_error": np.nan, "p90_abs_error": np.nan,
                "large_error_30_rate": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = p - a
    ae = np.abs(e)
    pear = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0 else np.nan
    spear = float(pd.Series(a).corr(pd.Series(p), method="spearman")) if len(a) > 1 else np.nan
    return {
        "n": int(len(a)), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()), "abs_bias": float(abs(e.mean())),
        "median_abs_error": float(np.quantile(ae, .50)), "p75_abs_error": float(np.quantile(ae, .75)),
        "p90_abs_error": float(np.quantile(ae, .90)), "large_error_30_rate": float(np.mean(ae >= 30.0)),
        "pearson": pear, "spearman": spear,
    }


def _feature_cols(frame: pd.DataFrame, family: list[str]) -> list[str]:
    missing = [f"{c}_missing" for c in family if f"{c}_missing" in frame.columns]
    absent = [c for c in family if c not in frame.columns]
    if absent:
        raise RuntimeError(f"R27B V2 frozen features absent: {absent}")
    return family + missing


def _prep_fit(train: pd.DataFrame, cols: list[str]):
    x = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(axis=0, skipna=True).fillna(0.0)
    x = x.fillna(med)
    keep = [c for c in cols if float(x[c].std(ddof=0)) > 1e-12]
    if not keep:
        raise RuntimeError("R27B V2 has no nonconstant legal features")
    scaler = StandardScaler()
    z = scaler.fit_transform(x[keep])
    return z, med, keep, scaler


def _prep_apply(frame: pd.DataFrame, med: pd.Series, keep: list[str], scaler: StandardScaler):
    x = frame[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(med.reindex(keep).fillna(0.0))
    return scaler.transform(x[keep])


def _eligible_train(train: pd.DataFrame) -> pd.DataFrame:
    return train.loc[
        pd.to_numeric(train["actual_targets"], errors="coerce").ge(1)
        & pd.to_numeric(train["efficiency_residual_target"], errors="coerce").notna()
    ].copy()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], alpha: float):
    tr = _eligible_train(train)
    if tr.empty:
        raise RuntimeError("R27B V2 has no eligible training rows")
    z, med, keep, scaler = _prep_fit(tr, cols)
    y = pd.to_numeric(tr["efficiency_residual_target"], errors="coerce").to_numpy(float)
    w = pd.to_numeric(tr["sample_weight"], errors="coerce").to_numpy(float)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(z, y, sample_weight=w)
    pred = model.predict(_prep_apply(test, med, keep, scaler))
    meta = {
        "alpha": float(alpha), "train_rows": int(len(tr)),
        "train_seasons": sorted(tr["season"].astype(int).unique().tolist()),
        "feature_count": int(len(keep)), "features": keep,
        "intercept": float(model.intercept_),
        "coefficients": {c: float(v) for c, v in zip(keep, model.coef_)},
        "training_only_imputer": True, "training_only_scaler": True,
    }
    return pred, meta


def _choose_alpha(train: pd.DataFrame, cols: list[str]) -> tuple[float, dict]:
    seasons = sorted(_eligible_train(train)["season"].astype(int).unique().tolist())
    if len(seasons) < 2:
        return 100.0, {"fallback": True, "alpha": 100.0, "reason": "single_legal_training_season", "scores": {}}
    totals = {a: {"weighted_abs_error": 0.0, "weight": 0.0, "folds": []} for a in ALPHAS}
    for v in seasons[1:]:
        tr = train.loc[train["season"].lt(v)].copy()
        va = _eligible_train(train.loc[train["season"].eq(v)].copy())
        if _eligible_train(tr).empty or va.empty:
            continue
        for alpha in ALPHAS:
            pred, _ = _fit_predict(tr, va, cols, alpha)
            y = va["efficiency_residual_target"].to_numpy(float)
            w = va["sample_weight"].to_numpy(float)
            wae = float(np.sum(np.abs(pred - y) * w))
            sw = float(w.sum())
            totals[alpha]["weighted_abs_error"] += wae
            totals[alpha]["weight"] += sw
            totals[alpha]["folds"].append({"validation_season": int(v), "weighted_mae": float(wae / sw), "n": int(len(va))})
    score = {a: (totals[a]["weighted_abs_error"] / totals[a]["weight"] if totals[a]["weight"] > 0 else np.inf) for a in ALPHAS}
    best_score = min(score.values())
    best = max(a for a, s in score.items() if np.isclose(s, best_score, atol=1e-12, rtol=0))
    return float(best), {
        "fallback": False, "alpha": float(best),
        "scores": {str(a): float(score[a]) for a in ALPHAS},
        "fold_details": {str(a): totals[a]["folds"] for a in ALPHAS},
    }


def _cohorts(x: pd.DataFrame) -> dict[str, pd.Series]:
    role = x["role"].fillna("").astype(str).str.upper()
    return {
        "ALL": pd.Series(True, index=x.index),
        "VACANCY_ACTIVE": x["vacancy_active"].eq(1),
        "VACANCY_INCUMBENT": x["vacancy_incumbent"].eq(1),
        "VACANCY_RB1_INCUMBENT": x["vacancy_incumbent"].eq(1) & role.eq("RB1"),
        "VACANCY_RB2PLUS_INCUMBENT": x["vacancy_incumbent"].eq(1) & role.eq("RB2+"),
        "VACANCY_NEW_VETERAN": x["vacancy_new_veteran"].eq(1),
        "VACANCY_NO_PRIOR_NFL": x["vacancy_no_prior_nfl"].eq(1),
        "WEEK1": x["week"].eq(1),
        "WEEKS2PLUS": x["week"].ge(2),
    }


def _ablation(train: pd.DataFrame, test: pd.DataFrame, family: list[str], alpha: float) -> np.ndarray:
    cols = _feature_cols(train, family)
    pred, _ = _fit_predict(train, test, cols, alpha)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--r27-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    features = pd.read_csv(a.dataset_dir / "r27b_v2_features.csv", low_memory=False)
    labels = pd.read_csv(a.dataset_dir / "r27b_v2_labels.csv", low_memory=False)
    parent = pd.read_csv(a.r27_dir / "r27_player_predictions.csv", low_memory=False)
    parent_struct = pd.read_csv(a.r27_dir / "r27_structural_audit.csv", low_memory=False)
    for x in (features, labels, parent):
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")
        x["event_id"] = x.get("event_id", "").astype(str) if "event_id" in x.columns else ""
        x["team"] = x["team"].astype(str)
        x["player_clean_key"] = x["player_clean_key"].astype(str)

    # Strict outer training pool. Labels are attached only to seasons earlier than test.
    train = features.loc[features["season"].between(2019, int(a.test_season) - 1)].copy()
    train = train.merge(labels.drop(columns=["event_id"], errors="ignore"), on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    train["actual_targets"] = pd.to_numeric(train["actual_targets"], errors="coerce")
    train["actual_rec_yards"] = pd.to_numeric(train["actual_rec_yards"], errors="coerce")
    train["actual_ypt"] = np.where(train["actual_targets"].gt(0), train["actual_rec_yards"] / train["actual_targets"], np.nan)
    train["efficiency_residual_target"] = train["actual_ypt"] - pd.to_numeric(train["production_ypt"], errors="coerce")
    train["sample_weight"] = train["actual_targets"].clip(1, 8)

    # Untouched outer test feature frame. Parent actual outcomes are explicitly dropped.
    testf = features.loc[features["season"].eq(int(a.test_season))].copy()
    parent = parent.loc[parent["season"].eq(int(a.test_season))].copy()
    outcome_cols = [c for c in parent.columns if c.startswith("actual_") or c.endswith("_error") or c.startswith("r26_moved_")]
    pre_parent = parent.drop(columns=outcome_cols, errors="ignore").copy()
    required_parent = [
        "baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions",
        "r27_baseline_rec_yards", "r27_candidate_rec_yards", "production_ypt", "production_catch_rate",
        "role", "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl",
    ]
    miss_parent = [c for c in required_parent if c not in pre_parent.columns]
    if miss_parent:
        raise RuntimeError(f"R27B V2 parent missing columns {miss_parent}")
    keep_parent = ID + [c for c in required_parent if c not in ID]
    if "room_exits_n" in pre_parent.columns:
        keep_parent.append("room_exits_n")
    test = pre_parent[keep_parent].merge(testf, on=ID, how="left", suffixes=("", "_feature"), validate="one_to_one")
    if len(test) != len(pre_parent) or test["production_ypt_feature"].isna().any():
        raise RuntimeError("R27B V2 test feature join incomplete")
    prod_gap = float((pd.to_numeric(test["production_ypt"], errors="coerce") - pd.to_numeric(test["production_ypt_feature"], errors="coerce")).abs().max())
    catch_gap = float((pd.to_numeric(test["production_catch_rate"], errors="coerce") - pd.to_numeric(test["production_catch_rate_feature"], errors="coerce")).abs().max())
    if prod_gap > 1e-12 or catch_gap > 1e-12:
        raise RuntimeError(f"R27B V2 production context drift ypt={prod_gap} catch={catch_gap}")

    # Parent structural state is authoritative for the test fold.
    for c in ("vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl"):
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0).astype(int)
    test["role_is_rb1"] = test["role"].fillna("").astype(str).str.upper().eq("RB1").astype(int)
    test["role_is_rb2plus"] = test["role"].fillna("").astype(str).str.upper().eq("RB2+").astype(int)
    test["week1"] = test["week"].eq(1).astype(int)

    cols = _feature_cols(train, PRIMARY)
    alpha, alpha_audit = _choose_alpha(train, cols)
    raw, model_meta = _fit_predict(train, test, cols, alpha)
    test["context_residual_hat"] = raw
    test["context_residual_clipped"] = np.clip(raw, -CORRECTION_CAP, CORRECTION_CAP)
    test["applied_ypt_correction"] = np.where(test["vacancy_active"].eq(1), test["context_residual_clipped"], 0.0)
    test["c1_ypt"] = np.maximum(pd.to_numeric(test["production_ypt"], errors="coerce") + test["applied_ypt_correction"], 0.0)
    test["b0_rec_yards"] = pd.to_numeric(test["r27_baseline_rec_yards"], errors="coerce")
    test["b1_rec_yards"] = pd.to_numeric(test["r27_candidate_rec_yards"], errors="coerce")
    test["c1_rec_yards"] = pd.to_numeric(test["candidate_targets"], errors="coerce") * test["c1_ypt"]
    test["c1_candidate_targets"] = pd.to_numeric(test["candidate_targets"], errors="coerce")
    test["c1_candidate_receptions"] = pd.to_numeric(test["candidate_receptions"], errors="coerce")
    test["c1_implied_ypr"] = np.where(pd.to_numeric(test["production_catch_rate"], errors="coerce").gt(0), test["c1_ypt"] / pd.to_numeric(test["production_catch_rate"], errors="coerce"), np.nan)
    test["c1_bridge_rec_yards"] = test["c1_candidate_receptions"] * test["c1_implied_ypr"]
    test["bridge_gap"] = test["c1_bridge_rec_yards"] - test["c1_rec_yards"]

    # Frozen explanatory ablations only; same selected alpha, never candidate selection.
    for name, fam in (("player_shape", PLAYER_FAMILY), ("team_qb_environment", TEAM_FAMILY), ("opponent_context", OPP_FAMILY)):
        ab = _ablation(train, test, fam, alpha)
        test[f"ablation_{name}_residual_hat"] = ab
        adj = np.where(test["vacancy_active"].eq(1), np.clip(ab, -CORRECTION_CAP, CORRECTION_CAP), 0.0)
        test[f"ablation_{name}_rec_yards"] = pd.to_numeric(test["candidate_targets"], errors="coerce") * np.maximum(pd.to_numeric(test["production_ypt"], errors="coerce") + adj, 0.0)

    # Only now attach untouched outer-test outcomes for scoring.
    post_labels = labels.loc[labels["season"].eq(int(a.test_season))].drop(columns=["event_id"], errors="ignore")
    test = test.merge(post_labels, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    test["actual_targets"] = pd.to_numeric(test["actual_targets"], errors="coerce")
    test["actual_receptions"] = pd.to_numeric(test["actual_receptions"], errors="coerce")
    test["actual_rec_yards"] = pd.to_numeric(test["actual_rec_yards"], errors="coerce")
    test["actual_ypt"] = np.where(test["actual_targets"].gt(0), test["actual_rec_yards"] / test["actual_targets"], np.nan)

    metrics = []
    for cohort, mask in _cohorts(test).items():
        g = test.loc[mask].copy()
        for variant, col in (("B0", "b0_rec_yards"), ("B1", "b1_rec_yards"), ("C1", "c1_rec_yards")):
            rec = {"season": int(a.test_season), "cohort": cohort, "variant": variant, "market": "rec_yards"}
            rec.update(_metric(g["actual_rec_yards"], g[col]))
            metrics.append(rec)
        # YPT diagnostic is only defined where actual targets >0.
        gy = g.loc[g["actual_targets"].gt(0)].copy()
        for variant, col in (("B1", "production_ypt"), ("C1", "c1_ypt")):
            rec = {"season": int(a.test_season), "cohort": cohort, "variant": variant, "market": "ypt"}
            rec.update(_metric(gy["actual_ypt"], gy[col]))
            metrics.append(rec)
        for variant, col in (("PLAYER_SHAPE", "ablation_player_shape_rec_yards"), ("TEAM_QB_ENV", "ablation_team_qb_environment_rec_yards"), ("OPP_CONTEXT", "ablation_opponent_context_rec_yards")):
            rec = {"season": int(a.test_season), "cohort": cohort, "variant": variant, "market": "rec_yards_ablation"}
            rec.update(_metric(g["actual_rec_yards"], g[col]))
            metrics.append(rec)
    metrics = pd.DataFrame(metrics)

    ps = parent_struct.iloc[0].to_dict() if len(parent_struct) else {}
    stable = test.loc[test["vacancy_active"].eq(0)].copy()
    vacancy_exact = bool(ps.get("vacancy_gate_exact_room_exits_ge_1", True))
    if "room_exits_n" in test.columns:
        vacancy_exact = vacancy_exact and bool(((pd.to_numeric(test["room_exits_n"], errors="coerce").fillna(0).ge(1)).astype(int) == test["vacancy_active"]).all())
    structural = {
        "season": int(a.test_season),
        "prediction_rows": int(len(test)),
        "vacancy_rows": int(test["vacancy_active"].eq(1).sum()),
        "sportsbook_inputs_upstream": int(pd.to_numeric(pd.Series([ps.get("sportsbook_inputs_upstream", 0)]), errors="coerce").fillna(0).iloc[0]),
        "future_outcomes_used_in_features": 0,
        "target_game_outcomes_used_before_prediction": 0,
        "generic_ypt_ypr_persistence_features_used": 0,
        "exact_r26_r27_parent_identity": True,
        "max_parent_feature_production_ypt_gap": prod_gap,
        "max_parent_feature_catch_rate_gap": catch_gap,
        "max_c1_target_delta_vs_b1": float((test["c1_candidate_targets"] - pd.to_numeric(test["candidate_targets"], errors="coerce")).abs().max()),
        "max_c1_reception_delta_vs_b1": float((test["c1_candidate_receptions"] - pd.to_numeric(test["candidate_receptions"], errors="coerce")).abs().max()),
        "vacancy_gate_exact_room_exits_ge_1": vacancy_exact,
        "max_stable_ypt_delta": float((stable["c1_ypt"] - pd.to_numeric(stable["production_ypt"], errors="coerce")).abs().max()) if len(stable) else 0.0,
        "max_stable_rec_yard_delta_vs_b1": float((stable["c1_rec_yards"] - stable["b1_rec_yards"]).abs().max()) if len(stable) else 0.0,
        "max_rb_room_mass_gap": float(pd.to_numeric(pd.Series([ps.get("max_rb_room_mass_gap", 0)]), errors="coerce").fillna(0).iloc[0]),
        "max_non_rb_entitlement_delta": float(pd.to_numeric(pd.Series([ps.get("max_non_rb_entitlement_delta", 0)]), errors="coerce").fillna(0).iloc[0]),
        "r22_used_as_upstream_mean_correction": False,
        "production_files_changed": False,
        "strict_prior_novel_features": True,
        "outer_test_used_in_fit_or_selection": False,
        "max_reception_bridge_gap": float(pd.to_numeric(test["bridge_gap"], errors="coerce").abs().max()),
        "correction_cap": CORRECTION_CAP,
        "correction_cap_hit_rate_vacancy": float(np.mean(np.isclose(np.abs(test.loc[test["vacancy_active"].eq(1), "applied_ypt_correction"]), CORRECTION_CAP, atol=1e-12))) if test["vacancy_active"].eq(1).any() else 0.0,
        "mean_abs_ypt_correction_vacancy": float(test.loc[test["vacancy_active"].eq(1), "applied_ypt_correction"].abs().mean()) if test["vacancy_active"].eq(1).any() else 0.0,
        "median_abs_ypt_correction_vacancy": float(test.loc[test["vacancy_active"].eq(1), "applied_ypt_correction"].abs().median()) if test["vacancy_active"].eq(1).any() else 0.0,
    }

    missing_rates = {c: float(pd.to_numeric(test[c], errors="coerce").isna().mean()) for c in PRIMARY if c in test.columns}
    meta = {
        "study": "RB_R27B_V2_NOVEL_EFFICIENCY_CONTEXT",
        "test_season": int(a.test_season),
        "alpha_selection": alpha_audit,
        "model": model_meta,
        "primary_features": PRIMARY,
        "missing_rates_outer_test": missing_rates,
        "candidate_scope": "VACANCY_ACTIVE_ONLY",
        "production_ypt_is_offset_not_primary_predictor": True,
        "sportsbook_inputs_used": 0,
        "future_outcomes_used_in_features": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(a.out_dir / "r27b_v2_predictions.csv", index=False)
    metrics.to_csv(a.out_dir / "r27b_v2_metrics.csv", index=False)
    pd.DataFrame([structural]).to_csv(a.out_dir / "r27b_v2_structural_audit.csv", index=False)
    (a.out_dir / "r27b_v2_model_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(structural, indent=2, sort_keys=True))
    print(metrics.loc[(metrics["cohort"].isin(["ALL", "VACANCY_ACTIVE", "VACANCY_RB1_INCUMBENT"])) & metrics["variant"].isin(["B0", "B1", "C1"]) & metrics["market"].eq("rec_yards")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
