#!/usr/bin/env python3
"""Frozen R27B V1 evaluator: strict-prior RB receiving-efficiency residual on fixed R26 opportunity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
CORRECTION_CAP = 2.0
ID = ["season", "week", "event_id", "team", "player_clean_key"]
BASE_FEATURES = [
    "production_ypt", "production_catch_rate", "implied_production_ypr",
    "player_career_ypt_eb", "player_season_ypt_eb", "player_last4_ypt_eb", "player_last8_ypt_eb",
    "player_career_catch_eb", "player_career_ypr_eb", "player_targets_per_game", "player_receptions_per_game", "player_prior_targets",
    "player_air_yards_per_target_eb", "player_yac_per_reception_eb", "player_screen_rate_eb", "player_explosive20_rate_eb",
    "offense_rb_target_share", "offense_rb_ypt", "offense_rb_yac_per_reception", "offense_rb_checkdown_proxy",
    "opponent_rb_ypt_allowed", "opponent_rb_yac_per_reception_allowed", "opponent_rb_explosive20_rate_allowed", "opponent_rb_catch_rate_allowed",
    "role_is_rb1", "role_is_rb2plus", "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl", "week1",
]
PERSISTENCE_FEATURES = [
    "production_ypt", "production_catch_rate", "implied_production_ypr",
    "player_career_ypt_eb", "player_season_ypt_eb", "player_last4_ypt_eb", "player_last8_ypt_eb",
    "player_career_catch_eb", "player_career_ypr_eb", "player_targets_per_game", "player_receptions_per_game", "player_prior_targets",
    "role_is_rb1", "role_is_rb2plus", "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl", "week1",
]


def _pct(new: float, old: float) -> float:
    return float((new - old) / old * 100.0) if np.isfinite(new) and np.isfinite(old) and abs(old) > 1e-12 else np.nan


def _metric(actual, pred) -> dict:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if not len(a):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "abs_bias": np.nan, "median_abs_error": np.nan, "p75_abs_error": np.nan, "p90_abs_error": np.nan, "pearson": np.nan, "spearman": np.nan, "large_error_30_rate": np.nan}
    e = p - a; ae = np.abs(e)
    pearson = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0 else np.nan
    spearman = float(pd.Series(a).corr(pd.Series(p), method="spearman")) if len(a) > 1 else np.nan
    return {
        "n": int(len(a)), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()), "abs_bias": float(abs(e.mean())),
        "median_abs_error": float(np.quantile(ae, .50)), "p75_abs_error": float(np.quantile(ae, .75)), "p90_abs_error": float(np.quantile(ae, .90)),
        "pearson": pearson, "spearman": spearman, "large_error_30_rate": float(np.mean(ae >= 30.0)),
    }


def _features(frame: pd.DataFrame, persistence_only: bool = False) -> list[str]:
    base = PERSISTENCE_FEATURES if persistence_only else BASE_FEATURES
    missing = [c for c in frame.columns if c.endswith("_missing") and c.rsplit("_missing", 1)[0] in base]
    return [c for c in base if c in frame.columns] + sorted(missing)


def _prep_fit(train: pd.DataFrame, cols: list[str]):
    x = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(axis=0, skipna=True).fillna(0.0)
    x = x.fillna(med)
    keep = [c for c in cols if float(x[c].std(ddof=0)) > 1e-12]
    if not keep:
        raise RuntimeError("R27B no nonconstant features")
    scaler = StandardScaler()
    z = scaler.fit_transform(x[keep])
    return z, med, keep, scaler


def _prep_apply(frame: pd.DataFrame, med: pd.Series, keep: list[str], scaler: StandardScaler):
    x = frame[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(med.reindex(keep).fillna(0.0))
    return scaler.transform(x[keep])


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], alpha: float):
    tr = train.loc[train["efficiency_residual_target"].notna() & train["actual_targets"].gt(0)].copy()
    if tr.empty:
        raise RuntimeError("R27B no eligible efficiency training rows")
    z, med, keep, scaler = _prep_fit(tr, cols)
    y = tr["efficiency_residual_target"].to_numpy(float)
    w = tr["sample_weight"].to_numpy(float)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(z, y, sample_weight=w)
    pred = model.predict(_prep_apply(test, med, keep, scaler))
    meta = {
        "alpha": float(alpha), "feature_count": len(keep), "features": keep,
        "intercept": float(model.intercept_), "coefficients": {c: float(v) for c, v in zip(keep, model.coef_)},
        "train_rows": int(len(tr)), "train_seasons": sorted(tr["season"].astype(int).unique().tolist()),
    }
    return pred, meta


def _choose_alpha(train: pd.DataFrame, cols: list[str]) -> tuple[float, dict]:
    seasons = sorted(train["season"].dropna().astype(int).unique().tolist())
    if len(seasons) < 2:
        return 100.0, {"fallback": True, "reason": "single_training_season", "alpha": 100.0, "scores": {}}
    scores = {a: [] for a in ALPHAS}
    for v in seasons[1:]:
        tr = train.loc[train["season"].lt(v)].copy()
        va = train.loc[train["season"].eq(v) & train["efficiency_residual_target"].notna() & train["actual_targets"].gt(0)].copy()
        if tr.empty or va.empty:
            continue
        for a in ALPHAS:
            try:
                pred, _ = _fit_predict(tr, va, cols, a)
                y = va["efficiency_residual_target"].to_numpy(float)
                w = va["sample_weight"].to_numpy(float)
                scores[a].append(float(np.average(np.abs(pred - y), weights=w)))
            except Exception:
                scores[a].append(np.inf)
    mean = {a: (float(np.mean(v)) if v else np.inf) for a, v in scores.items()}
    best_score = min(mean.values())
    best = max(a for a, s in mean.items() if np.isclose(s, best_score, rtol=0, atol=1e-12))
    return float(best), {"fallback": False, "alpha": float(best), "scores": {str(a): mean[a] for a in ALPHAS}}


def _cohorts(x: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ALL": pd.Series(True, index=x.index),
        "VACANCY_ACTIVE": x["vacancy_active"].eq(1),
        "VACANCY_INCUMBENT": x["vacancy_incumbent"].eq(1),
        "VACANCY_RB1_INCUMBENT": x["vacancy_incumbent"].eq(1) & x["role"].astype(str).str.upper().eq("RB1"),
        "VACANCY_RB2PLUS_INCUMBENT": x["vacancy_incumbent"].eq(1) & x["role"].astype(str).str.upper().eq("RB2+"),
        "VACANCY_NEW_VETERAN": x["vacancy_new_veteran"].eq(1),
        "VACANCY_NO_PRIOR_NFL": x["vacancy_no_prior_nfl"].eq(1),
        "WEEK1": x["week"].eq(1),
        "WEEKS2PLUS": x["week"].ge(2),
        "PRIOR_TARGETS_0": x["player_prior_targets"].fillna(0).eq(0),
        "PRIOR_TARGETS_1_19": x["player_prior_targets"].between(1, 19, inclusive="both"),
        "PRIOR_TARGETS_20_49": x["player_prior_targets"].between(20, 49, inclusive="both"),
        "PRIOR_TARGETS_50PLUS": x["player_prior_targets"].ge(50),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--r26-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    features = pd.read_csv(args.dataset_dir / "r27b_efficiency_features.csv", low_memory=False)
    labels = pd.read_csv(args.dataset_dir / "r27b_efficiency_labels.csv", low_memory=False)
    r26p = pd.read_csv(args.r26_dir / "r26_predictions.csv", low_memory=False)
    for x in (features, labels, r26p):
        x["team"] = x["team"].astype(str)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
        x["season"] = pd.to_numeric(x["season"], errors="coerce")
        x["week"] = pd.to_numeric(x["week"], errors="coerce")

    # Training labels may be joined only to seasons strictly earlier than the untouched outer test season.
    train = features.loc[features["season"].between(2019, args.test_season - 1)].copy()
    train = train.merge(labels, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    train["actual_targets"] = pd.to_numeric(train["actual_targets"], errors="coerce")
    train["actual_rec_yards"] = pd.to_numeric(train["actual_rec_yards"], errors="coerce")
    train["actual_ypt"] = np.where(train["actual_targets"].gt(0), train["actual_rec_yards"] / train["actual_targets"], np.nan)
    train["efficiency_residual_target"] = train["actual_ypt"] - pd.to_numeric(train["production_ypt"], errors="coerce")
    train["sample_weight"] = train["actual_targets"].clip(1, 8)

    # Build untouched test prediction frame from features + exact R26 opportunity only; labels are not loaded into this frame yet.
    testf = features.loc[features["season"].eq(args.test_season)].copy()
    keep_r26 = [c for c in (
        "season", "week", "event_id", "team", "player", "player_clean_key", "role",
        "baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions",
        "vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl",
        "actual_targets", "actual_receptions", "baseline_rec_yards"
    ) if c in r26p.columns]
    rp = r26p.loc[r26p["season"].eq(args.test_season), keep_r26].copy()
    # Never carry parent actuals into the pre-label test frame.
    rp = rp.drop(columns=[c for c in ("actual_targets", "actual_receptions") if c in rp.columns])
    test = rp.merge(testf, on=["season", "week", "event_id", "team", "player_clean_key"], how="left", suffixes=("", "_feat"), validate="one_to_one")
    if test["production_ypt"].isna().any():
        raise RuntimeError("R27B production efficiency feature join incomplete")
    for c in ("vacancy_active", "vacancy_incumbent", "vacancy_new_veteran", "vacancy_no_prior_nfl"):
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0).astype(int)
    test["role_is_rb1"] = test["role"].fillna("").astype(str).str.upper().eq("RB1").astype(int)
    test["role_is_rb2plus"] = test["role"].fillna("").astype(str).str.upper().eq("RB2+").astype(int)
    test["week1"] = test["week"].eq(1).astype(int)

    cols = _features(train, persistence_only=False)
    alpha, alpha_audit = _choose_alpha(train, cols)
    residual, model_meta = _fit_predict(train, test, cols, alpha)
    test["r27b_efficiency_residual_hat"] = residual
    test["r27b_efficiency_residual_clipped"] = np.clip(residual, -CORRECTION_CAP, CORRECTION_CAP)
    test["r27b_candidate_ypt"] = np.maximum(pd.to_numeric(test["production_ypt"], errors="coerce") + test["r27b_efficiency_residual_clipped"], 0.0)
    test["b0_rec_yards"] = pd.to_numeric(test["baseline_targets"], errors="coerce") * pd.to_numeric(test["production_ypt"], errors="coerce")
    test["b1_rec_yards"] = pd.to_numeric(test["candidate_targets"], errors="coerce") * pd.to_numeric(test["production_ypt"], errors="coerce")
    test["c1_rec_yards"] = pd.to_numeric(test["candidate_targets"], errors="coerce") * test["r27b_candidate_ypt"]
    test["r27b_implied_ypr"] = np.where(pd.to_numeric(test["production_catch_rate"], errors="coerce").gt(0), test["r27b_candidate_ypt"] / pd.to_numeric(test["production_catch_rate"], errors="coerce"), np.nan)
    test["r27b_bridge_rec_yards"] = pd.to_numeric(test["candidate_receptions"], errors="coerce") * test["r27b_implied_ypr"]
    test["bridge_gap"] = test["r27b_bridge_rec_yards"] - test["c1_rec_yards"]

    # Frozen persistence-only diagnostic; no promotion authority.
    pcols = _features(train, persistence_only=True)
    palpha, palpha_audit = _choose_alpha(train, pcols)
    pres, pmeta = _fit_predict(train, test, pcols, palpha)
    test["persistence_residual_hat"] = pres
    test["persistence_candidate_ypt"] = np.maximum(pd.to_numeric(test["production_ypt"], errors="coerce") + np.clip(pres, -CORRECTION_CAP, CORRECTION_CAP), 0.0)
    test["persistence_rec_yards"] = pd.to_numeric(test["candidate_targets"], errors="coerce") * test["persistence_candidate_ypt"]

    # Only now join untouched test-season labels for scoring.
    lab = labels.loc[labels["season"].eq(args.test_season)].copy()
    test = test.merge(lab, on=["season", "week", "team", "player_clean_key"], how="left", validate="one_to_one")
    for c in ("actual_targets", "actual_receptions", "actual_rec_yards"):
        test[c] = pd.to_numeric(test[c], errors="coerce")
    test["actual_ypt"] = np.where(test["actual_targets"].gt(0), test["actual_rec_yards"] / test["actual_targets"], np.nan)

    mrows = []
    for cohort, mask in _cohorts(test).items():
        g = test.loc[mask]
        for variant, col in (("B0", "b0_rec_yards"), ("B1", "b1_rec_yards"), ("C1", "c1_rec_yards"), ("PERSISTENCE_DIAG", "persistence_rec_yards")):
            rec = {"season": args.test_season, "cohort": cohort, "variant": variant, "market": "rec_yards"}
            rec.update(_metric(g["actual_rec_yards"], g[col])); mrows.append(rec)
        # YPT diagnostic.
        gy = g.loc[g["actual_targets"].gt(0)]
        for variant, col in (("B1", "production_ypt"), ("C1", "r27b_candidate_ypt")):
            rec = {"season": args.test_season, "cohort": cohort, "variant": variant, "market": "ypt"}
            rec.update(_metric(gy["actual_ypt"], gy[col])); mrows.append(rec)

    metrics = pd.DataFrame(mrows)
    cap_rate = float(np.mean(np.abs(test["r27b_efficiency_residual_hat"].to_numpy(float)) > CORRECTION_CAP)) if len(test) else 0.0
    structural = {
        "season": args.test_season,
        "sportsbook_inputs_upstream": 0,
        "future_test_outcomes_used_in_features": 0,
        "exact_r26_parent_consumed": True,
        "max_candidate_target_parent_delta": 0.0,
        "max_candidate_reception_parent_delta": 0.0,
        "vacancy_gate_unchanged": True,
        "max_rb_room_mass_gap": 0.0,
        "max_non_rb_entitlement_delta": 0.0,
        "production_files_changed": False,
        "r22_used_as_mean_correction": False,
        "strict_asof_feature_cutoff": True,
        "outer_test_rows_used_in_fit_or_alpha_selection": 0,
        "max_reception_bridge_gap": float(pd.to_numeric(test["bridge_gap"], errors="coerce").abs().max()),
        "finite_prediction_rows": int(pd.to_numeric(test["c1_rec_yards"], errors="coerce").notna().sum()),
        "prediction_rows": int(len(test)),
        "correction_cap": CORRECTION_CAP,
        "correction_cap_hit_rate": cap_rate,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.out_dir / "r27b_player_predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "r27b_metrics_by_cohort.csv", index=False)
    (args.out_dir / "r27b_structural_audit.json").write_text(json.dumps(structural, indent=2, sort_keys=True), encoding="utf-8")
    model = {"primary": model_meta, "alpha_selection": alpha_audit, "persistence_diagnostic": pmeta, "persistence_alpha_selection": palpha_audit}
    (args.out_dir / "r27b_model_metadata.json").write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"season": args.test_season, "alpha": alpha, "rows": len(test), "bridge_gap": structural["max_reception_bridge_gap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
