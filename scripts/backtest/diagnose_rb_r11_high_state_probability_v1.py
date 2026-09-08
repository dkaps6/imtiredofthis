#!/usr/bin/env python3
"""RB-R11 pregame high-receiving-state probability diagnostic.

This is a diagnostic bridge between the R10 state forensic and any future mixture
candidate. It does not alter R9/R10 projections or promote a model.

The R10 forensic showed a sharp outcome-state split among top-20 receiving-identity
backs: R9 hurt ordinary 0-4 target games but materially improved 5+ and especially
7+ target games. Before building a mixture model, this script asks whether the high
state itself is distinguishable before kickoff from frozen strict-prior signals.

Modern walk-forward folds (not pristine confirmation):
  train 2022 -> test 2023
  train 2022-2023 -> test 2024
  train 2022-2024 -> test 2025

Only TOP20 strict-prior receiving-identity rows are modeled. Labels are postgame and
used only as targets. Features are all pregame outputs already present in R10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R9_IDENTITY_SHRINKAGE"
KEYS = ["season", "week", "team", "player_clean_key"]
FEATURES = [
    "baseline_pred_targets",
    "prior_rb_room_share",
    "target_delta",
    "r9_raw_r8_residual",
]
FOLDS = [((2022,), 2023), ((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]

# Frozen before executing this diagnostic. These are diagnostic-support thresholds,
# not promotion gates and not claims of pristine scientific confirmation.
MIN_COMBINED_AUC = 0.65
MIN_FOLD_AUC = 0.58
MIN_FOLDS_AUC = 2
MIN_FOLDS_BRIER_BEAT_BASE = 2
MIN_TOP_QUINTILE_LIFT = 1.50
MIN_TOP_QUINTILE_CAPTURE = 0.30


def _prepare(pred: pd.DataFrame) -> pd.DataFrame:
    req = set(KEYS + ["variant", "actual_targets", "pred_targets", "prior_rb_room_share", "r9_raw_r8_residual"])
    missing = sorted(req - set(pred.columns))
    if missing:
        raise RuntimeError(f"R10 predictions missing columns: {missing}")
    b = pred.loc[pred.variant.eq(BASE), KEYS + ["actual_targets", "pred_targets", "prior_rb_room_share", "r9_raw_r8_residual"]].copy()
    c = pred.loc[pred.variant.eq(CAND), KEYS + ["pred_targets"]].copy()
    z = b.merge(c, on=KEYS, suffixes=("_b", "_c"), validate="one_to_one")
    z["actual_targets"] = pd.to_numeric(z.actual_targets, errors="coerce")
    z["baseline_pred_targets"] = pd.to_numeric(z.pred_targets_b, errors="coerce")
    z["candidate_pred_targets"] = pd.to_numeric(z.pred_targets_c, errors="coerce")
    z["target_delta"] = z.candidate_pred_targets - z.baseline_pred_targets
    z["prior_rb_room_share"] = pd.to_numeric(z.prior_rb_room_share, errors="coerce").fillna(0.0)
    z["r9_raw_r8_residual"] = pd.to_numeric(z.r9_raw_r8_residual, errors="coerce").fillna(0.0)
    z["identity_pct"] = z.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    z = z.loc[z.identity_pct.gt(0.80)].copy()
    z["high5"] = z.actual_targets.ge(5).astype(int)
    z["high7"] = z.actual_targets.ge(7).astype(int)
    z = z.dropna(subset=FEATURES + ["actual_targets"]).reset_index(drop=True)
    return z


def _fit(train: pd.DataFrame, label: str):
    if train[label].nunique() < 2:
        raise RuntimeError(f"training data has one class for {label}")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=911),
    )
    model.fit(train[FEATURES], train[label])
    return model


def _score(y: np.ndarray, p: np.ndarray, base_p: float) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    base = np.full(len(y), float(np.clip(base_p, 1e-6, 1 - 1e-6)))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, base))
    ll = float(log_loss(y, p, labels=[0, 1]))
    base_ll = float(log_loss(y, base, labels=[0, 1]))
    pct = pd.Series(p).rank(pct=True, method="average").to_numpy(float)
    hi = pct > 0.80
    event_rate = float(y.mean())
    hi_rate = float(y[hi].mean()) if hi.any() else np.nan
    capture = float(y[hi].sum() / y.sum()) if y.sum() else np.nan
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": event_rate,
        "auc": auc,
        "brier": brier,
        "base_rate_brier": base_brier,
        "brier_gain": float(base_brier - brier),
        "log_loss": ll,
        "base_rate_log_loss": base_ll,
        "log_loss_gain": float(base_ll - ll),
        "top_quintile_n": int(hi.sum()),
        "top_quintile_event_rate": hi_rate,
        "top_quintile_lift": float(hi_rate / event_rate) if event_rate > 0 and np.isfinite(hi_rate) else np.nan,
        "top_quintile_capture": capture,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r11_high_state_probability_v1"))
    a = ap.parse_args()

    z = _prepare(pd.read_csv(a.predictions, low_memory=False))
    fold_rows = []
    pred_rows = []
    coef_rows = []

    for train_seasons, test_season in FOLDS:
        tr = z.loc[z.season.isin(train_seasons)].copy()
        te = z.loc[z.season.eq(test_season)].copy()
        if tr.empty or te.empty:
            raise RuntimeError(f"empty R11 fold train={train_seasons} test={test_season}")
        for label in ("high5", "high7"):
            model = _fit(tr, label)
            p = model.predict_proba(te[FEATURES])[:, 1]
            base_p = float(tr[label].mean())
            m = _score(te[label].to_numpy(int), p, base_p)
            fold_rows.append({
                "label": label,
                "train_seasons": ",".join(map(str, train_seasons)),
                "test_season": int(test_season),
                "train_rows": int(len(tr)),
                "train_event_rate": base_p,
                **m,
            })
            q = te[KEYS + ["actual_targets", "baseline_pred_targets", "candidate_pred_targets", "prior_rb_room_share", "target_delta", "r9_raw_r8_residual", label]].copy()
            q["label"] = label
            q["pred_probability"] = p
            q["train_seasons"] = ",".join(map(str, train_seasons))
            pred_rows.append(q)
            scaler = model.named_steps["standardscaler"]
            lr = model.named_steps["logisticregression"]
            for i, f in enumerate(FEATURES):
                coef_rows.append({
                    "label": label,
                    "train_seasons": ",".join(map(str, train_seasons)),
                    "test_season": int(test_season),
                    "feature": f,
                    "scaler_mean": float(scaler.mean_[i]),
                    "scaler_scale": float(scaler.scale_[i]),
                    "coefficient": float(lr.coef_[0, i]),
                    "intercept": float(lr.intercept_[0]),
                })

    folds = pd.DataFrame(fold_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    coefs = pd.DataFrame(coef_rows)

    combined_rows = []
    for label in ("high5", "high7"):
        g = preds.loc[preds.label.eq(label)].copy()
        y = pd.to_numeric(g[label], errors="coerce").fillna(0).astype(int).to_numpy()
        p = pd.to_numeric(g.pred_probability, errors="coerce").to_numpy(float)
        # Fold-specific training base rates are already embodied in each fitted model;
        # combined comparison uses the pooled observed rate only as a descriptive null.
        pooled_base = float(y.mean())
        combined_rows.append({"label": label, **_score(y, p, pooled_base)})
    combined = pd.DataFrame(combined_rows)

    high5_folds = folds.loc[folds.label.eq("high5")].copy()
    high5_comb = combined.loc[combined.label.eq("high5")].iloc[0]
    auc_folds = int((high5_folds.auc >= MIN_FOLD_AUC).sum())
    brier_folds = int((high5_folds.brier_gain > 0).sum())
    support = bool(
        float(high5_comb.auc) >= MIN_COMBINED_AUC
        and auc_folds >= MIN_FOLDS_AUC
        and brier_folds >= MIN_FOLDS_BRIER_BEAT_BASE
        and float(high5_comb.top_quintile_lift) >= MIN_TOP_QUINTILE_LIFT
        and float(high5_comb.top_quintile_capture) >= MIN_TOP_QUINTILE_CAPTURE
    )

    result = {
        "diagnostic": "RB_R11_HIGH_STATE_PROBABILITY_V1",
        "disposition": "RB_R11_HIGH_STATE_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY" if support else "RB_R11_HIGH_STATE_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY",
        "state_signal_supported": support,
        "r9_status": "RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS_UNCHANGED",
        "r10_status": "RB_R10_MODERN_STABILITY_FAIL_UNCHANGED",
        "forensic_parent": "RB_R10_RECEIVING_STATE_FORENSIC_V1",
        "governance_note": "2023-2025 are not pristine confirmation; this only tests modern walk-forward separability before prospective 2026",
        "population": "strict-prior top20 receiving-identity RBs only",
        "high_state_definition": "5+ actual targets for diagnostic label; 7+ is supportive secondary label",
        "features": FEATURES,
        "model": "StandardScaler + LogisticRegression(C=1,L2)",
        "combined_high5_auc": float(high5_comb.auc),
        "combined_high5_brier_gain_vs_pooled_base": float(high5_comb.brier_gain),
        "combined_high5_top_quintile_lift": float(high5_comb.top_quintile_lift),
        "combined_high5_top_quintile_capture": float(high5_comb.top_quintile_capture),
        "high5_folds_auc_at_least_threshold": auc_folds,
        "high5_folds_brier_better_than_training_base": brier_folds,
        "diagnostic_thresholds": {
            "min_combined_auc": MIN_COMBINED_AUC,
            "min_fold_auc": MIN_FOLD_AUC,
            "min_folds_auc": MIN_FOLDS_AUC,
            "min_folds_brier_beat_base": MIN_FOLDS_BRIER_BEAT_BASE,
            "min_top_quintile_lift": MIN_TOP_QUINTILE_LIFT,
            "min_top_quintile_capture": MIN_TOP_QUINTILE_CAPTURE,
        },
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(a.out_dir / "rb_r11_state_probability_fold_summary.csv", index=False)
    combined.to_csv(a.out_dir / "rb_r11_state_probability_combined_summary.csv", index=False)
    preds.to_csv(a.out_dir / "rb_r11_state_probability_predictions.csv", index=False)
    coefs.to_csv(a.out_dir / "rb_r11_state_probability_coefficients.csv", index=False)
    (a.out_dir / "rb_r11_state_probability_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== folds ===")
    print(folds.to_string(index=False))
    print("\n=== combined ===")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
