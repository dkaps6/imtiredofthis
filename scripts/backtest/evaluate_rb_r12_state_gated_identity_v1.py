#!/usr/bin/env python3
"""RB-R12 state-gated receiving-identity stability candidate.

R10 showed the full R9 identity adjustment improves high-target states but hurts
ordinary games for top receiving-identity backs. R11 showed those high states are
meaningfully predictable pregame. R12 combines those findings without changing the
canonical RB receiving pool:

- REST80 backs start from the R9 target mean.
- TOP20 receiving-identity backs blend between baseline and R9.
- A strict-prior logistic model estimates P(5+ targets).
- Training-only ridge calibration learns how much of the R9 movement to retain as a
  linear function of that probability: alpha(p) = b0 + b1*p, clipped to [0,1].
- The entire team RB target vector is then renormalized back to the exact baseline
  RB target pool.
- Receiving yards use the frozen baseline yards-per-target mapping, so this run tests
  entitlement/state gating only, not a new efficiency model.

Modern folds are already research-visible and therefore NOT pristine confirmation:
  2022 -> 2023
  2022-2023 -> 2024
  2022-2024 -> 2025
A successful stability result would only authorize a frozen prospective 2026 shadow
candidate. It does not promote R12.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "M38_EXPLICIT_BASELINE"
R9 = "RB_R9_IDENTITY_SHRINKAGE"
R12 = "RB_R12_STATE_GATED_IDENTITY"
KEYS = ["season", "week", "team", "player_clean_key"]
STATE_FEATURES = ["baseline_pred_targets", "prior_rb_room_share", "target_delta", "r9_raw_r8_residual"]
FOLDS = [((2022,), 2023), ((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]

# Frozen before executing R12.
MIN_COMBINED_TARGET_GAIN_VS_R9 = 0.00
MIN_COMBINED_REC_GAIN_VS_R9 = 0.00
MIN_TOP20_REC_GAIN_VS_R9 = 0.20
MIN_TOP20_REC_GAIN_VS_BASE = 0.00
MIN_TOP20_SEASONS_NONWORSE_VS_R9 = 2
MAX_P90_WORSEN_VS_R9 = 0.25
MAX_CAT30_RATE_WORSEN = 0.0025
MAX_CAT50_RATE_WORSEN = 0.0025
MAX_TEAM_POOL_GAP = 1e-10


def _prepare(pred: pd.DataFrame) -> pd.DataFrame:
    req = set(KEYS + ["variant", "actual_targets", "pred_targets", "actual_rec_yards", "mc_rec_yards", "prior_rb_room_share", "r9_raw_r8_residual"])
    missing = sorted(req - set(pred.columns))
    if missing:
        raise RuntimeError(f"R10 predictions missing columns: {missing}")
    b = pred.loc[pred.variant.eq(BASE), KEYS + ["actual_targets", "pred_targets", "actual_rec_yards", "mc_rec_yards", "prior_rb_room_share", "r9_raw_r8_residual"]].copy()
    c = pred.loc[pred.variant.eq(R9), KEYS + ["pred_targets", "mc_rec_yards"]].copy()
    z = b.merge(c, on=KEYS, suffixes=("_b", "_r9"), validate="one_to_one")
    z["actual_targets"] = pd.to_numeric(z.actual_targets, errors="coerce")
    z["actual_rec_yards"] = pd.to_numeric(z.actual_rec_yards, errors="coerce")
    z["baseline_pred_targets"] = pd.to_numeric(z.pred_targets_b, errors="coerce")
    z["r9_pred_targets"] = pd.to_numeric(z.pred_targets_r9, errors="coerce")
    z["baseline_pred_rec_yards"] = pd.to_numeric(z.mc_rec_yards_b, errors="coerce")
    z["r9_pred_rec_yards"] = pd.to_numeric(z.mc_rec_yards_r9, errors="coerce")
    z["prior_rb_room_share"] = pd.to_numeric(z.prior_rb_room_share, errors="coerce").fillna(0.0)
    z["r9_raw_r8_residual"] = pd.to_numeric(z.r9_raw_r8_residual, errors="coerce").fillna(0.0)
    z["target_delta"] = z.r9_pred_targets - z.baseline_pred_targets
    z["identity_pct"] = z.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    z["identity_bucket"] = np.where(z.identity_pct.gt(0.80), "TOP20", "REST80")
    z["high5"] = z.actual_targets.ge(5).astype(int)
    return z.dropna(subset=STATE_FEATURES + ["actual_targets", "actual_rec_yards"]).reset_index(drop=True)


def _fit_state(train_top: pd.DataFrame):
    if train_top.high5.nunique() < 2:
        raise RuntimeError("R12 training top20 has one high5 class")
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=912))
    m.fit(train_top[STATE_FEATURES], train_top.high5)
    return m


def _fit_blend(train_top: pd.DataFrame, p_train: np.ndarray):
    d = train_top.target_delta.to_numpy(float)
    y = (train_top.actual_targets - train_top.baseline_pred_targets).to_numpy(float)
    X = np.column_stack([d, d * np.asarray(p_train, float)])
    # Training-only regularization; no intercept because zero R9 movement must imply
    # zero R12 movement from baseline.
    m = Ridge(alpha=1.0, fit_intercept=False)
    m.fit(X, y)
    return m


def _apply_fold(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tr_top = train.loc[train.identity_bucket.eq("TOP20")].copy()
    te = test.copy()
    state = _fit_state(tr_top)
    p_train = state.predict_proba(tr_top[STATE_FEATURES])[:, 1]
    blend = _fit_blend(tr_top, p_train)

    te["state_probability"] = 0.0
    top_mask = te.identity_bucket.eq("TOP20")
    te.loc[top_mask, "state_probability"] = state.predict_proba(te.loc[top_mask, STATE_FEATURES])[:, 1]
    coef0, coef1 = [float(v) for v in blend.coef_]
    te["blend_alpha_raw"] = np.where(top_mask, coef0 + coef1 * te.state_probability, 1.0)
    te["blend_alpha"] = np.where(top_mask, np.clip(te.blend_alpha_raw, 0.0, 1.0), 1.0)

    te["r12_targets_preconserve"] = np.where(
        top_mask,
        te.baseline_pred_targets + te.blend_alpha * te.target_delta,
        te.r9_pred_targets,
    )
    te["r12_targets_preconserve"] = te.r12_targets_preconserve.clip(lower=0.0)

    te["baseline_team_rb_targets"] = te.groupby(["season", "week", "team"])["baseline_pred_targets"].transform("sum")
    te["r12_team_preconserve"] = te.groupby(["season", "week", "team"])["r12_targets_preconserve"].transform("sum")
    scale = np.where(te.r12_team_preconserve.gt(1e-12), te.baseline_team_rb_targets / te.r12_team_preconserve, 1.0)
    te["r12_pred_targets"] = te.r12_targets_preconserve * scale

    ypt = np.where(te.baseline_pred_targets.gt(1e-9), te.baseline_pred_rec_yards / te.baseline_pred_targets, np.nan)
    ypt2 = np.where(te.r9_pred_targets.gt(1e-9), te.r9_pred_rec_yards / te.r9_pred_targets, np.nan)
    ypt = np.where(np.isfinite(ypt), ypt, ypt2)
    ypt = np.where(np.isfinite(ypt), ypt, 0.0)
    te["frozen_ypt"] = ypt
    te["r12_pred_rec_yards"] = te.r12_pred_targets * te.frozen_ypt
    te["team_pool_gap"] = te.groupby(["season", "week", "team"])["r12_pred_targets"].transform("sum") - te.baseline_team_rb_targets

    meta = {
        "state_train_rows": int(len(tr_top)),
        "state_train_rate_5plus": float(tr_top.high5.mean()),
        "blend_coef_base_movement": coef0,
        "blend_coef_state_interaction": coef1,
        "test_top20_rows": int(top_mask.sum()),
        "mean_top20_state_probability": float(te.loc[top_mask, "state_probability"].mean()),
        "mean_top20_blend_alpha": float(te.loc[top_mask, "blend_alpha"].mean()),
        "top20_alpha_zero_rate": float(te.loc[top_mask, "blend_alpha"].le(1e-12).mean()),
        "top20_alpha_one_rate": float(te.loc[top_mask, "blend_alpha"].ge(1 - 1e-12).mean()),
        "max_abs_team_pool_gap": float(te.team_pool_gap.abs().max()),
    }
    return te, meta


def _metrics(g: pd.DataFrame, target_col: str, rec_col: str) -> dict:
    ta = (g[target_col] - g.actual_targets).abs()
    re = g[rec_col] - g.actual_rec_yards
    ra = re.abs()
    return {
        "n": int(len(g)),
        "target_mae": float(ta.mean()),
        "rec_mae": float(ra.mean()),
        "rec_rmse": float(np.sqrt(np.mean(re * re))),
        "rec_bias": float(re.mean()),
        "rec_p90": float(np.quantile(ra, 0.90)),
        "cat30_rate": float(ra.ge(30).mean()),
        "cat50_rate": float(ra.ge(50).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r12_state_gated_identity_v1"))
    a = ap.parse_args()

    z = _prepare(pd.read_csv(a.predictions, low_memory=False))
    test_parts = []
    fold_meta = []
    for train_seasons, test_season in FOLDS:
        tr = z.loc[z.season.isin(train_seasons)].copy()
        te = z.loc[z.season.eq(test_season)].copy()
        out, meta = _apply_fold(tr, te)
        out["train_seasons"] = ",".join(map(str, train_seasons))
        test_parts.append(out)
        fold_meta.append({"train_seasons": list(train_seasons), "test_season": int(test_season), **meta})

    x = pd.concat(test_parts, ignore_index=True)
    rows = []
    for season_bucket, g0 in [("COMBINED", x)] + [(str(s), g) for s, g in x.groupby("season")]:
        for population, g in [("ALL", g0), ("TOP20", g0.loc[g0.identity_bucket.eq("TOP20")]), ("REST80", g0.loc[g0.identity_bucket.eq("REST80")])]:
            if g.empty:
                continue
            for variant, tc, rc in [
                (BASE, "baseline_pred_targets", "baseline_pred_rec_yards"),
                (R9, "r9_pred_targets", "r9_pred_rec_yards"),
                (R12, "r12_pred_targets", "r12_pred_rec_yards"),
            ]:
                rows.append({"season_bucket": season_bucket, "population": population, "variant": variant, **_metrics(g, tc, rc)})
    summary = pd.DataFrame(rows)

    def row(season_bucket: str, population: str, variant: str) -> pd.Series:
        q = summary.loc[summary.season_bucket.eq(season_bucket) & summary.population.eq(population) & summary.variant.eq(variant)]
        if len(q) != 1:
            raise RuntimeError(f"R12 summary lookup failed {season_bucket=} {population=} {variant=}")
        return q.iloc[0]

    b = row("COMBINED", "ALL", BASE)
    r9 = row("COMBINED", "ALL", R9)
    r12 = row("COMBINED", "ALL", R12)
    bt = row("COMBINED", "TOP20", BASE)
    r9t = row("COMBINED", "TOP20", R9)
    r12t = row("COMBINED", "TOP20", R12)

    top20_nonworse = 0
    for s in (2023, 2024, 2025):
        a9 = row(str(s), "TOP20", R9)
        a12 = row(str(s), "TOP20", R12)
        top20_nonworse += int(float(a12.rec_mae) <= float(a9.rec_mae) + 1e-12)

    gates = {
        "combined_target_gain_vs_r9": float(r9.target_mae - r12.target_mae) >= MIN_COMBINED_TARGET_GAIN_VS_R9,
        "combined_rec_gain_vs_r9": float(r9.rec_mae - r12.rec_mae) >= MIN_COMBINED_REC_GAIN_VS_R9,
        "top20_rec_gain_vs_r9": float(r9t.rec_mae - r12t.rec_mae) >= MIN_TOP20_REC_GAIN_VS_R9,
        "top20_rec_nonworse_vs_baseline": float(r12t.rec_mae) <= float(bt.rec_mae) - MIN_TOP20_REC_GAIN_VS_BASE + 1e-12,
        "top20_seasons_nonworse_vs_r9": top20_nonworse >= MIN_TOP20_SEASONS_NONWORSE_VS_R9,
        "combined_p90_guard_vs_r9": float(r12.rec_p90) <= float(r9.rec_p90) + MAX_P90_WORSEN_VS_R9,
        "combined_cat30_guard_vs_r9": float(r12.cat30_rate) <= float(r9.cat30_rate) + MAX_CAT30_RATE_WORSEN,
        "combined_cat50_guard_vs_r9": float(r12.cat50_rate) <= float(r9.cat50_rate) + MAX_CAT50_RATE_WORSEN,
        "team_rb_target_pool_conservation": float(x.team_pool_gap.abs().max()) <= MAX_TEAM_POOL_GAP,
        "sportsbook_zero": True,
    }
    passed = all(gates.values())

    result = {
        "candidate": "RB_R12_STATE_GATED_IDENTITY_V1",
        "disposition": "RB_R12_STATE_GATED_MODERN_STABILITY_PASS_DIAGNOSTIC_ONLY" if passed else "RB_R12_STATE_GATED_MODERN_STABILITY_FAIL_DIAGNOSTIC_ONLY",
        "modern_stability_pass": bool(passed),
        "fresh_confirmation": False,
        "governance_note": "2023-2025 are research-visible. A pass only authorizes freezing a prospective 2026 shadow candidate.",
        "parents": ["RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS", "RB_R10_MODERN_STABILITY_FAIL", "RB_R11_HIGH_STATE_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY"],
        "state_features": STATE_FEATURES,
        "combined_baseline_target_mae": float(b.target_mae),
        "combined_r9_target_mae": float(r9.target_mae),
        "combined_r12_target_mae": float(r12.target_mae),
        "combined_baseline_rec_mae": float(b.rec_mae),
        "combined_r9_rec_mae": float(r9.rec_mae),
        "combined_r12_rec_mae": float(r12.rec_mae),
        "combined_top20_baseline_rec_mae": float(bt.rec_mae),
        "combined_top20_r9_rec_mae": float(r9t.rec_mae),
        "combined_top20_r12_rec_mae": float(r12t.rec_mae),
        "top20_seasons_nonworse_vs_r9": int(top20_nonworse),
        "folds": fold_meta,
        "gates": gates,
        "sportsbook_inputs_added": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "rb_r12_predictions.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r12_summary.csv", index=False)
    (a.out_dir / "rb_r12_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n=== R12 summary ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
