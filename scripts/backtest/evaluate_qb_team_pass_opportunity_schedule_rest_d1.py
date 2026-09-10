#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHA = 20.0
BOOTSTRAP_SEED = 5611
BOOTSTRAP_N = 10_000
TOL = 1e-6
FEATURES = [
    "home",
    "rest_days_minus_7",
    "opponent_rest_days_minus_7",
    "rest_diff",
    "short_week",
    "long_rest",
    "thursday",
    "monday",
]
KEYS = ["season", "week", "team", "player_clean_key"]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_features(source_root: Path) -> pd.DataFrame:
    f = pd.read_csv(one(source_root, "schedule_rest_target_feasibility_2024_2025.csv"), low_memory=False)
    f.columns = [str(c).strip().lower() for c in f.columns]
    required = set(KEYS + [
        "home", "rest_days", "opponent_rest_days", "rest_diff",
        "short_week", "long_rest", "thursday", "monday",
    ])
    missing = sorted(required - set(f.columns))
    if missing:
        raise RuntimeError(f"schedule/rest source missing {missing}")
    f = f[list(required)].copy()
    f["season"] = num(f["season"])
    f["week"] = num(f["week"])
    f["team"] = f["team"].fillna("").astype(str).str.upper().str.strip()
    f["player_clean_key"] = f["player_clean_key"].fillna("").astype(str).str.strip()
    f["rest_days_minus_7"] = num(f["rest_days"]) - 7.0
    f["opponent_rest_days_minus_7"] = num(f["opponent_rest_days"]) - 7.0
    for c in FEATURES:
        f[c] = num(f[c])
    if f[FEATURES].isna().any().any():
        bad = f.loc[f[FEATURES].isna().any(axis=1), KEYS + FEATURES].head(10)
        raise RuntimeError(f"non-finite frozen schedule/rest features: {bad.to_dict('records')}")
    if f.duplicated(KEYS).any():
        raise RuntimeError("duplicate schedule/rest feature keys")
    return f[KEYS + FEATURES].copy()


def load_2024_chain(chain_root: Path) -> pd.DataFrame:
    cols = KEYS + [
        "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa",
        "football_synthesis", "pred_D", "pred_C", "pred_S", "actual_D",
    ]
    x = pd.read_csv(one(chain_root, "qb_opportunity_chain_casebook.csv"), usecols=cols, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ["season", "week", "actual_pass_yards", "actual_attempts", "pred_attempts", "pred_ypa", "football_synthesis", "pred_d", "pred_c", "pred_s", "actual_d"]:
        x[c] = num(x[c])
    x["team"] = x["team"].fillna("").astype(str).str.upper().str.strip()
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x = x.loc[x["season"].eq(2024)].copy()
    if x.empty:
        raise RuntimeError("no 2024 opportunity-chain rows")
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate 2024 chain keys")
    return x


def metrics(actual: pd.Series, pred: pd.Series, *, miss_levels: tuple[float, ...] = ()) -> dict:
    a = num(actual).to_numpy(float)
    p = num(pred).to_numpy(float)
    if len(a) != len(p) or len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise RuntimeError("invalid metric arrays")
    err = p - a
    out = {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "corr": float(np.corrcoef(a, p)[0, 1]) if len(a) >= 2 else np.nan,
        "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
    }
    for level in miss_levels:
        out[f"miss_{int(level)}_plus_rate"] = float(np.mean(np.abs(err) >= level))
    return out


def bootstrap_mae_gain(actual: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict:
    row_gain = np.abs(baseline - actual) - np.abs(candidate - actual)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(row_gain)
    means = np.empty(BOOTSTRAP_N, dtype=float)
    chunk = 1000
    for start in range(0, BOOTSTRAP_N, chunk):
        m = min(chunk, BOOTSTRAP_N - start)
        idx = rng.integers(0, n, size=(m, n))
        means[start:start+m] = row_gain[idx].mean(axis=1)
    return {
        "observed_mae_gain": float(row_gain.mean()),
        "p_gain_gt_0": float(np.mean(means > 0)),
        "bootstrap_n": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "p05": float(np.quantile(means, 0.05)),
        "p50": float(np.quantile(means, 0.50)),
        "p95": float(np.quantile(means, 0.95)),
    }


def serialize_model(scaler: StandardScaler, model: Ridge) -> dict:
    return {
        "model": "StandardScaler + Ridge",
        "alpha": ALPHA,
        "fit_intercept": True,
        "features": FEATURES,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "ridge_coef": [float(v) for v in model.coef_],
        "ridge_intercept": float(model.intercept_),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    feats = load_features(a.source_root)
    chain = load_2024_chain(a.chain_root)
    f24 = feats.loc[feats["season"].eq(2024)].copy()
    z = chain.merge(f24, on=KEYS, how="inner", validate="one_to_one")
    if len(z) != len(chain):
        raise RuntimeError(f"2024 feature alignment drift chain={len(chain)} aligned={len(z)}")

    z["baseline_attempt_identity"] = z["pred_d"] * z["pred_c"] * z["pred_s"]
    identity_gap = float((z["baseline_attempt_identity"] - z["pred_attempts"]).abs().max())

    fit = z.loc[z["week"].between(1, 9)].copy()
    hold = z.loc[z["week"].between(10, 18)].copy()
    if fit.empty or hold.empty:
        raise RuntimeError(f"empty temporal split fit={len(fit)} hold={len(hold)}")
    if fit.duplicated(KEYS).any() or hold.duplicated(KEYS).any():
        raise RuntimeError("duplicate split keys")

    X_fit = fit[FEATURES].to_numpy(float)
    y_fit = (fit["actual_d"] - fit["pred_d"]).to_numpy(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)
    model = Ridge(alpha=ALPHA, fit_intercept=True)
    model.fit(Xs, y_fit)

    hold_corr = model.predict(scaler.transform(hold[FEATURES].to_numpy(float)))
    hold["predicted_d_residual"] = hold_corr
    hold["candidate_d"] = hold["pred_d"] + hold["predicted_d_residual"]
    hold["candidate_attempts"] = hold["candidate_d"] * hold["pred_c"] * hold["pred_s"]
    hold["candidate_pass_yards"] = (
        hold["football_synthesis"]
        + (hold["candidate_attempts"] - hold["pred_attempts"]) * hold["pred_ypa"]
    )

    d_base = metrics(hold["actual_d"], hold["pred_d"])
    d_cand = metrics(hold["actual_d"], hold["candidate_d"])
    a_base = metrics(hold["actual_attempts"], hold["pred_attempts"], miss_levels=(8, 10))
    a_cand = metrics(hold["actual_attempts"], hold["candidate_attempts"], miss_levels=(8, 10))
    y_base = metrics(hold["actual_pass_yards"], hold["football_synthesis"], miss_levels=(75, 100))
    y_cand = metrics(hold["actual_pass_yards"], hold["candidate_pass_yards"], miss_levels=(75, 100))

    boot = bootstrap_mae_gain(
        hold["actual_pass_yards"].to_numpy(float),
        hold["football_synthesis"].to_numpy(float),
        hold["candidate_pass_yards"].to_numpy(float),
    )

    integrity = {
        "immutable_source_artifact_contract": True,
        "immutable_chain_artifact_contract": True,
        "sportsbook_or_result_features_used": False,
        "target_2025_rows_used_for_fit_selection_or_scoring": 0,
        "exact_frozen_feature_count": int(len(FEATURES)),
        "exact_frozen_features": FEATURES,
        "ridge_alpha": ALPHA,
        "production_files_changed": False,
        "baseline_attempt_identity_max_abs_gap": identity_gap,
        "candidate_only_changes_team_pass_opportunity_before_propagation": True,
        "fit_rows_2024_w1_9": int(len(fit)),
        "holdout_rows_2024_w10_18": int(len(hold)),
    }
    integrity_gates = {
        "exact_immutable_sources": True,
        "zero_sportsbook_result_features": True,
        "zero_2025_target_use": True,
        "exact_eight_features": len(FEATURES) == 8,
        "one_ridge_alpha_20_only": ALPHA == 20.0,
        "no_production_change": True,
        "baseline_attempt_identity_reconciles": identity_gap <= TOL,
        "candidate_only_team_pass_opportunity": True,
        "temporal_split_nonempty_unique": len(fit) > 0 and len(hold) > 0,
    }
    all_integrity = all(integrity_gates.values())

    dev_gates = {
        "team_pass_opportunity_mae_gain_ge_0_25": (d_base["mae"] - d_cand["mae"]) >= 0.25,
        "qb_attempt_mae_gain_ge_0_10": (a_base["mae"] - a_cand["mae"]) >= 0.10,
        "qb_pass_yard_mae_gain_ge_0_75": (y_base["mae"] - y_cand["mae"]) >= 0.75,
        "qb_pass_yard_rmse_nonworse": y_cand["rmse"] <= y_base["rmse"] + 1e-12,
        "qb_pass_yard_corr_nonworse": y_cand["corr"] >= y_base["corr"] - 1e-12,
        "qb_pass_yard_p90_nonworse": y_cand["p90_abs_error"] <= y_base["p90_abs_error"] + 1e-12,
        "qb_pass_yard_100_plus_nonworse": y_cand["miss_100_plus_rate"] <= y_base["miss_100_plus_rate"] + 1e-12,
        "qb_attempt_10_plus_nonworse": a_cand["miss_10_plus_rate"] <= a_base["miss_10_plus_rate"] + 1e-12,
        "bootstrap_p_gain_ge_0_70": boot["p_gain_gt_0"] >= 0.70,
        "all_integrity_gates_pass": all_integrity,
    }
    survivor = all(dev_gates.values())
    disposition = (
        "MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE"
        if not all_integrity else
        "SCHEDULE_REST_D1_DEVELOPMENT_SURVIVOR"
        if survivor else
        "SCHEDULE_REST_D1_FAIL_NO_CONFIRMATION"
    )

    result = {
        "migration": "QB_TEAM_PASS_OPPORTUNITY_SCHEDULE_REST_D1",
        "disposition": disposition,
        "production_actionable": False,
        "confirmation_authorized": bool(survivor and all_integrity),
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "development_gates": dev_gates,
        "metrics": {
            "team_pass_opportunity": {"baseline": d_base, "candidate": d_cand, "mae_gain": d_base["mae"] - d_cand["mae"]},
            "qb_attempts": {"baseline": a_base, "candidate": a_cand, "mae_gain": a_base["mae"] - a_cand["mae"]},
            "qb_pass_yards": {"baseline": y_base, "candidate": y_cand, "mae_gain": y_base["mae"] - y_cand["mae"]},
        },
        "correction": {
            "mean": float(np.mean(hold_corr)),
            "mean_abs": float(np.mean(np.abs(hold_corr))),
            "p90_abs": float(np.quantile(np.abs(hold_corr), 0.90)),
            "min": float(np.min(hold_corr)),
            "max": float(np.max(hold_corr)),
        },
        "development_model": serialize_model(scaler, model),
        "bootstrap": boot,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    keep = KEYS + FEATURES + [
        "actual_d", "pred_d", "predicted_d_residual", "candidate_d",
        "actual_attempts", "pred_attempts", "candidate_attempts",
        "actual_pass_yards", "football_synthesis", "candidate_pass_yards",
    ]
    hold[keep].to_csv(a.out_dir / "schedule_rest_d1_development_predictions.csv", index=False)
    (a.out_dir / "schedule_rest_d1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if survivor and all_integrity:
        scaler_all = StandardScaler()
        X_all = scaler_all.fit_transform(z[FEATURES].to_numpy(float))
        y_all = (z["actual_d"] - z["pred_d"]).to_numpy(float)
        model_all = Ridge(alpha=ALPHA, fit_intercept=True)
        model_all.fit(X_all, y_all)
        frozen = serialize_model(scaler_all, model_all)
        frozen.update({
            "training_scope": "2024_ALL_M89_ALIGNED_ROWS",
            "training_rows": int(len(z)),
            "target": "actual_D_minus_pred_D",
            "source_audit_run": 34533408818,
            "opportunity_chain_run": 34523313743,
            "2025_target_outcomes_used": False,
        })
        (a.out_dir / "schedule_rest_d1_frozen_2025_confirmation_model.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_integrity else 2


if __name__ == "__main__":
    raise SystemExit(main())
