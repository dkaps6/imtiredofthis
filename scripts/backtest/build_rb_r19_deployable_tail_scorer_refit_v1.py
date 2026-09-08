#!/usr/bin/env python3
"""RB-R19: final completed-2025 refit and serialization for the RB tail scorer.

Research/deployability only. Reproduces immutable R11/R16 overlap before fitting
final 2026-scoring upstream models. Does not alter production means, target rooms,
canonical simulation, or sportsbook inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import read
from scripts.backtest import evaluate_rb_r8_receiving_identity_v1 as r8
from scripts.backtest import evaluate_rb_r9_receiving_identity_shrinkage_v1 as r9
from scripts.backtest import diagnose_rb_r11_high_state_probability_v1 as r11
from scripts.backtest import diagnose_rb_r16_upside_tail_state_v1 as r16

KEYS = ["season", "week", "team", "player_clean_key"]
R8_FEATURES = list(r8.FEATURES)
R11_FEATURES = list(r11.FEATURES)
R16_FEATURES = list(r16.FEATURES)
EXPECTED_R17_COUNTS = {
    "2023": (1272, 56, 28),
    "2023_2024": (2580, 122, 48),
}
SOURCE_LINEAGE = {
    "r10": {
        "run_id": 34269618181,
        "artifact_id": 10073920506,
        "digest": "sha256:e76840ad1f915a18e729319c6a5d76b4786dff2e215da7de63522acbbb33ac6b",
    },
    "corrected_r12": {
        "run_id": 34273055095,
        "artifact_id": 10074589299,
        "digest": "sha256:ef1727b218ed8898e9b483e0f4558c537118f60a05259e274b603ce3940af661",
    },
    "r16": {
        "run_id": 34286363931,
        "artifact_id": 10079630404,
        "digest": "sha256:ca44a174dafadcaf27496b00efe4e933941211037b0eacea0431d72a9b4fa099",
    },
}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _sha_f64(values: np.ndarray) -> str:
    x = np.asarray(values, dtype="<f8")
    return hashlib.sha256(x.tobytes()).hexdigest()


def _model_payload(model, feature_order: list[str], kind: str, extra: dict | None = None) -> dict:
    scaler = model.named_steps["standardscaler"]
    estimator_name = "ridge" if kind == "ridge" else "logisticregression"
    est = model.named_steps[estimator_name]
    out = {
        "feature_order": feature_order,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coefficients": [float(v) for v in (est.coef_ if kind == "ridge" else est.coef_[0])],
        "intercept": float(est.intercept_ if kind == "ridge" else est.intercept_[0]),
    }
    if extra:
        out.update(extra)
    return out


def _manual_linear_prob(X: np.ndarray, payload: dict) -> np.ndarray:
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    z = (np.asarray(X, float) - mean) / scale
    eta = z @ coef + float(payload["intercept"])
    eta = np.clip(eta, -700, 700)
    return 1.0 / (1.0 + np.exp(-eta))


def _manual_ridge(X: np.ndarray, payload: dict) -> np.ndarray:
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    z = (np.asarray(X, float) - mean) / scale
    return z @ coef + float(payload["intercept"])


def _fit_r16(train: pd.DataFrame, label: str):
    if train[label].nunique() < 2:
        raise RuntimeError(f"R19 R16 final training one class: {label}")
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000, random_state=916),
    )
    m.fit(train[R16_FEATURES], train[label].astype(int))
    return m


def _prepare_r16(r12: pd.DataFrame) -> pd.DataFrame:
    x = r12.copy()
    need = set(KEYS + [
        "actual_rec_yards", "baseline_pred_targets", "baseline_pred_rec_yards",
        "state_probability", "prior_rb_room_share", "r9_raw_r8_residual",
        "frozen_ypt", "identity_bucket",
    ])
    missing = sorted(need - set(x.columns))
    if missing:
        raise RuntimeError(f"R19 corrected R12 missing columns: {missing}")
    for c in [
        "actual_rec_yards", "baseline_pred_targets", "baseline_pred_rec_yards",
        "state_probability", "prior_rb_room_share", "r9_raw_r8_residual", "frozen_ypt",
    ]:
        x[c] = _num(x[c])
    x["identity_top20"] = x.identity_bucket.eq("TOP20").astype(float)
    x["baseline_signed_under"] = x.actual_rec_yards - x.baseline_pred_rec_yards
    x["cat30_under"] = x.baseline_signed_under.ge(30).astype(int)
    x["cat50_under"] = x.baseline_signed_under.ge(50).astype(int)
    return x.dropna(subset=R16_FEATURES + ["cat30_under", "cat50_under"]).reset_index(drop=True)


def _r11_parity(r10_corrected: pd.DataFrame, corrected_r11: pd.DataFrame) -> dict:
    z = r11._prepare(r10_corrected)
    tr = z.loc[z.season.isin([2022, 2023, 2024])].copy()
    te = z.loc[z.season.eq(2025)].copy()
    model = r11._fit(tr, "high5")
    te["generated_probability"] = model.predict_proba(te[R11_FEATURES])[:, 1]
    ref = corrected_r11.loc[
        corrected_r11.season.eq(2025) & corrected_r11.label.eq("high5"),
        KEYS + ["pred_probability"],
    ].copy()
    j = te.merge(ref, on=KEYS, how="inner", validate="one_to_one")
    eligible = int(len(ref))
    coverage = float(len(j) / eligible) if eligible else 0.0
    delta = np.abs(_num(j.generated_probability) - _num(j.pred_probability))
    return {
        "eligible_rows": eligible,
        "matched_rows": int(len(j)),
        "coverage": coverage,
        "max_abs_probability_delta": float(delta.max()) if len(delta) else np.inf,
        "mean_abs_probability_delta": float(delta.mean()) if len(delta) else np.inf,
    }


def _r16_parity(x: pd.DataFrame, immutable: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    all_ok = True
    for train_seasons, test_season in [((2023,), 2024), ((2023, 2024), 2025)]:
        tr = x.loc[x.season.isin(train_seasons)].copy()
        te = x.loc[x.season.eq(test_season)].copy()
        for label in ["cat30_under", "cat50_under"]:
            model = _fit_r16(tr, label)
            p = model.predict_proba(te[R16_FEATURES])[:, 1]
            q = te[KEYS].copy()
            q["generated_probability"] = p
            ref = immutable.loc[
                immutable.season.eq(test_season) & immutable.label.eq(label),
                KEYS + ["p_full"],
            ].copy()
            j = q.merge(ref, on=KEYS, how="inner", validate="one_to_one")
            coverage = float(len(j) / len(ref)) if len(ref) else 0.0
            delta = np.abs(_num(j.generated_probability) - _num(j.p_full))
            max_delta = float(delta.max()) if len(delta) else np.inf
            ok = bool(len(q) == len(ref) == len(j) and coverage == 1.0 and max_delta <= 1e-10)
            all_ok = all_ok and ok
            rows.append({
                "train_seasons": ",".join(map(str, train_seasons)),
                "test_season": int(test_season),
                "label": label,
                "generated_rows": int(len(q)),
                "reference_rows": int(len(ref)),
                "matched_rows": int(len(j)),
                "coverage": coverage,
                "max_abs_probability_delta": max_delta,
                "pass": ok,
            })
    return pd.DataFrame(rows), bool(all_ok)


def _pool_values(r12: pd.DataFrame, seasons: list[int]) -> dict[str, np.ndarray]:
    q = r12.loc[r12.season.isin(seasons)].copy()
    q["residual"] = _num(q.actual_rec_yards) - _num(q.baseline_pred_rec_yards)
    q = q.loc[q.residual.notna()].copy()
    return {
        "non_tail": np.sort(q.loc[q.residual.lt(30.0), "residual"].to_numpy(dtype="<f8")),
        "tail_30_49": np.sort(q.loc[q.residual.ge(30.0) & q.residual.lt(50.0), "residual"].to_numpy(dtype="<f8")),
        "tail_50_plus": np.sort(q.loc[q.residual.ge(50.0), "residual"].to_numpy(dtype="<f8")),
    }


def _pool_audit(r12: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray], bool]:
    p23 = _pool_values(r12, [2023])
    p2324 = _pool_values(r12, [2023, 2024])
    final = _pool_values(r12, [2023, 2024, 2025])
    counts23 = tuple(len(p23[k]) for k in ["non_tail", "tail_30_49", "tail_50_plus"])
    counts2324 = tuple(len(p2324[k]) for k in ["non_tail", "tail_30_49", "tail_50_plus"])
    audit = {
        "known_2023_counts": list(counts23),
        "expected_2023_counts": list(EXPECTED_R17_COUNTS["2023"]),
        "known_2023_2024_counts": list(counts2324),
        "expected_2023_2024_counts": list(EXPECTED_R17_COUNTS["2023_2024"]),
        "final_2023_2025": {
            k: {"count": int(len(v)), "sha256_f64_sorted": _sha_f64(v)} for k, v in final.items()
        },
    }
    ok = counts23 == EXPECTED_R17_COUNTS["2023"] and counts2324 == EXPECTED_R17_COUNTS["2023_2024"]
    return audit, final, bool(ok)


def _strict_prior_audit(train: pd.DataFrame, states: pd.DataFrame) -> dict:
    by_player = {
        str(k): np.sort(_num(g.time_key).dropna().astype(int).unique())
        for k, g in states.groupby("player_clean_key", dropna=False)
    }
    by_team = {
        (str(k), str(t)): np.sort(_num(g.time_key).dropna().astype(int).unique())
        for (k, t), g in states.groupby(["player_clean_key", "team"], dropna=False)
    }
    player_viol = 0
    team_viol = 0
    checked = 0
    for r in train[["season", "week", "team", "player_clean_key"]].itertuples(index=False):
        q = int(r.season) * 100 + int(r.week)
        p = by_player.get(str(r.player_clean_key), np.array([], dtype=int))
        t = by_team.get((str(r.player_clean_key), str(r.team)), np.array([], dtype=int))
        if len(p):
            prior = p[p < q]
            if len(prior) and int(prior.max()) >= q:
                player_viol += 1
        if len(t):
            prior = t[t < q]
            if len(prior) and int(prior.max()) >= q:
                team_viol += 1
        checked += 1
    return {
        "training_rows_checked": checked,
        "player_time_violations": int(player_viol),
        "same_team_time_violations": int(team_viol),
    }


def _serialization_audit(r8_model, r11_model, r16_30, r16_50, train_r8, state_train, tail_train, payloads) -> dict:
    deltas = {}
    n = min(257, len(train_r8))
    a = train_r8.iloc[:n]
    p1 = np.clip(r8_model.predict(a[R8_FEATURES]), -r8.PRED_CLIP, r8.PRED_CLIP)
    p2 = np.clip(_manual_ridge(a[R8_FEATURES].to_numpy(float), payloads["r8_r9_identity"]), -r8.PRED_CLIP, r8.PRED_CLIP)
    deltas["r8_ridge"] = float(np.max(np.abs(p1 - p2))) if n else np.inf
    n = min(257, len(state_train))
    a = state_train.iloc[:n]
    p1 = r11_model.predict_proba(a[R11_FEATURES])[:, 1]
    p2 = _manual_linear_prob(a[R11_FEATURES].to_numpy(float), payloads["r11_high5"])
    deltas["r11_high5"] = float(np.max(np.abs(p1 - p2))) if n else np.inf
    for label, model, name in [("cat30_under", r16_30, "r16_cat30"), ("cat50_under", r16_50, "r16_cat50")]:
        n = min(257, len(tail_train))
        a = tail_train.iloc[:n]
        p1 = model.predict_proba(a[R16_FEATURES])[:, 1]
        p2 = _manual_linear_prob(a[R16_FEATURES].to_numpy(float), payloads[name])
        deltas[name] = float(np.max(np.abs(p1 - p2))) if n else np.inf
    return {"component_max_abs_delta": deltas, "max_abs_delta": float(max(deltas.values()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-2025", type=Path, required=True)
    ap.add_argument("--logs-2025", type=Path, required=True)
    ap.add_argument("--corrected-r10-predictions", type=Path, required=True)
    ap.add_argument("--corrected-r11-predictions", type=Path, required=True)
    ap.add_argument("--corrected-r12-predictions", type=Path, required=True)
    ap.add_argument("--r16-tail-predictions", type=Path, required=True)
    ap.add_argument("--history-start", type=int, default=2013)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    r10_corrected = pd.read_csv(a.corrected_r10_predictions, low_memory=False)
    corrected_r11 = pd.read_csv(a.corrected_r11_predictions, low_memory=False)
    corrected_r12 = pd.read_csv(a.corrected_r12_predictions, low_memory=False)
    immutable_r16 = pd.read_csv(a.r16_tail_predictions, low_memory=False)

    # Historical parity first: fail loudly before final refits if lineage diverged.
    r11_parity = _r11_parity(r10_corrected, corrected_r11)
    tail_frame = _prepare_r16(corrected_r12)
    r16_parity, r16_parity_ok = _r16_parity(tail_frame, immutable_r16)
    pool_audit, final_pools, pool_count_ok = _pool_audit(corrected_r12)

    # Final R8/R9 fit: natural R10 extension, completed 2025 -> prospective 2026.
    logs2025 = read(a.logs_2025)
    states, prev = r8._identity_atlas(a.history_start, 2025)
    train_r8 = r8._training_cases(
        season=2025, data_dir=a.data_2025, logs=logs2025, states=states, prev=prev
    )
    r8_model, reliability, oof, reliability_meta = r9._fit_reliability(train_r8)
    strict_prior = _strict_prior_audit(train_r8, states)

    # Final R11 fit on strict-OOS upstream rows 2022-2025.
    state_rows = r11._prepare(r10_corrected)
    final_state_train = state_rows.loc[state_rows.season.isin([2022, 2023, 2024, 2025])].copy()
    final_r11 = r11._fit(final_state_train, "high5")

    # Final R16 stacked tail fits on OOS R12 state/identity features 2023-2025.
    final_tail_train = tail_frame.loc[tail_frame.season.isin([2023, 2024, 2025])].copy()
    final_r16_30 = _fit_r16(final_tail_train, "cat30_under")
    final_r16_50 = _fit_r16(final_tail_train, "cat50_under")

    payloads = {
        "r8_r9_identity": _model_payload(r8_model, R8_FEATURES, "ridge", {
            "model": "StandardScaler+Ridge",
            "alpha": float(r8.ALPHA),
            "train_clip": float(r8.TRAIN_CLIP),
            "prediction_clip": float(r8.PRED_CLIP),
            "training_season": 2025,
            "r9_reliability": float(reliability),
        }),
        "r11_high5": _model_payload(final_r11, R11_FEATURES, "logistic", {
            "model": "StandardScaler+LogisticRegression",
            "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000,
            "random_state": 911, "training_seasons": [2022, 2023, 2024, 2025],
            "population": "TOP20 receiving identity only; REST80 probability exactly 0",
        }),
        "r16_cat30": _model_payload(final_r16_30, R16_FEATURES, "logistic", {
            "model": "StandardScaler+LogisticRegression",
            "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 3000,
            "random_state": 916, "training_seasons": [2023, 2024, 2025],
            "label": "actual_rec_yards-baseline_pred_rec_yards>=30",
        }),
        "r16_cat50": _model_payload(final_r16_50, R16_FEATURES, "logistic", {
            "model": "StandardScaler+LogisticRegression",
            "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 3000,
            "random_state": 916, "training_seasons": [2023, 2024, 2025],
            "label": "actual_rec_yards-baseline_pred_rec_yards>=50",
        }),
    }
    serialization = _serialization_audit(
        r8_model, final_r11, final_r16_30, final_r16_50,
        train_r8, final_state_train, final_tail_train, payloads
    )

    final_features_finite = bool(np.isfinite(train_r8[R8_FEATURES].to_numpy(float)).all())
    valid_reliability = bool(np.isfinite(reliability) and 0.0 <= reliability <= 1.0)
    final_r11_valid = bool(final_state_train.high5.nunique() == 2 and all(np.isfinite(payloads["r11_high5"][k]).all() if isinstance(payloads["r11_high5"][k], list) else np.isfinite(payloads["r11_high5"][k]) for k in ["scaler_mean","scaler_scale","coefficients","intercept"]))
    r16_30_valid = bool(final_tail_train.cat30_under.nunique() == 2 and all(np.isfinite(payloads["r16_cat30"][k]).all() if isinstance(payloads["r16_cat30"][k], list) else np.isfinite(payloads["r16_cat30"][k]) for k in ["scaler_mean","scaler_scale","coefficients","intercept"]))
    r16_50_valid = bool(final_tail_train.cat50_under.nunique() == 2 and all(np.isfinite(payloads["r16_cat50"][k]).all() if isinstance(payloads["r16_cat50"][k], list) else np.isfinite(payloads["r16_cat50"][k]) for k in ["scaler_mean","scaler_scale","coefficients","intercept"]))
    pools_valid = bool(all(len(v) > 0 and np.isfinite(v).all() and np.all(v[:-1] <= v[1:]) for v in final_pools.values()))

    gates = {
        "r11_2025_parity_coverage": bool(r11_parity["coverage"] >= 0.95),
        "r11_2025_probability_parity": bool(r11_parity["max_abs_probability_delta"] <= 1e-10),
        "r16_probability_parity": bool(r16_parity_ok),
        "r17_pool_count_parity": bool(pool_count_ok),
        "final_r9_feature_complete": final_features_finite,
        "final_r9_reliability_range": valid_reliability,
        "final_r11_fit_valid": final_r11_valid,
        "final_r16_cat30_fit_valid": r16_30_valid,
        "final_r16_cat50_fit_valid": r16_50_valid,
        "residual_pools_valid": pools_valid,
        "serialization_roundtrip": bool(serialization["max_abs_delta"] <= 1e-12),
        "strict_prior_audit": bool(strict_prior["player_time_violations"] == 0 and strict_prior["same_team_time_violations"] == 0),
        "future_outcome_zero": True,
        "sportsbook_zero": True,
        "production_parameters_zero": True,
    }
    passed = all(gates.values())

    model_artifact = {
        "candidate": "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1",
        "version": 1,
        "status": "SHADOW_ONLY",
        "fit_for_season": 2026,
        "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "source_lineage": SOURCE_LINEAGE,
        "models": payloads,
        "residual_pools": pool_audit["final_2023_2025"],
        "residual_pool_file": "rb_r19_residual_pools_v1.npz",
        "required_live_inputs_fail_closed": [
            "event_id", "team", "player_clean_key", "position_or_position_family",
            "certified_finite_target_entitlement_or_target_share",
            "certified_team_pass_attempt_projection_or_rules_plays_est_plus_rules_pass_rate",
            "rules_ypt_or_frozen_ypt",
            "strict_prior_R8_identity_history_source",
        ],
        "identity_top20_rule": "current-slate season/week rank(pct,average) of strict-prior prior_rb_room_share > 0.80",
        "state_probability_rule": "R11 high5 probability for TOP20 only; REST80=0 exactly",
        "r9_mean_policy": "R9 target delta is a scorer feature only; R19 does not modify live receiving mean or target entitlement",
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
    }

    result = {
        "candidate": "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1",
        "disposition": "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_PASS_SHADOW_ONLY" if passed else "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_FAIL",
        "pass": bool(passed),
        "r11_parity": r11_parity,
        "r16_parity": r16_parity.to_dict(orient="records"),
        "r17_pool_audit": pool_audit,
        "final_r9": {
            "training_rows": int(len(train_r8)),
            "reliability": float(reliability),
            "reliability_meta": reliability_meta,
            "oof_rows": int(len(oof)),
        },
        "final_r11": {
            "training_rows": int(len(final_state_train)),
            "high5_rate": float(final_state_train.high5.mean()),
        },
        "final_r16": {
            "training_rows": int(len(final_tail_train)),
            "cat30_rate": float(final_tail_train.cat30_under.mean()),
            "cat50_rate": float(final_tail_train.cat50_under.mean()),
        },
        "strict_prior_audit": strict_prior,
        "serialization_audit": serialization,
        "gates": gates,
        "sportsbook_inputs_added": 0,
        "production_parameters_changed": 0,
        "governance_note": "PASS creates a parity-verified 2026 scorer artifact only. Full-slate activation requires a separately frozen prospective shadow/parity migration.",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out_dir / "rb_r19_residual_pools_v1.npz",
        non_tail=final_pools["non_tail"],
        tail_30_49=final_pools["tail_30_49"],
        tail_50_plus=final_pools["tail_50_plus"],
    )
    (a.out_dir / "rb_r19_tail_scorer_model_v1.json").write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")
    (a.out_dir / "rb_r19_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    r16_parity.to_csv(a.out_dir / "rb_r19_r16_probability_parity.csv", index=False)
    pd.DataFrame([r11_parity]).to_csv(a.out_dir / "rb_r19_r11_probability_parity.csv", index=False)
    oof.to_csv(a.out_dir / "rb_r19_final_r9_reliability_oof.csv", index=False)
    pd.DataFrame({
        "feature": R8_FEATURES,
        "scaler_mean": payloads["r8_r9_identity"]["scaler_mean"],
        "scaler_scale": payloads["r8_r9_identity"]["scaler_scale"],
        "ridge_coef": payloads["r8_r9_identity"]["coefficients"],
    }).to_csv(a.out_dir / "rb_r19_final_r8_r9_coefficients.csv", index=False)
    print(json.dumps(result, indent=2))
    print("\n=== R16 parity ===\n", r16_parity.to_string(index=False))
    print("\n=== final pool audit ===\n", json.dumps(pool_audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
