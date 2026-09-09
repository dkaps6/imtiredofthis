#!/usr/bin/env python3
"""RB-R21 Phase A: seal exact Week-1 CONTROL/SHADOW receiving-yard forecasts.

This wrapper does not create a new football model. It executes the unchanged frozen
R20 evaluator, intercepts the already-produced canonical/adapted RB rec-yard draw
arrays, and persists those exact arrays plus immutable hashes before 2026 outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_r20_real_slate_shadow_integration_v1 as r20

CANDIDATE = "RB_R21_2026_PROSPECTIVE_TAIL_FORECAST_LOCK_V1"
PLAN_COMMIT = "cdd503919e56860c6f17df12e01f4f34072d888c"
KICKOFF_CUTOFF_UTC = "2026-09-10T00:20:00Z"
EXPECTED_R20 = {
    "run_id": 34291433027,
    "artifact_id": 10081502774,
    "name": "rb-r20-real-slate-shadow-integration-v1",
    "digest": "sha256:853c0d3aea971c058ae6cc3b80c99ae2a5a0f681fae642835bbfa544da8283ca",
    "head_sha": "587bf2a89ca16f11361016df3915361390289a7e",
}
EXPECTED_ROWS = 94
EXPECTED_DRAWS = 10000


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def sha256_f64(arr: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = raw.get("artifacts", [])
    matches = [a for a in arts if a.get("name") == EXPECTED_R20["name"]]
    if len(matches) != 1:
        return {"pass": False, "reason": f"expected one R20 artifact, got {len(matches)}"}
    a = matches[0]
    wr = a.get("workflow_run") or {}
    checks = {
        "id": int(a.get("id", -1)) == EXPECTED_R20["artifact_id"],
        "name": a.get("name") == EXPECTED_R20["name"],
        "digest": a.get("digest") == EXPECTED_R20["digest"],
        "run_id": int(wr.get("id", -1)) == EXPECTED_R20["run_id"],
        "head_sha": wr.get("head_sha") == EXPECTED_R20["head_sha"],
        "not_expired": a.get("expired") is False,
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "observed": a}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-slate-dir", type=Path, required=True)
    ap.add_argument("--r19-model", type=Path, required=True)
    ap.add_argument("--r19-pools", type=Path, required=True)
    ap.add_argument("--replay-artifact-metadata", type=Path, required=True)
    ap.add_argument("--raw-paid-artifact-metadata", type=Path, required=True)
    ap.add_argument("--r19-artifact-metadata", type=Path, required=True)
    ap.add_argument("--r20-artifact-dir", type=Path, required=True)
    ap.add_argument("--r20-artifact-metadata", type=Path, required=True)
    ap.add_argument("--current-run-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    current_run = json.loads(a.current_run_metadata.read_text(encoding="utf-8"))
    started_at = parse_utc(current_run["run_started_at"])
    cutoff = parse_utc(KICKOFF_CUTOFF_UTC)
    pre_kickoff = started_at < cutoff

    r20_meta = artifact_record(a.r20_artifact_metadata)
    parent_result_path = a.r20_artifact_dir / "rb_r20_result.json"
    parent_casebook_path = a.r20_artifact_dir / "rb_r20_shadow_casebook.csv"
    parent_player_audit_path = a.r20_artifact_dir / "rb_r20_adapter_player_audit.csv"
    for p in [parent_result_path, parent_casebook_path, parent_player_audit_path]:
        if not p.is_file() or p.stat().st_size == 0:
            raise RuntimeError(f"missing immutable R20 evidence file: {p}")

    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_pass = bool(
        parent_result.get("pass") is True
        and parent_result.get("disposition") == "RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_PASS_SHADOW_ONLY"
        and int(parent_result.get("current_or_future_outcomes_used", -1)) == 0
        and int(parent_result.get("sportsbook_inputs_added", -1)) == 0
        and int(parent_result.get("production_parameters_changed", -1)) == 0
    )

    capture: dict[str, object] = {}
    original_adapt = r20.adapt_rb_receiving_tail

    def capture_adapt(canonical, metrics, risk, pools, *, seed):
        adapted, audit = original_adapt(canonical, metrics, risk, pools, seed=seed)
        if not capture:
            control = {
                k: np.array(v, dtype=float, copy=True)
                for k, v in canonical.values.items()
                if isinstance(k, tuple) and len(k) == 3 and k[2] == "rec_yards"
            }
            shadow = {
                k: np.array(v, dtype=float, copy=True)
                for k, v in adapted.values.items()
                if isinstance(k, tuple) and len(k) == 3 and k[2] == "rec_yards"
            }
            capture["control"] = control
            capture["shadow"] = shadow
            capture["metrics"] = metrics.copy(deep=True)
            capture["risk"] = risk.copy(deep=True)
            capture["audit"] = audit.copy(deep=True)
        return adapted, audit

    r20.adapt_rb_receiving_tail = capture_adapt
    reexec_dir = a.out_dir / "r20_reexecution"
    saved_argv = list(sys.argv)
    try:
        sys.argv = [
            "evaluate_rb_r20_real_slate_shadow_integration_v1.py",
            "--full-slate-dir", str(a.full_slate_dir),
            "--r19-model", str(a.r19_model),
            "--r19-pools", str(a.r19_pools),
            "--replay-artifact-metadata", str(a.replay_artifact_metadata),
            "--raw-paid-artifact-metadata", str(a.raw_paid_artifact_metadata),
            "--r19-artifact-metadata", str(a.r19_artifact_metadata),
            "--out-dir", str(reexec_dir),
        ]
        r20_rc = int(r20.main())
    finally:
        sys.argv = saved_argv
        r20.adapt_rb_receiving_tail = original_adapt

    reexec_result_path = reexec_dir / "rb_r20_result.json"
    if not reexec_result_path.is_file():
        raise RuntimeError("R20 reexecution did not emit result")
    reexec_result = json.loads(reexec_result_path.read_text(encoding="utf-8"))
    reexec_pass = bool(r20_rc == 0 and reexec_result.get("pass") is True)
    if not capture:
        raise RuntimeError("R21 failed to capture frozen R20 draw arrays")

    parent_casebook = pd.read_csv(parent_casebook_path, low_memory=False)
    parent_audit = pd.read_csv(parent_player_audit_path, low_memory=False)
    rb = parent_casebook.loc[
        parent_casebook["position_family"].astype(str).str.upper().eq("RB")
        & parent_casebook["adapted_applied"].astype(bool)
    ].copy()
    rb = rb.sort_values(["event_id", "player_clean_key"], kind="mergesort").reset_index(drop=True)

    keys = ["event_id", "player_clean_key"]
    unique_keys = bool(not rb.duplicated(keys).any())
    control_map = capture["control"]
    shadow_map = capture["shadow"]

    control_rows = []
    shadow_rows = []
    index_rows = []
    for i, row in rb.iterrows():
        event_id = str(row["event_id"])
        pkey = str(row["player_clean_key"])
        k = (event_id, pkey, "rec_yards")
        if k not in control_map or k not in shadow_map:
            raise RuntimeError(f"missing captured rec_yards draw key: {k}")
        x0 = np.asarray(control_map[k], dtype="<f8")
        x1 = np.asarray(shadow_map[k], dtype="<f8")
        if len(x0) != EXPECTED_DRAWS or len(x1) != EXPECTED_DRAWS:
            raise RuntimeError(f"draw count drift for {k}: {len(x0)}, {len(x1)}")
        control_rows.append(x0)
        shadow_rows.append(x1)
        index_rows.append({
            "row_index": int(i),
            "season": int(row["season"]),
            "week": int(row["week"]),
            "event_id": event_id,
            "team": str(row["team"]),
            "opponent": str(row.get("opponent", "")),
            "player": str(row.get("player", "")),
            "player_clean_key": pkey,
            "position_family": "RB",
            "identity_top20": float(row["identity_top20"]),
            "state_probability": float(row["state_probability"]),
            "r19_p30": float(row["p30"]),
            "r19_p50": float(row["p50"]),
        })

    control = np.ascontiguousarray(np.vstack(control_rows), dtype="<f8")
    shadow = np.ascontiguousarray(np.vstack(shadow_rows), dtype="<f8")
    shape_exact = bool(control.shape == (EXPECTED_ROWS, EXPECTED_DRAWS) and shadow.shape == control.shape)
    finite_nonnegative = bool(
        np.isfinite(control).all() and np.isfinite(shadow).all()
        and (control >= 0.0).all() and (shadow >= 0.0).all()
    )
    means0 = control.mean(axis=1)
    means1 = shadow.mean(axis=1)
    max_mean_delta = float(np.max(np.abs(means1 - means0)))

    q_levels = [0.05, 0.10, 0.50, 0.75, 0.90, 0.95]
    q0 = np.quantile(control, q_levels, axis=1)
    q1 = np.quantile(shadow, q_levels, axis=1)
    p30_0 = np.mean(control >= (means0[:, None] + 30.0), axis=1)
    p30_1 = np.mean(shadow >= (means0[:, None] + 30.0), axis=1)
    p50_0 = np.mean(control >= (means0[:, None] + 50.0), axis=1)
    p50_1 = np.mean(shadow >= (means0[:, None] + 50.0), axis=1)

    ledger = pd.DataFrame(index_rows)
    ledger["frozen_mean"] = means0
    ledger["shadow_mean"] = means1
    ledger["mean_delta"] = means1 - means0
    for j, label in enumerate(["q05", "q10", "q50", "q75", "q90", "q95"]):
        ledger[f"control_{label}"] = q0[j]
        ledger[f"shadow_{label}"] = q1[j]
    ledger["control_prob_mu_plus_30"] = p30_0
    ledger["shadow_prob_mu_plus_30"] = p30_1
    ledger["control_prob_mu_plus_50"] = p50_0
    ledger["shadow_prob_mu_plus_50"] = p50_1
    ledger["control_draw_sha256"] = [sha256_f64(control[i]) for i in range(len(ledger))]
    ledger["shadow_draw_sha256"] = [sha256_f64(shadow[i]) for i in range(len(ledger))]

    parent_cmp = parent_audit.loc[
        parent_audit["adapted_applied"].astype(bool)
    ].copy()
    parent_cmp["event_id"] = parent_cmp["event_id"].astype(str)
    parent_cmp["player_clean_key"] = parent_cmp["player_clean_key"].astype(str)
    parent_cmp = ledger[keys + [
        "frozen_mean", "shadow_mean", "control_q50", "shadow_q50",
        "control_q75", "shadow_q75", "control_q90", "shadow_q90",
        "control_q95", "shadow_q95", "control_prob_mu_plus_30",
        "shadow_prob_mu_plus_30", "control_prob_mu_plus_50",
        "shadow_prob_mu_plus_50",
    ]].merge(parent_cmp, on=keys, how="left", validate="one_to_one")
    parity_pairs = [
        ("frozen_mean", "canonical_mean"), ("shadow_mean", "adapted_mean"),
        ("control_q50", "canonical_q50"), ("shadow_q50", "adapted_q50"),
        ("control_q75", "canonical_q75"), ("shadow_q75", "adapted_q75"),
        ("control_q90", "canonical_q90"), ("shadow_q90", "adapted_q90"),
        ("control_q95", "canonical_q95"), ("shadow_q95", "adapted_q95"),
        ("control_prob_mu_plus_30", "canonical_prob_mu_plus_30"),
        ("shadow_prob_mu_plus_30", "adapted_prob_mu_plus_30"),
        ("control_prob_mu_plus_50", "canonical_prob_mu_plus_50"),
        ("shadow_prob_mu_plus_50", "adapted_prob_mu_plus_50"),
    ]
    parity_max = 0.0
    parity_complete = bool(len(parent_cmp) == EXPECTED_ROWS and not parent_cmp[[b for _, b in parity_pairs]].isna().any().any())
    if parity_complete:
        parity_max = max(float(np.max(np.abs(parent_cmp[a].to_numpy(float) - parent_cmp[b].to_numpy(float)))) for a, b in parity_pairs)
    parent_draw_summary_parity = bool(parity_complete and parity_max <= 1e-10)

    draws_path = a.out_dir / "rb_r21_locked_rec_yards_draws_v1.npz"
    np.savez_compressed(draws_path, control=control, shadow=shadow)
    ledger_path = a.out_dir / "rb_r21_forecast_ledger.csv"
    ledger.to_csv(ledger_path, index=False)

    control_hash = sha256_f64(control)
    shadow_hash = sha256_f64(shadow)
    inherited_r20_gates = reexec_result.get("gates", {})
    sportsbook_zero = bool(
        parent_result.get("sportsbook_inputs_added") == 0
        and reexec_result.get("sportsbook_inputs_added") == 0
        and inherited_r20_gates.get("sportsbook_zero_to_shadow_scorer") is True
    )
    outcome_zero = bool(
        parent_result.get("current_or_future_outcomes_used") == 0
        and reexec_result.get("current_or_future_outcomes_used") == 0
        and getattr(r20, "HISTORY_THROUGH", None) == 2025
        and inherited_r20_gates.get("future_2026_outcome_zero") is True
    )
    production_zero = bool(
        parent_result.get("production_parameters_changed") == 0
        and reexec_result.get("production_parameters_changed") == 0
        and inherited_r20_gates.get("production_parameters_zero") is True
    )

    gates = {
        "pre_kickoff_lock": pre_kickoff,
        "r20_parent_exact": bool(r20_meta.get("pass")) and parent_pass,
        "r20_reexecution_pass": reexec_pass,
        "r20_parent_draw_summary_parity": parent_draw_summary_parity,
        "rb_shape_exact": shape_exact,
        "player_keys_unique": unique_keys,
        "finite_nonnegative": finite_nonnegative,
        "mean_parity": max_mean_delta <= 1e-8,
        "draw_hashes_present": bool(control_hash and shadow_hash),
        "sportsbook_zero_upstream": sportsbook_zero,
        "outcome_zero": outcome_zero,
        "production_parameters_zero": production_zero,
    }
    passed = bool(all(gates.values()))

    manifest = {
        "candidate": CANDIDATE,
        "plan_commit": PLAN_COMMIT,
        "implementation_git_sha": os.getenv("GITHUB_SHA", ""),
        "disposition": "RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_PASS_SHADOW_ONLY" if passed else "RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_FAIL",
        "pass": passed,
        "season": 2026,
        "week": 1,
        "kickoff_cutoff_utc": KICKOFF_CUTOFF_UTC,
        "github_run_id": int(os.getenv("GITHUB_RUN_ID", "0") or 0),
        "github_run_started_at": current_run.get("run_started_at"),
        "r20_parent": EXPECTED_R20,
        "r20_parent_result_sha256": sha256_text(parent_result_path),
        "r20_parent_casebook_sha256": sha256_text(parent_casebook_path),
        "r20_parent_player_audit_sha256": sha256_text(parent_player_audit_path),
        "r20_reexecution_result_sha256": sha256_text(reexec_result_path),
        "locked_shape": {"rb_rows": int(control.shape[0]), "draws_per_row": int(control.shape[1])},
        "control_matrix_sha256_f64": control_hash,
        "shadow_matrix_sha256_f64": shadow_hash,
        "max_mean_delta": max_mean_delta,
        "parent_summary_parity_max_abs_delta": parity_max,
        "forecast_ledger_sha256": sha256_text(ledger_path),
        "draw_archive_sha256": sha256_text(draws_path),
        "gates": gates,
        "sportsbook_inputs_added": 0,
        "current_or_future_outcomes_used": 0,
        "production_parameters_changed": 0,
        "governance_note": "This artifact seals pre-outcome Week-1 CONTROL/SHADOW RB receiving-yard forecasts only. It does not promote the tail adapter into production.",
    }
    result_path = a.out_dir / "rb_r21_forecast_lock_result.json"
    result_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("\n=== locked ledger sample ===")
    print(ledger[["team", "player", "frozen_mean", "control_q90", "shadow_q90", "r19_p30", "r19_p50"]].head(30).to_string(index=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
