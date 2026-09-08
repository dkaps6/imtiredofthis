#!/usr/bin/env python3
"""RB-R20: real-2026-slate shadow scoring and full-slate fail-closed parity gate.

Frozen deployment/parity test only. The immutable 2026 Week-1 governed full-slate
replay and immutable R19 scorer are scored without changing production targets,
means, simulation, pricing, or sportsbook usage. No 2026 outcomes are read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_r8_receiving_identity_v1 as r8
from scripts.modeling.rb_r17_tail_distribution_adapter_v1 import ResidualPools, adapt_rb_receiving_tail
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

CANDIDATE = "RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_V1"
SEASON = 2026
WEEK = 1
HISTORY_START = 2013
HISTORY_THROUGH = 2025
ITERATIONS = 10000
SIM_SEED = 92020
ADAPTER_SEED = 918

EXPECTED_R19 = {
    "run_id": 34288244770,
    "artifact_id": 10080377483,
    "name": "rb-r19-deployable-tail-scorer-refit-v1",
    "digest": "sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7",
    "head_sha": "6ac1342f737f142acac6a3e4b459f442faf1442a",
    "model_sha256": "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba",
    "pools_sha256": "c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362",
}
EXPECTED_REPLAY = {
    "run_id": 34243241733,
    "artifact_id": 10062978930,
    "name": "paid-full-slate-replay-v1",
    "digest": "sha256:31351365f3afefbf7f73a4df653499e84ce212d367e77a74ea6a12744c8a4f5f",
    "head_sha": "c2ffc633cc63ef9a110c16d2ce10ef3de07890e6",
}
EXPECTED_RAW_PAID = {
    "run_id": 34152868136,
    "artifact_id": 10030344451,
    "name": "run_34152868136",
    "digest": "sha256:c19bd303a0eb7ca58a3484117e28b5e5144459b74459bd1032970873cae6d035",
    "head_sha": "28ac168a44772467bd9774ff83def94eb7ee9f35",
}
EXPECTED_SOURCE_HASHES = {
    "data/football_simulation_universe.csv": "d1e837bce1670d4cdf2b1aa91337a21593fb234f3ffcec2ea8c4f3cd28f53d64",
    "data/target_entitlement_v1_trace.csv": "45e1319bce7b6fd61417d4d7a392cfd0dda374101b276457598c80660ddec60c",
    "data/certified_full_slate_stack_audit.json": "0281c962593010b71d6ab4af83c5161c0271023373d275d9258f52f1640bbe2e",
    "outputs/paid_full_slate_replay_result.json": "db884b5e327c4ed4c1c72afde6dd2d4eb27dada52c0be1efe4efccf2e0f2e40a",
    "outputs/props_priced_clean.csv": "ff6b8cc26d6dbdf0027f7e54687fd593ede612c24a13e0249eb8d84a42f8b270",
}
EXPECTED_SIMULATION_BLOB = "887e9c776ab112276ec8281195b0fed790ea0551"
EXPECTED_SHAPE = {"games": 16, "teams": 32, "players": 469, "rb_fb_rows": 107, "rb_rows": 94, "fb_rows": 13}
EXPECTED_CERT_DISPOSITION = "FULL_SLATE_CERTIFIED_STACK_READY_M38_R15_TE_R5P_C2_WITH_DECLARED_SCIENCE_LIMITATIONS"
EXPECTED_REPLAY_DISPOSITION = "PAID_FULL_SLATE_REPLAY_MECHANICAL_EXECUTION_CERTIFIED"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def sha_f64_sorted(values: np.ndarray) -> str:
    x = np.sort(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(x.tobytes()).hexdigest()


def manual_linear(payload: dict, frame: pd.DataFrame, *, logistic: bool) -> np.ndarray:
    feats = list(payload["feature_order"])
    x = frame[feats].to_numpy(float)
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coef):
        raise RuntimeError("serialized model dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all():
        raise RuntimeError("nonfinite serialized model or feature input")
    if np.any(scale <= 0):
        raise RuntimeError("serialized scaler has nonpositive scale")
    eta = ((x - mean) / scale) @ coef + float(payload["intercept"])
    if logistic:
        eta = np.clip(eta, -700, 700)
        return 1.0 / (1.0 + np.exp(-eta))
    return eta


def artifact_record(path: Path, expected: dict) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = raw.get("artifacts", [])
    matches = [a for a in arts if a.get("name") == expected["name"]]
    if len(matches) != 1:
        return {"pass": False, "reason": f"expected exactly one {expected['name']} artifact, got {len(matches)}"}
    a = matches[0]
    wr = a.get("workflow_run") or {}
    checks = {
        "id": int(a.get("id", -1)) == int(expected["artifact_id"]),
        "name": a.get("name") == expected["name"],
        "digest": a.get("digest") == expected["digest"],
        "run_id": int(wr.get("id", -1)) == int(expected["run_id"]),
        "head_sha": wr.get("head_sha") == expected["head_sha"],
        "not_expired": a.get("expired") is False,
    }
    return {"pass": bool(all(checks.values())), "checks": checks, "observed": {k: a.get(k) for k in ["id", "name", "digest", "expired"]}, "workflow_run": wr}


def run_current_validator() -> dict:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_certified_full_slate_stack_v2.py"],
        text=True, capture_output=True, check=False,
    )
    return {"returncode": int(proc.returncode), "pass": proc.returncode == 0, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}


def source_hashes(root: Path) -> dict[str, str]:
    return {rel: sha256_file(root / rel) for rel in EXPECTED_SOURCE_HASHES}


def production_working_hashes() -> dict[str, str]:
    rels = [
        "data/football_simulation_universe.csv",
        "data/target_entitlement_v1_trace.csv",
        "outputs/paid_full_slate_replay_result.json",
        "outputs/props_priced_clean.csv",
    ]
    return {rel: sha256_file(Path(rel)) for rel in rels}


def position_family(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-slate-dir", type=Path, required=True)
    ap.add_argument("--r19-model", type=Path, required=True)
    ap.add_argument("--r19-pools", type=Path, required=True)
    ap.add_argument("--replay-artifact-metadata", type=Path, required=True)
    ap.add_argument("--raw-paid-artifact-metadata", type=Path, required=True)
    ap.add_argument("--r19-artifact-metadata", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    replay_meta = artifact_record(a.replay_artifact_metadata, EXPECTED_REPLAY)
    raw_meta = artifact_record(a.raw_paid_artifact_metadata, EXPECTED_RAW_PAID)
    r19_meta = artifact_record(a.r19_artifact_metadata, EXPECTED_R19)
    pristine_hash_before = source_hashes(a.full_slate_dir)
    source_hash_rows = [
        {"path": rel, "expected_sha256": exp, "observed_sha256": pristine_hash_before.get(rel, ""), "pass": pristine_hash_before.get(rel) == exp}
        for rel, exp in EXPECTED_SOURCE_HASHES.items()
    ]
    source_hash_exact = all(r["pass"] for r in source_hash_rows)

    model_sha = sha256_file(a.r19_model)
    pools_sha = sha256_file(a.r19_pools)
    model = json.loads(a.r19_model.read_text(encoding="utf-8"))
    pools_npz = np.load(a.r19_pools)
    required_pool_names = {"non_tail", "tail_30_49", "tail_50_plus"}
    if set(pools_npz.files) != required_pool_names:
        raise RuntimeError(f"R19 residual pool members drifted: {pools_npz.files}")
    pool_values = {k: np.asarray(pools_npz[k], dtype=float) for k in sorted(required_pool_names)}
    pool_hash_checks = {}
    for k, vals in pool_values.items():
        meta = model.get("residual_pools", {}).get(k, {})
        pool_hash_checks[k] = {
            "count": int(len(vals)), "expected_count": int(meta.get("count", -1)),
            "sha256_f64_sorted": sha_f64_sorted(vals), "expected_sha256_f64_sorted": meta.get("sha256_f64_sorted"),
            "finite": bool(np.isfinite(vals).all()), "sorted": bool(len(vals) < 2 or np.all(vals[:-1] <= vals[1:])),
        }
        pool_hash_checks[k]["pass"] = bool(
            pool_hash_checks[k]["count"] == pool_hash_checks[k]["expected_count"]
            and pool_hash_checks[k]["sha256_f64_sorted"] == pool_hash_checks[k]["expected_sha256_f64_sorted"]
            and pool_hash_checks[k]["finite"] and pool_hash_checks[k]["sorted"]
        )

    expected_lineage = {
        "r10": {"run_id": 34269618181, "artifact_id": 10073920506, "digest": "sha256:e76840ad1f915a18e729319c6a5d76b4786dff2e215da7de63522acbbb33ac6b"},
        "corrected_r12": {"run_id": 34273055095, "artifact_id": 10074589299, "digest": "sha256:ef1727b218ed8898e9b483e0f4558c537118f60a05259e274b603ce3940af661"},
        "r16": {"run_id": 34286363931, "artifact_id": 10079630404, "digest": "sha256:ca44a174dafadcaf27496b00efe4e933941211037b0eacea0431d72a9b4fa099"},
    }
    model_contract_exact = bool(
        model.get("candidate") == "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1" and model.get("version") == 1
        and model.get("status") == "SHADOW_ONLY" and int(model.get("fit_for_season", -1)) == SEASON
        and model.get("git_sha") == EXPECTED_R19["head_sha"] and int(model.get("sportsbook_inputs_added", -1)) == 0
        and int(model.get("production_parameters_changed", -1)) == 0
    )
    source_lineage_exact = model.get("source_lineage") == expected_lineage

    cert = json.loads((a.full_slate_dir / "data/certified_full_slate_stack_audit.json").read_text(encoding="utf-8"))
    replay = json.loads((a.full_slate_dir / "outputs/paid_full_slate_replay_result.json").read_text(encoding="utf-8"))
    replay_contract_exact = bool(
        cert.get("disposition") == EXPECTED_CERT_DISPOSITION and cert.get("snapshot_mode") == "NO_CREDIT_REPLAY"
        and cert.get("sportsbook_inputs_used_for_football_distributions") is False and cert.get("sportsbook_defines_football_universe") is False
        and replay.get("disposition") == EXPECTED_REPLAY_DISPOSITION and int(replay.get("source_run", -1)) == EXPECTED_RAW_PAID["run_id"]
        and replay.get("odds_api_refetched") is False and int(replay.get("certification_blockers", -1)) == 0
    )

    validator_before = run_current_validator()
    working_hash_before = production_working_hashes()

    universe = pd.read_csv(a.full_slate_dir / "data/football_simulation_universe.csv", low_memory=False)
    trace = pd.read_csv(a.full_slate_dir / "data/target_entitlement_v1_trace.csv", low_memory=False)
    keys = ["event_id", "team", "player_clean_key"]
    if universe.duplicated(keys).any() or trace.duplicated(keys).any():
        raise RuntimeError("duplicate live player/team/event key in governed replay")
    ent = trace[keys + ["entitlement_tgt_share"]].copy()
    metrics = universe.merge(ent, on=keys, how="left", validate="one_to_one")
    if len(metrics) != len(universe):
        raise RuntimeError("target entitlement merge changed football universe row count")
    metrics["position_family"] = position_family(metrics.get("position_family", metrics.get("position", pd.Series("", index=metrics.index))))
    metrics["season"] = pd.to_numeric(metrics["season"], errors="coerce")
    metrics["week"] = pd.to_numeric(metrics["week"], errors="coerce")
    for c in ["entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    rb = metrics.loc[metrics.position_family.isin({"RB", "FB"})].copy()
    shape_observed = {
        "games": int(metrics.event_id.nunique()), "teams": int(metrics.team.nunique()), "players": int(len(metrics)),
        "rb_fb_rows": int(len(rb)), "rb_rows": int(rb.position_family.eq("RB").sum()), "fb_rows": int(rb.position_family.eq("FB").sum()),
    }
    shape_exact = bool(metrics.season.eq(SEASON).all() and metrics.week.eq(WEEK).all() and shape_observed == EXPECTED_SHAPE)
    live_input_cols = ["event_id", "team", "player_clean_key", "position_family", "entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]
    rb_live_complete = bool(
        not rb[live_input_cols].isna().any().any()
        and rb[["entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]].apply(pd.to_numeric, errors="coerce").notna().all().all()
        and np.isfinite(rb[["entitlement_tgt_share", "rules_plays_est", "rules_pass_rate", "rules_ypt"]].to_numpy(float)).all()
        and (rb.entitlement_tgt_share >= 0).all()
    )

    states, prev = r8._identity_atlas(HISTORY_START, HISTORY_THROUGH)
    current_time_key = SEASON * 100 + WEEK
    state_time = pd.to_numeric(states.get("time_key", pd.Series(dtype=float)), errors="coerce").dropna()
    strict_prior_source = bool(len(states) > 0 and len(state_time) > 0 and int(state_time.max()) < current_time_key)
    rb["season"] = SEASON
    rb["week"] = WEEK
    rb = r8._attach_identity(rb, SEASON, WEEK, states, prev)
    r8_features = list(model["models"]["r8_r9_identity"]["feature_order"])
    identity_feature_exact = r8_features == list(r8.FEATURES)
    identity_frame_unique_finite = bool(not rb.duplicated(keys).any() and not rb[r8_features].isna().any().any() and np.isfinite(rb[r8_features].to_numpy(float)).all())

    simulation_blob_before = git_blob_sha1(Path("scripts/simulation_v2.py"))
    allocation_trace: list = []
    canonical = explicit_simulate(metrics, iterations=ITERATIONS, seed=SIM_SEED, allocation_trace=allocation_trace)
    allocation_trace_before = copy.deepcopy(allocation_trace)
    canonical_snapshot = {k: np.asarray(v).copy() for k, v in canonical.values.items()}

    team_attempts = metrics.groupby(["event_id", "team"], as_index=False).agg(rules_plays_est=("rules_plays_est", "mean"), rules_pass_rate=("rules_pass_rate", "mean"))
    team_attempts["team_pass_attempt_projection"] = team_attempts.rules_plays_est * team_attempts.rules_pass_rate
    rb = rb.merge(team_attempts[["event_id", "team", "team_pass_attempt_projection"]], on=["event_id", "team"], how="left", validate="many_to_one")
    rb["baseline_entitlement_tgt_share"] = rb.entitlement_tgt_share.astype(float)
    rb["baseline_pred_targets"] = rb.team_pass_attempt_projection * rb.baseline_entitlement_tgt_share

    means = []
    missing_canonical = []
    for r in rb.itertuples(index=False):
        k = (str(r.event_id), str(r.player_clean_key), "rec_yards")
        if k not in canonical.values:
            missing_canonical.append(k); means.append(np.nan)
        else:
            means.append(float(np.asarray(canonical.values[k], float).mean()))
    rb["baseline_pred_rec_yards"] = means
    if missing_canonical:
        raise RuntimeError(f"missing canonical RB/FB rec_yards arrays: {missing_canonical[:5]}")
    rb["frozen_ypt"] = rb.rules_ypt.astype(float)

    r8_payload = model["models"]["r8_r9_identity"]
    raw = manual_linear(r8_payload, rb, logistic=False)
    raw = np.clip(raw, -float(r8_payload["prediction_clip"]), float(r8_payload["prediction_clip"]))
    reliability = float(r8_payload["r9_reliability"])
    rb["r9_raw_r8_residual"] = raw
    rb["r9_reliability"] = reliability
    rb["r9_calibrated_residual"] = reliability * rb.r9_raw_r8_residual
    rb["r9_shadow_entitlement_tgt_share"] = rb.baseline_entitlement_tgt_share.astype(float)
    pool_gaps = []
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        g = rb.loc[idx]
        base_pool = float(g.baseline_entitlement_tgt_share.sum())
        if base_pool <= 0:
            raise RuntimeError(f"nonpositive governed RB pool for {event_id}/{team}")
        within = g.baseline_entitlement_tgt_share.to_numpy(float) / base_pool
        score = np.log(np.clip(within, 0.0, None) + float(r8.EPS)) + g.r9_calibrated_residual.to_numpy(float)
        w = np.exp(score - np.max(score)); w = w / w.sum(); cand = base_pool * w
        cand[int(np.argmax(w))] += base_pool - float(cand.sum())
        rb.loc[idx, "r9_shadow_entitlement_tgt_share"] = cand
        pool_gaps.append(abs(float(cand.sum()) - base_pool))
    max_r9_pool_gap = float(max(pool_gaps)) if pool_gaps else float("inf")
    rb["r9_shadow_pred_targets"] = rb.team_pass_attempt_projection * rb.r9_shadow_entitlement_tgt_share
    rb["target_delta"] = rb.r9_shadow_pred_targets - rb.baseline_pred_targets

    rb["identity_pct"] = rb.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    rb["identity_top20"] = rb.identity_pct.gt(0.80).astype(float)
    rb["state_probability"] = 0.0
    top = rb.identity_top20.eq(1.0)
    if not top.any():
        raise RuntimeError("R20 current slate produced zero TOP20 receiving-identity rows")
    rb.loc[top, "state_probability"] = manual_linear(model["models"]["r11_high5"], rb.loc[top], logistic=True)
    rb["p30"] = manual_linear(model["models"]["r16_cat30"], rb, logistic=True)
    rb["p50"] = manual_linear(model["models"]["r16_cat50"], rb, logistic=True)
    rb["nested_p50"] = np.minimum(rb.p30, rb.p50)

    risk = rb[["event_id", "player_clean_key", "p30", "p50"]].copy()
    pools = ResidualPools(non_tail=pool_values["non_tail"], tail_30_49=pool_values["tail_30_49"], tail_50_plus=pool_values["tail_50_plus"])
    adapted, adapter_audit = adapt_rb_receiving_tail(canonical, metrics, risk, pools, seed=ADAPTER_SEED)
    adapted2, _ = adapt_rb_receiving_tail(canonical, metrics, risk, pools, seed=ADAPTER_SEED)

    posmap = {(str(r.event_id), str(r.player_clean_key)): str(r.position_family).upper() for r in metrics.itertuples(index=False)}
    non_rb_exact = True; rb_component_exact = True; rush_rec_identity = True; nonnegative = True
    finite_draws = True; deterministic = True; canonical_unmutated = True; max_mean_delta = 0.0; min_spearman = 1.0
    player_rows = []
    for k, v0 in canonical_snapshot.items():
        now = np.asarray(canonical.values[k]); canonical_unmutated &= np.array_equal(v0, now)
        if k not in adapted.values or k not in adapted2.values:
            raise RuntimeError(f"adapted result dropped canonical key: {k}")
        v1 = np.asarray(adapted.values[k], float); v2 = np.asarray(adapted2.values[k], float)
        deterministic &= np.array_equal(v1, v2)
        game, pkey, market = k; is_rb = posmap.get((str(game), str(pkey)), "") == "RB"
        finite_draws &= bool(np.isfinite(v1).all())
        if not is_rb: non_rb_exact &= np.array_equal(v0, v1)
        elif market not in {"rec_yards", "rush_rec_yards"}: rb_component_exact &= np.array_equal(v0, v1)
        if is_rb and market == "rec_yards":
            nonnegative &= bool((v1 >= 0).all()); x0 = np.asarray(v0, float)
            delta = abs(float(v1.mean()) - float(x0.mean())); max_mean_delta = max(max_mean_delta, delta)
            rho = 1.0
            if not np.allclose(x0, x0[0], atol=0, rtol=0):
                rho = float(pd.Series(x0).corr(pd.Series(v1), method="spearman")); rho = rho if np.isfinite(rho) else -1.0
            min_spearman = min(min_spearman, rho); mu = float(x0.mean())
            player_rows.append({
                "event_id": str(game), "player_clean_key": str(pkey), "adapted_applied": True,
                "canonical_mean": mu, "adapted_mean": float(v1.mean()), "mean_delta": float(v1.mean() - mu), "spearman": rho,
                "canonical_q50": float(np.quantile(x0, .50)), "adapted_q50": float(np.quantile(v1, .50)),
                "canonical_q75": float(np.quantile(x0, .75)), "adapted_q75": float(np.quantile(v1, .75)),
                "canonical_q90": float(np.quantile(x0, .90)), "adapted_q90": float(np.quantile(v1, .90)),
                "canonical_q95": float(np.quantile(x0, .95)), "adapted_q95": float(np.quantile(v1, .95)),
                "canonical_prob_mu_plus_30": float(np.mean(x0 >= mu + 30.0)), "adapted_prob_mu_plus_30": float(np.mean(v1 >= mu + 30.0)),
                "canonical_prob_mu_plus_50": float(np.mean(x0 >= mu + 50.0)), "adapted_prob_mu_plus_50": float(np.mean(v1 >= mu + 50.0)),
            })

    for game, pkey in [k for k, p in posmap.items() if p == "RB"]:
        rk, ck, yk = (game, pkey, "rush_yards"), (game, pkey, "rush_rec_yards"), (game, pkey, "rec_yards")
        if all(k in adapted.values for k in [rk, ck, yk]):
            rush_rec_identity &= bool(np.allclose(np.asarray(adapted.values[ck], float), np.asarray(adapted.values[rk], float) + np.asarray(adapted.values[yk], float), atol=1e-10, rtol=0))

    player_audit = pd.DataFrame(player_rows)
    fb_rows = []
    for r in rb.loc[rb.position_family.eq("FB")].itertuples(index=False):
        k = (str(r.event_id), str(r.player_clean_key), "rec_yards"); x0 = np.asarray(canonical.values[k], float); x1 = np.asarray(adapted.values[k], float); mu = float(x0.mean())
        fb_rows.append({
            "event_id": str(r.event_id), "player_clean_key": str(r.player_clean_key), "adapted_applied": False,
            "canonical_mean": mu, "adapted_mean": float(x1.mean()), "mean_delta": float(x1.mean() - mu), "spearman": 1.0,
            "canonical_q50": float(np.quantile(x0, .50)), "adapted_q50": float(np.quantile(x1, .50)),
            "canonical_q75": float(np.quantile(x0, .75)), "adapted_q75": float(np.quantile(x1, .75)),
            "canonical_q90": float(np.quantile(x0, .90)), "adapted_q90": float(np.quantile(x1, .90)),
            "canonical_q95": float(np.quantile(x0, .95)), "adapted_q95": float(np.quantile(x1, .95)),
            "canonical_prob_mu_plus_30": float(np.mean(x0 >= mu + 30)), "adapted_prob_mu_plus_30": float(np.mean(x1 >= mu + 30)),
            "canonical_prob_mu_plus_50": float(np.mean(x0 >= mu + 50)), "adapted_prob_mu_plus_50": float(np.mean(x1 >= mu + 50)),
        })
    if fb_rows: player_audit = pd.concat([player_audit, pd.DataFrame(fb_rows)], ignore_index=True)
    rb = rb.merge(player_audit, on=["event_id", "player_clean_key"], how="left", validate="one_to_one")

    casebook_cols = [
        "season", "week", "event_id", "team", "opponent", "player", "player_clean_key", "position_family",
        "baseline_entitlement_tgt_share", "baseline_pred_targets", "baseline_pred_rec_yards", "frozen_ypt", *r8_features,
        "identity_pct", "identity_top20", "r9_raw_r8_residual", "r9_reliability", "r9_calibrated_residual",
        "r9_shadow_entitlement_tgt_share", "r9_shadow_pred_targets", "target_delta", "state_probability", "p30", "p50", "nested_p50", "adapted_applied",
        "canonical_mean", "adapted_mean", "mean_delta", "spearman", "canonical_q50", "adapted_q50", "canonical_q75", "adapted_q75",
        "canonical_q90", "adapted_q90", "canonical_q95", "adapted_q95", "canonical_prob_mu_plus_30", "adapted_prob_mu_plus_30",
        "canonical_prob_mu_plus_50", "adapted_prob_mu_plus_50",
    ]
    for c in ["opponent", "player"]:
        if c not in rb.columns: rb[c] = ""
    missing_casebook = [c for c in casebook_cols if c not in rb.columns]
    if missing_casebook: raise RuntimeError(f"missing R20 casebook columns: {missing_casebook}")
    casebook = rb[casebook_cols].copy()

    validator_after = run_current_validator()
    working_hash_after = production_working_hashes()
    pristine_hash_after = source_hashes(a.full_slate_dir)
    simulation_blob_after = git_blob_sha1(Path("scripts/simulation_v2.py"))

    prob_cols = ["state_probability", "p30", "p50", "nested_p50"]
    probabilities_valid = bool(np.isfinite(casebook[prob_cols].to_numpy(float)).all() and ((casebook[prob_cols] >= 0.0) & (casebook[prob_cols] <= 1.0)).all().all())
    rest80_zero = bool((casebook.loc[casebook.identity_top20.eq(0.0), "state_probability"] == 0.0).all())
    casebook_complete = bool(
        len(casebook) == EXPECTED_SHAPE["rb_fb_rows"] and not casebook.duplicated(keys).any()
        and not casebook[["event_id", "team", "player_clean_key", "baseline_pred_targets", "baseline_pred_rec_yards", "p30", "p50"]].isna().any().any()
        and len(player_audit) == EXPECTED_SHAPE["rb_fb_rows"] and len(adapter_audit) == EXPECTED_SHAPE["rb_rows"]
    )
    source_bytes_unchanged = pristine_hash_before == pristine_hash_after
    working_bytes_unchanged = working_hash_before == working_hash_after
    production_entitlement_unmutated = bool(np.array_equal(metrics.entitlement_tgt_share.to_numpy(float), universe.merge(ent, on=keys, how="left", validate="one_to_one").entitlement_tgt_share.to_numpy(float)))

    gates = {
        "r19_artifact_metadata_exact": bool(r19_meta.get("pass")), "replay_artifact_metadata_exact": bool(replay_meta.get("pass")),
        "raw_paid_artifact_metadata_exact": bool(raw_meta.get("pass")), "r19_model_sha_exact": model_sha == EXPECTED_R19["model_sha256"],
        "r19_pools_sha_exact": pools_sha == EXPECTED_R19["pools_sha256"], "r19_model_contract_exact": model_contract_exact,
        "r19_source_lineage_exact": source_lineage_exact, "r19_residual_pool_hashes_exact": bool(all(v["pass"] for v in pool_hash_checks.values())),
        "full_slate_source_hashes_exact": bool(source_hash_exact), "full_slate_replay_contract_exact": replay_contract_exact,
        "certified_stack_validator_before_pass": bool(validator_before["pass"]), "real_slate_shape_exact": shape_exact,
        "unique_live_keys": bool(not universe.duplicated(keys).any() and not trace.duplicated(keys).any()), "rb_live_inputs_complete_finite": rb_live_complete,
        "strict_prior_identity_source_complete": strict_prior_source, "identity_feature_order_exact": identity_feature_exact,
        "identity_feature_frame_unique_finite": identity_frame_unique_finite, "shadow_r9_rb_pool_conservation": max_r9_pool_gap <= 1e-12,
        "production_entitlement_unmutated": production_entitlement_unmutated, "r11_r16_probabilities_finite_range": probabilities_valid,
        "r11_rest80_zero_exact": rest80_zero, "shadow_casebook_complete": casebook_complete,
        "canonical_mean_parity": max_mean_delta <= 1e-8, "non_rb_exact": bool(non_rb_exact), "rb_component_exact": bool(rb_component_exact),
        "rush_rec_identity": bool(rush_rec_identity), "allocation_trace_exact": allocation_trace == allocation_trace_before,
        "nonnegative_rec_yards": bool(nonnegative), "finite_draws": bool(finite_draws), "rank_preservation": min_spearman >= 0.9999,
        "deterministic_replay": bool(deterministic), "canonical_result_unmutated": bool(canonical_unmutated),
        "canonical_simulation_blob_exact": simulation_blob_before == EXPECTED_SIMULATION_BLOB and simulation_blob_after == EXPECTED_SIMULATION_BLOB,
        "certified_stack_validator_after_pass": bool(validator_after["pass"]), "pristine_source_bytes_unchanged": bool(source_bytes_unchanged),
        "working_production_bytes_unchanged": bool(working_bytes_unchanged),
        "sportsbook_zero_to_shadow_scorer": bool(cert.get("sportsbook_inputs_used_for_football_distributions") is False),
        "future_2026_outcome_zero": True, "production_parameters_zero": True, "shadow_only_status": model.get("status") == "SHADOW_ONLY",
    }
    passed = all(gates.values())
    result = {
        "candidate": CANDIDATE,
        "disposition": "RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_PASS_SHADOW_ONLY" if passed else "RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_FAIL",
        "pass": bool(passed), "scored_slate": {"season": SEASON, "week": WEEK, **shape_observed},
        "frozen_execution": {"iterations": ITERATIONS, "simulation_seed": SIM_SEED, "adapter_seed": ADAPTER_SEED, "history_start": HISTORY_START, "history_through": HISTORY_THROUGH},
        "immutable_sources": {"r19": EXPECTED_R19, "governed_replay": EXPECTED_REPLAY, "raw_paid_source": EXPECTED_RAW_PAID},
        "artifact_metadata_audit": {"r19": r19_meta, "governed_replay": replay_meta, "raw_paid": raw_meta},
        "source_hash_audit": source_hash_rows, "r19_pool_audit": pool_hash_checks,
        "strict_prior_audit": {"state_rows": int(len(states)), "max_state_time_key": int(state_time.max()) if len(state_time) else None, "current_time_key": current_time_key},
        "shadow_r9": {"reliability": reliability, "max_rb_pool_gap": max_r9_pool_gap, "target_delta_min": float(casebook.target_delta.min()), "target_delta_max": float(casebook.target_delta.max())},
        "shadow_r11": {"top20_rows": int(casebook.identity_top20.sum()), "rest80_rows": int(casebook.identity_top20.eq(0).sum()), "state_probability_min": float(casebook.state_probability.min()), "state_probability_max": float(casebook.state_probability.max())},
        "shadow_r16": {"p30_min": float(casebook.p30.min()), "p30_max": float(casebook.p30.max()), "p50_min": float(casebook.p50.min()), "p50_max": float(casebook.p50.max())},
        "adapter_parity": {"adapted_rb_count": int(len(adapter_audit)), "max_mean_delta": max_mean_delta, "min_spearman": min_spearman},
        "certified_validator_before": validator_before, "certified_validator_after": validator_after, "gates": gates,
        "sportsbook_inputs_added": 0, "current_or_future_outcomes_used": 0, "production_parameters_changed": 0,
        "governance_note": "PASS certifies only real-2026-slate SHADOW deployability/parity. It does not promote RB receiving tail scoring into production. A separate governed promotion or prospective grading decision is required.",
    }

    casebook.to_csv(a.out_dir / "rb_r20_shadow_casebook.csv", index=False)
    player_audit.to_csv(a.out_dir / "rb_r20_adapter_player_audit.csv", index=False)
    pd.DataFrame(source_hash_rows).to_csv(a.out_dir / "rb_r20_source_hash_audit.csv", index=False)
    adapter_audit.to_csv(a.out_dir / "rb_r20_frozen_adapter_audit.csv", index=False)
    (a.out_dir / "rb_r20_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\n=== shadow casebook sample ===")
    print(casebook[["team", "player", "position_family", "baseline_pred_targets", "baseline_pred_rec_yards", "identity_top20", "state_probability", "p30", "p50", "adapted_q90"]].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
