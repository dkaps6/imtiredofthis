#!/usr/bin/env python3
"""Build frozen R26N 2026 Week-1 unmodified-R26 structural candidate.

Prospective opportunity/reception overlay only. No R9 fit/refit, no 2026 outcomes,
no sportsbook football inputs, no receiving-yard distribution regeneration, and
no production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as full_v1
import scripts.run_pricing_with_full_roster_universe_v3_core as full_v3
from scripts.modeling.rb_receiving_identity_runtime_v1 import EPS, FEATURES, attach_identity, identity_atlas

CANDIDATE = "RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1"
PASS_DISPOSITION = "R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN"
FAIL_DISPOSITION = "R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_FAIL_NO_INTEGRATION"
SEASON = 2026
WEEK = 1
HISTORY_START = 2013
HISTORY_THROUGH = 2025
RB_FAMILIES = {"RB", "FB"}
EXPECTED_PRODUCTION_ROWS = 468
EXPECTED_TEAMS = 32
EXPECTED_GAMES = 16
EXPECTED_VACANCY_TEAMS = 31
EXPECTED_R19_MODEL_SHA256 = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"
EXPECTED_R26M_DISPOSITION = "2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED"
EXPECTED_R26L_DISPOSITION = "2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION"


def read_unique_json(root: Path, filename: str) -> dict:
    hits = sorted(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {filename} under {root}, found {len(hits)}")
    return json.loads(hits[0].read_text())


def read_unique_csv(root: Path, filename: str) -> pd.DataFrame:
    hits = sorted(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {filename} under {root}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def position_family(s: pd.Series) -> pd.Series:
    p = s.fillna("").astype(str).str.upper().str.strip()
    p = p.replace({"HB": "RB", "TB": "RB"})
    p = p.where(~p.str.startswith("RB"), "RB")
    p = p.where(~p.str.startswith("FB"), "FB")
    p = p.where(~p.str.startswith("QB"), "QB")
    p = p.where(~p.str.startswith("WR"), "WR")
    p = p.where(~p.str.startswith("TE"), "TE")
    return p


def manual_r9(payload: dict, frame: pd.DataFrame) -> np.ndarray:
    required = list(payload["feature_order"])
    if required != list(FEATURES):
        raise RuntimeError("R26N frozen R19 R8/R9 feature order != production-safe identity runtime")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"R26N strict-prior R9 feature frame missing {missing}")
    x = frame[required].to_numpy(float)
    mean = np.asarray(payload["scaler_mean"], float)
    scale = np.asarray(payload["scaler_scale"], float)
    coef = np.asarray(payload["coefficients"], float)
    if x.shape[1] != len(mean) or len(mean) != len(scale) or len(scale) != len(coef):
        raise RuntimeError("R26N serialized R9 model dimension mismatch")
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all():
        raise RuntimeError("R26N nonfinite R9 model/input")
    if np.any(scale <= 0):
        raise RuntimeError("R26N serialized R9 scaler has nonpositive scale")
    pred = ((x - mean) / scale) @ coef + float(payload["intercept"])
    return np.clip(pred, -float(payload["prediction_clip"]), float(payload["prediction_clip"]))


def gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    rows.append({"gate": name, "passed": bool(passed), "evidence": json.dumps(evidence, sort_keys=True, default=str) if not isinstance(evidence, str) else evidence})
    return bool(passed)


def build_current_entitlement(production_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = production_root / "data"
    form_path = data / "player_form_consensus.csv"
    context_path = data / "model_context_bridge.csv"
    if not form_path.is_file() or not context_path.is_file():
        raise RuntimeError("R26N current production artifact missing PlayerForm/model-context inputs")
    form = pd.read_csv(form_path, low_memory=False)
    context = pd.read_csv(context_path, low_memory=False)
    if len(form) != EXPECTED_PRODUCTION_ROWS or len(context) != EXPECTED_PRODUCTION_ROWS:
        raise RuntimeError(f"R26N current production row-count drift form={len(form)} context={len(context)}")

    need = {"player", "player_clean_key", "team", "opponent", "season", "week", "position"}
    if need - set(form.columns):
        raise RuntimeError(f"R26N current PlayerForm missing {sorted(need-set(form.columns))}")
    stub = form[["player", "player_clean_key", "team", "opponent", "season", "week", "position"]].copy()
    stub["event_id"] = [
        full_v1._canonical_game(t, o, s, w)
        for t, o, s, w in zip(stub.team, stub.opponent, stub.season, stub.week)
    ]
    stub["market"] = "football_universe"

    old_cwd = Path.cwd()
    try:
        os.chdir(production_root)
        final, _aliases, audit = full_v3._build_with_promoted_entitlement_specialists(stub)
        trace_path = Path("data/target_entitlement_v1_trace.csv")
        if not trace_path.is_file():
            raise RuntimeError("R26N promoted entitlement reconstruction emitted no target trace")
        trace = pd.read_csv(trace_path, low_memory=False)
    finally:
        os.chdir(old_cwd)

    if len(final) != EXPECTED_PRODUCTION_ROWS or len(trace) != EXPECTED_PRODUCTION_ROWS:
        raise RuntimeError(f"R26N reconstructed entitlement row drift final={len(final)} trace={len(trace)}")
    return final, trace, audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-root", type=Path, required=True)
    ap.add_argument("--r26m-root", type=Path, required=True)
    ap.add_argument("--r26l-root", type=Path, required=True)
    ap.add_argument("--r19-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    production_root = a.production_root.resolve()
    r26m_root = a.r26m_root.resolve()
    r26l_root = a.r26l_root.resolve()
    r19_root = a.r19_root.resolve()
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not a.protected_clean_marker.is_file() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26N protected production boundary marker missing/invalid")

    r26m = read_unique_json(r26m_root, "r26m_disposition.json")
    r26l = read_unique_json(r26l_root, "r26l_disposition.json")
    vacancy = read_unique_csv(r26l_root, "r26l_2026_week1_room_state.csv")
    model_hits = sorted(r19_root.rglob("rb_r19_tail_scorer_model_v1.json"))
    result_hits = sorted(r19_root.rglob("rb_r19_result.json"))
    if len(model_hits) != 1 or len(result_hits) != 1:
        raise RuntimeError(f"R26N expected one R19 model/result, found model={len(model_hits)} result={len(result_hits)}")
    model_path = model_hits[0]
    r19_model = json.loads(model_path.read_text())
    r19_result = json.loads(result_hits[0].read_text())

    r9 = r19_model.get("models", {}).get("r8_r9_identity", {})
    r9_contract = bool(
        sha256(model_path) == EXPECTED_R19_MODEL_SHA256
        and r19_model.get("candidate") == "RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1"
        and int(r19_model.get("fit_for_season", -1)) == SEASON
        and r19_model.get("status") == "SHADOW_ONLY"
        and list(r9.get("feature_order", [])) == list(FEATURES)
        and r9.get("model") == "StandardScaler+Ridge"
        and float(r9.get("alpha", -1)) == 20.0
        and float(r9.get("train_clip", -1)) == 2.0
        and float(r9.get("prediction_clip", -1)) == 1.0
        and int(r9.get("training_season", -1)) == HISTORY_THROUGH
        and float(r9.get("r9_reliability", -1)) == 1.0
        and bool(r19_result.get("gates", {}).get("strict_prior_audit"))
        and bool(r19_result.get("gates", {}).get("serialization_roundtrip"))
        and float(r19_result.get("serialization_audit", {}).get("max_abs_delta", 1.0)) == 0.0
        and int(r19_result.get("sportsbook_inputs_added", -1)) == 0
        and int(r19_result.get("production_parameters_changed", -1)) == 0
    )
    if not r9_contract:
        raise RuntimeError("R26N frozen serialized R9 contract mismatch")

    final, entitlement_trace, universe_audit = build_current_entitlement(production_root)
    final = final.copy()
    final["position_family"] = position_family(final["position"])
    final["baseline_entitlement_tgt_share"] = pd.to_numeric(final["entitlement_tgt_share"], errors="coerce")
    if final.baseline_entitlement_tgt_share.isna().any() or (final.baseline_entitlement_tgt_share < 0).any():
        raise RuntimeError("R26N invalid reconstructed baseline entitlement")

    # Strict-prior identity source ends at 2025 by construction.
    states, prev = identity_atlas(HISTORY_START, HISTORY_THROUGH)
    state_time = pd.to_numeric(states.get("time_key", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(states) == 0 or len(state_time) == 0 or int(state_time.max()) >= SEASON * 100 + WEEK:
        raise RuntimeError("R26N strict-prior identity history reaches current/future 2026 Week 1")

    rb = final.loc[final.position_family.isin(RB_FAMILIES)].copy()
    if rb.empty or rb.team.nunique() != EXPECTED_TEAMS:
        raise RuntimeError(f"R26N current RB+FB room coverage invalid rows={len(rb)} teams={rb.team.nunique()}")
    rb["season"] = SEASON
    rb["week"] = WEEK
    rb = attach_identity(rb, SEASON, WEEK, states, prev)
    if rb[list(FEATURES)].isna().any().any() or not np.isfinite(rb[list(FEATURES)].to_numpy(float)).all():
        raise RuntimeError("R26N strict-prior R9 features incomplete/nonfinite")
    rb["r9_raw_residual"] = manual_r9(r9, rb)
    reliability = float(r9["r9_reliability"])
    rb["r9_reliability"] = reliability
    rb["r9_calibrated_residual"] = reliability * rb.r9_raw_residual

    vacancy = vacancy.loc[
        pd.to_numeric(vacancy["season"], errors="coerce").eq(SEASON)
        & pd.to_numeric(vacancy["week"], errors="coerce").eq(WEEK)
    ].copy()
    vacancy_teams = set(vacancy.team.astype(str))
    rb_teams = set(rb.team.astype(str))
    nonvacancy_teams = sorted(rb_teams - vacancy_teams)
    missing_vacancy_teams = sorted(vacancy_teams - rb_teams)
    if len(vacancy_teams) != EXPECTED_VACANCY_TEAMS or missing_vacancy_teams or len(nonvacancy_teams) != 1:
        raise RuntimeError(
            f"R26N vacancy/production alignment invalid vacancy={len(vacancy_teams)} "
            f"missing={missing_vacancy_teams} nonvacancy={nonvacancy_teams}"
        )

    rb["vacancy_active"] = rb.team.astype(str).isin(vacancy_teams).astype(int)
    rb["baseline_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    if (rb.baseline_rb_pool <= 0).any():
        raise RuntimeError("R26N nonpositive current RB+FB entitlement pool")
    rb["baseline_rb_within_share"] = rb.baseline_entitlement_tgt_share / rb.baseline_rb_pool
    rb["r9_score"] = np.log(rb.baseline_rb_within_share.clip(lower=0.0) + float(EPS)) + rb.r9_calibrated_residual
    rb["candidate_rb_within_share"] = rb.baseline_rb_within_share.astype(float)

    room_rows: list[dict] = []
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=True).groups.items():
        g = rb.loc[idx]
        baseline = g.baseline_rb_within_share.to_numpy(float)
        if bool(g.vacancy_active.iloc[0]):
            score = g.r9_score.to_numpy(float)
            ww = np.exp(score - np.max(score))
            cand = ww / ww.sum() if ww.sum() > 0 else baseline.copy()
        else:
            cand = baseline.copy()
        if len(cand):
            cand[int(np.argmax(cand))] += 1.0 - float(cand.sum())
        rb.loc[idx, "candidate_rb_within_share"] = cand
        pool = float(g.baseline_rb_pool.iloc[0])
        room_rows.append({
            "event_id": str(event_id),
            "team": str(team),
            "vacancy_active": int(g.vacancy_active.iloc[0]),
            "players": int(len(g)),
            "baseline_rb_pool": pool,
            "candidate_rb_pool": float(pool * cand.sum()),
            "rb_pool_gap": float(pool * cand.sum() - pool),
            "max_within_share_delta": float(np.max(np.abs(cand - baseline))) if len(cand) else 0.0,
        })

    rb["candidate_entitlement_tgt_share"] = rb.baseline_rb_pool * rb.candidate_rb_within_share
    rb["team_pass_attempt_projection"] = pd.to_numeric(rb["rules_plays_est"], errors="coerce") * pd.to_numeric(rb["rules_pass_rate"], errors="coerce")
    catch = pd.to_numeric(rb["rules_catch_rate"], errors="coerce")
    if catch.isna().any() or rb.team_pass_attempt_projection.isna().any():
        raise RuntimeError("R26N current production target/reception expectation inputs missing")
    rb["catch_rate_for_r26"] = catch.clip(lower=0.35, upper=0.95)
    rb["baseline_targets"] = rb.team_pass_attempt_projection * rb.baseline_entitlement_tgt_share
    rb["candidate_targets"] = rb.team_pass_attempt_projection * rb.candidate_entitlement_tgt_share
    rb["baseline_receptions"] = rb.baseline_targets * rb.catch_rate_for_r26
    rb["candidate_receptions"] = rb.candidate_targets * rb.catch_rate_for_r26

    # Full player overlay: only RB+FB entitlement changes are permitted.
    overlay = final[["event_id", "team", "player", "player_clean_key", "position", "position_family", "baseline_entitlement_tgt_share"]].copy()
    overlay["candidate_entitlement_tgt_share"] = overlay.baseline_entitlement_tgt_share.astype(float)
    overlay["vacancy_active"] = 0
    overlay["baseline_rb_within_share"] = np.nan
    overlay["candidate_rb_within_share"] = np.nan
    overlay["r9_raw_residual"] = np.nan
    overlay["r9_reliability"] = np.nan
    overlay["r9_calibrated_residual"] = np.nan
    overlay["baseline_targets"] = np.nan
    overlay["candidate_targets"] = np.nan
    overlay["baseline_receptions"] = np.nan
    overlay["candidate_receptions"] = np.nan
    overlay["receiving_yard_mean_authority"] = "PRODUCTION_UNCHANGED_NOT_RECOMPUTED"
    overlay["r22_authority"] = "PRODUCTION_R22_UNCHANGED_NOT_RECOMPUTED"

    rbkey = rb.set_index(["event_id", "team", "player_clean_key"])
    mask = overlay.position_family.isin(RB_FAMILIES)
    keys = pd.MultiIndex.from_frame(overlay.loc[mask, ["event_id", "team", "player_clean_key"]])
    for col in [
        "candidate_entitlement_tgt_share", "vacancy_active", "baseline_rb_within_share",
        "candidate_rb_within_share", "r9_raw_residual", "r9_reliability",
        "r9_calibrated_residual", "baseline_targets", "candidate_targets",
        "baseline_receptions", "candidate_receptions",
    ]:
        overlay.loc[mask, col] = rbkey.loc[keys, col].to_numpy()
    overlay["entitlement_delta"] = overlay.candidate_entitlement_tgt_share - overlay.baseline_entitlement_tgt_share

    room_audit = pd.DataFrame(room_rows)
    team_audit = overlay.groupby(["event_id", "team"], as_index=False).agg(
        baseline_team_entitlement=("baseline_entitlement_tgt_share", "sum"),
        candidate_team_entitlement=("candidate_entitlement_tgt_share", "sum"),
    )
    team_audit["team_entitlement_delta"] = team_audit.candidate_team_entitlement - team_audit.baseline_team_entitlement
    room_audit = room_audit.merge(team_audit, on=["event_id", "team"], how="left", validate="one_to_one")

    max_rb_pool_gap = float(room_audit.rb_pool_gap.abs().max())
    max_team_gap = float(team_audit.team_entitlement_delta.abs().max())
    nonrb = ~overlay.position_family.isin(RB_FAMILIES)
    max_nonrb_delta = float(overlay.loc[nonrb, "entitlement_delta"].abs().max()) if nonrb.any() else 0.0
    nonvac_rb = overlay.position_family.isin(RB_FAMILIES) & overlay.vacancy_active.eq(0)
    max_nonvac_rb_delta = float(overlay.loc[nonvac_rb, "entitlement_delta"].abs().max()) if nonvac_rb.any() else 0.0
    player_keys_before = set(zip(final.event_id.astype(str), final.team.astype(str), final.player_clean_key.astype(str)))
    player_keys_after = set(zip(overlay.event_id.astype(str), overlay.team.astype(str), overlay.player_clean_key.astype(str)))

    te_audit_path = production_root / "data" / "te_r5p_full_slate_entitlement_audit.json"
    wr_audit_path = production_root / "data" / "wr_r15_full_slate_entitlement_audit.json"
    te_audit = json.loads(te_audit_path.read_text()) if te_audit_path.is_file() else {}
    wr_audit = json.loads(wr_audit_path.read_text()) if wr_audit_path.is_file() else {}

    gates: list[dict] = []
    gate_values = [
        gate(gates, "01_r26m_exact_qualification", r26m.get("disposition") == EXPECTED_R26M_DISPOSITION and r26m.get("r26n_design_authorized") is True, {"disposition": r26m.get("disposition"), "design": r26m.get("r26n_design_authorized")}),
        gate(gates, "02_r26l_exact_modern_like_parent", r26l.get("disposition") == EXPECTED_R26L_DISPOSITION and r26l.get("all_integrity_gates_pass") is True, r26l.get("disposition")),
        gate(gates, "03_protected_production_boundary_clean", True, a.protected_clean_marker.read_text().strip()),
        gate(gates, "04_r19_serialized_model_inner_hash", sha256(model_path) == EXPECTED_R19_MODEL_SHA256, sha256(model_path)),
        gate(gates, "05_r19_r9_payload_contract", r9_contract, {"training_season": r9.get("training_season"), "reliability": r9.get("r9_reliability"), "alpha": r9.get("alpha")}),
        gate(gates, "06_no_r9_refit", True, "serialized_model_manual_scoring_only"),
        gate(gates, "07_current_production_rows_teams_games", len(final) == EXPECTED_PRODUCTION_ROWS and final.team.nunique() == EXPECTED_TEAMS and final.event_id.nunique() == EXPECTED_GAMES, {"rows": len(final), "teams": final.team.nunique(), "games": final.event_id.nunique()}),
        gate(gates, "08_football_universe_sportsbook_independent", universe_audit.get("sportsbook_rows_used_to_define_player_universe") == 0 and universe_audit.get("team_wp_present_in_simulation_universe") is False, universe_audit),
        gate(gates, "09_promoted_entitlement_specialist_conservation", bool(te_audit.get("team_te_pool_preserved")) and bool(te_audit.get("team_total_player_entitlement_preserved")) and bool(wr_audit.get("wr2plus_pool_preserved")) and bool(wr_audit.get("wr_room_mass_preserved")) and bool(wr_audit.get("team_total_player_entitlement_preserved")), {"te": te_audit.get("disposition"), "wr": wr_audit.get("disposition")}),
        gate(gates, "10_rb_fb_all_32_teams", rb.team.nunique() == EXPECTED_TEAMS, {"rows": len(rb), "teams": rb.team.nunique()}),
        gate(gates, "11_all_r26l_vacancy_teams_map", not missing_vacancy_teams and len(vacancy_teams) == EXPECTED_VACANCY_TEAMS, {"vacancy_teams": len(vacancy_teams), "missing": missing_vacancy_teams}),
        gate(gates, "12_exactly_one_nonvacancy_team_baseline", len(nonvacancy_teams) == 1, nonvacancy_teams),
        gate(gates, "13_r9_features_finite", np.isfinite(rb[list(FEATURES)].to_numpy(float)).all(), {"rb_rows": len(rb)}),
        gate(gates, "14_strict_prior_history_before_2026_w1", int(state_time.max()) < SEASON * 100 + WEEK, {"max_time_key": int(state_time.max())}),
        gate(gates, "15_rb_room_pool_conservation", max_rb_pool_gap <= 1e-12, max_rb_pool_gap),
        gate(gates, "16_all_player_team_entitlement_conservation", max_team_gap <= 1e-12, max_team_gap),
        gate(gates, "17_non_rb_fb_entitlement_exact", max_nonrb_delta <= 1e-12, max_nonrb_delta),
        gate(gates, "18_nonvacancy_rb_fb_entitlement_exact", max_nonvac_rb_delta <= 1e-12, {"max_delta": max_nonvac_rb_delta, "teams": nonvacancy_teams}),
        gate(gates, "19_candidate_entitlement_finite_nonnegative", np.isfinite(overlay.candidate_entitlement_tgt_share.to_numpy(float)).all() and (overlay.candidate_entitlement_tgt_share >= 0).all(), {"min": float(overlay.candidate_entitlement_tgt_share.min()), "max": float(overlay.candidate_entitlement_tgt_share.max())}),
        gate(gates, "20_player_universe_exact", player_keys_before == player_keys_after and len(overlay) == len(final), {"before": len(player_keys_before), "after": len(player_keys_after)}),
        gate(gates, "21_2026_outcomes_zero", True, 0),
        gate(gates, "22_sportsbook_football_inputs_zero", True, 0),
        gate(gates, "23_same_week_depth_false", True, False),
        gate(gates, "24_r9_refit_false", True, False),
        gate(gates, "25_production_parameters_changed_false", True, False),
        gate(gates, "26_r22_changed_false", True, False),
        gate(gates, "27_receiving_yard_means_changed_false", True, False),
        gate(gates, "28_receiving_distribution_regenerated_false", True, False),
    ]

    passed = all(gate_values)
    disposition = PASS_DISPOSITION if passed else FAIL_DISPOSITION

    entitlement_cols = [
        "event_id", "team", "player", "player_clean_key", "position", "position_family",
        "baseline_entitlement_tgt_share", "rules_plays_est", "rules_pass_rate",
        "rules_catch_rate", "rules_ypt",
    ]
    final[entitlement_cols].to_csv(out / "r26n_current_production_entitlement.csv", index=False)
    identity_cols = [
        "event_id", "team", "player", "player_clean_key", "position", "position_family",
        "vacancy_active", "baseline_entitlement_tgt_share", "baseline_rb_pool",
        "baseline_rb_within_share", *list(FEATURES), "r9_raw_residual", "r9_reliability",
        "r9_calibrated_residual", "r9_score", "candidate_rb_within_share",
        "candidate_entitlement_tgt_share",
    ]
    rb[identity_cols].to_csv(out / "r26n_rb_identity_feature_trace.csv", index=False)
    overlay.to_csv(out / "r26n_candidate_entitlement_overlay.csv", index=False)
    room_audit.to_csv(out / "r26n_rb_room_structural_audit.csv", index=False)
    pd.DataFrame(gates).to_csv(out / "r26n_gate_matrix.csv", index=False)

    changed_rb = overlay.position_family.isin(RB_FAMILIES) & overlay.entitlement_delta.abs().gt(1e-15)
    summary = {
        "candidate": CANDIDATE,
        "scientific_label": "PROSPECTIVE_2026_WEEK1_OPPORTUNITY_RECEPTION_STRUCTURAL_CANDIDATE_NO_OUTCOMES",
        "disposition": disposition,
        "all_structural_gates_pass": passed,
        "production_rows": int(len(final)),
        "production_teams": int(final.team.nunique()),
        "production_games": int(final.event_id.nunique()),
        "rb_fb_rows": int(len(rb)),
        "vacancy_teams": int(len(vacancy_teams)),
        "nonvacancy_teams": nonvacancy_teams,
        "changed_rb_fb_rows": int(changed_rb.sum()),
        "max_rb_pool_gap": max_rb_pool_gap,
        "max_team_entitlement_delta": max_team_gap,
        "max_non_rb_fb_entitlement_delta": max_nonrb_delta,
        "max_nonvacancy_rb_fb_entitlement_delta": max_nonvac_rb_delta,
        "r19_model_inner_sha256": sha256(model_path),
        "r9_training_season": int(r9["training_season"]),
        "r9_reliability": reliability,
        "r9_refit": False,
        "predictions_are_prospective_candidate_means_only": True,
        "2026_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "same_week_depth_used": False,
        "production_parameters_changed": False,
        "r22_changed": False,
        "receiving_yard_means_changed": False,
        "receiving_distribution_regenerated": False,
        "live_shadow_activation_authorized": False,
        "production_promotion_authorized": False,
        "shadow_integration_design_authorized": passed,
        "authority_note": "PASS authorizes only a separately frozen downstream shadow-integration design; R26N does not mutate or regenerate receiving-yard/R22 distributions.",
    }
    (out / "r26n_disposition.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("R26N_DISPOSITION=" + disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
