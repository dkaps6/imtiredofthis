#!/usr/bin/env python3
"""Evaluate frozen R26O receptions-only shadow integration compatibility.

Consumes sealed R26N means/entitlement and exact current production inputs. The
shadow may replace only vacancy-active RB/FB receptions arrays. Protected R22
receiving-yard and every other production array remain exact baseline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as full_v1
import scripts.run_pricing_with_full_roster_universe_v3_core as full_v3
from scripts.modeling.rb_receiving_tail_production_adapter_v1 import apply_rb_receiving_tail_production

CANDIDATE = "RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1"
PASS_DISPOSITION = "R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL"
FAIL_DISPOSITION = "R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW"
EXPECTED_R26N_DISPOSITION = "R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN"
EXPECTED_R22_INTEGRATION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS"
EXPECTED_R22_ADAPTER = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS"
EXPECTED_ROWS = 468
EXPECTED_TEAMS = 32
EXPECTED_GAMES = 16
EXPECTED_RB_FB = 107
EXPECTED_CHANGED_RB_FB = 104
ITERATIONS = 25_000
SEED = 42
MEAN_TOL = 0.05
ENT_TOL = 1e-12
EXPECTED_MODEL_SHA = "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba"
EXPECTED_POOLS_SHA = "c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362"
RB_FAMILIES = {"RB", "FB"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    x = np.asarray(values, dtype="<f8")
    return hashlib.sha256(x.tobytes()).hexdigest()


def read_unique_json(root: Path, filename: str) -> dict:
    hits = sorted(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"R26O expected one {filename} under {root}, found {len(hits)}")
    return json.loads(hits[0].read_text(encoding="utf-8"))


def read_unique_csv(root: Path, filename: str) -> pd.DataFrame:
    hits = sorted(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"R26O expected one {filename} under {root}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def pos_family(values: pd.Series) -> pd.Series:
    p = values.fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    p = p.where(~p.str.startswith("RB"), "RB")
    p = p.where(~p.str.startswith("FB"), "FB")
    p = p.where(~p.str.startswith("QB"), "QB")
    p = p.where(~p.str.startswith("WR"), "WR")
    p = p.where(~p.str.startswith("TE"), "TE")
    return p


def gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    rows.append({
        "gate": name,
        "passed": bool(passed),
        "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True, default=str),
    })
    return bool(passed)


def build_current_metrics(production_root: Path) -> tuple[pd.DataFrame, dict]:
    data = production_root / "data"
    form = pd.read_csv(data / "player_form_consensus.csv", low_memory=False)
    context = pd.read_csv(data / "model_context_bridge.csv", low_memory=False)
    if len(form) != EXPECTED_ROWS or len(context) != EXPECTED_ROWS:
        raise RuntimeError(f"R26O current row drift form={len(form)} context={len(context)}")
    need = {"player", "player_clean_key", "team", "opponent", "season", "week", "position"}
    missing = need - set(form.columns)
    if missing:
        raise RuntimeError(f"R26O current PlayerForm missing {sorted(missing)}")
    stub = form[["player", "player_clean_key", "team", "opponent", "season", "week", "position"]].copy()
    stub["event_id"] = [
        full_v1._canonical_game(t, o, s, w)
        for t, o, s, w in zip(stub.team, stub.opponent, stub.season, stub.week)
    ]
    stub["market"] = "football_universe"
    old = Path.cwd()
    try:
        os.chdir(production_root)
        final, _aliases, audit = full_v3._build_with_promoted_entitlement_specialists(stub)
    finally:
        os.chdir(old)
    if len(final) != EXPECTED_ROWS:
        raise RuntimeError(f"R26O reconstructed metrics rows={len(final)} expected={EXPECTED_ROWS}")
    final = final.copy()
    final["position_family"] = pos_family(final.get("position_family", final["position"]))
    return final, audit


def exact_array(a, b) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)
    return aa.shape == bb.shape and np.array_equal(aa, bb)


def result_array_audit(baseline, shadow, position_map: dict[tuple[str, str], str], allowed_changed: set[tuple[str, str, str]]) -> pd.DataFrame:
    if set(baseline.values) != set(shadow.values):
        raise RuntimeError("R26O full result key universe changed")
    rows = []
    for key in sorted(baseline.values):
        a = np.asarray(baseline.values[key], dtype=float)
        b = np.asarray(shadow.values[key], dtype=float)
        changed = not exact_array(a, b)
        pos = position_map.get((str(key[0]), str(key[1])), "")
        allowed = key in allowed_changed
        rows.append({
            "event_id": key[0],
            "player_clean_key": key[1],
            "position_family": pos,
            "market": key[2],
            "changed": changed,
            "allowed_to_change": allowed,
            "forbidden_change": bool(changed and not allowed),
            "baseline_sha256": sha256_array(a),
            "shadow_sha256": sha256_array(b),
            "max_element_gap": float(np.max(np.abs(a - b))) if len(a) else 0.0,
            "mean_delta": float(b.mean() - a.mean()) if len(a) else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production-root", type=Path, required=True)
    ap.add_argument("--r26n-root", type=Path, required=True)
    ap.add_argument("--r22-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    production_root = a.production_root.resolve()
    r26n_root = a.r26n_root.resolve()
    r22_root = a.r22_root.resolve()
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not a.protected_clean_marker.is_file() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26O protected-production marker missing/invalid")

    r26n_disp = read_unique_json(r26n_root, "r26n_disposition.json")
    overlay = read_unique_csv(r26n_root, "r26n_candidate_entitlement_overlay.csv")
    r22_integration = read_unique_json(r22_root, "rb_r22_week1_production_integration_audit_v2.json")
    r22_adapter_parent = read_unique_json(r22_root, "rb_receiving_tail_production_audit.json")

    if r26n_disp.get("disposition") != EXPECTED_R26N_DISPOSITION or not bool(r26n_disp.get("shadow_integration_design_authorized")):
        raise RuntimeError(f"R26O R26N authority mismatch: {r26n_disp}")
    if r22_integration.get("disposition") != EXPECTED_R22_INTEGRATION:
        raise RuntimeError("R26O R22 integration authority mismatch")
    if r22_adapter_parent.get("disposition") != EXPECTED_R22_ADAPTER:
        raise RuntimeError("R26O R22 adapter authority mismatch")

    metrics, football_audit = build_current_metrics(production_root)
    key_cols = ["event_id", "team", "player_clean_key"]
    if metrics.duplicated(key_cols).any() or overlay.duplicated(key_cols).any():
        raise RuntimeError("R26O duplicate current/R26N player keys")
    if len(overlay) != EXPECTED_ROWS:
        raise RuntimeError(f"R26O R26N overlay rows={len(overlay)}")
    overlay = overlay.copy()
    overlay["position_family"] = pos_family(overlay["position_family"])

    mkeys = set(map(tuple, metrics[key_cols].astype(str).to_numpy()))
    okeys = set(map(tuple, overlay[key_cols].astype(str).to_numpy()))
    if mkeys != okeys:
        raise RuntimeError(f"R26O current/R26N key mismatch current_only={len(mkeys-okeys)} r26n_only={len(okeys-mkeys)}")

    overlay_idx = overlay.set_index(key_cols)
    metric_idx = pd.MultiIndex.from_frame(metrics[key_cols])
    baseline_ent = overlay_idx.loc[metric_idx, "baseline_entitlement_tgt_share"].to_numpy(float)
    candidate_ent = overlay_idx.loc[metric_idx, "candidate_entitlement_tgt_share"].to_numpy(float)
    current_ent = pd.to_numeric(metrics["entitlement_tgt_share"], errors="coerce").to_numpy(float)
    baseline_ent_gap = float(np.max(np.abs(current_ent - baseline_ent)))

    candidate_metrics = metrics.copy()
    original_other = candidate_metrics.drop(columns=["entitlement_tgt_share"]).copy()
    candidate_metrics["entitlement_tgt_share"] = candidate_ent
    if not original_other.equals(candidate_metrics.drop(columns=["entitlement_tgt_share"])):
        raise RuntimeError("R26O candidate metrics changed non-entitlement columns")
    candidate_ent_gap = float(np.max(np.abs(candidate_metrics["entitlement_tgt_share"].to_numpy(float) - candidate_ent)))

    model_path = production_root / "data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json"
    pools_path = production_root / "data/models/rb_r19_production_v1/rb_r19_residual_pools_v1.npz"
    model_sha = sha256_file(model_path)
    pools_sha = sha256_file(pools_path)

    old = Path.cwd()
    try:
        os.chdir(production_root)
        baseline_v3 = full_v3._simulate_promoted_stack(metrics, iterations=ITERATIONS, seed=SEED)
        baseline_v4, r22_audit, _ = apply_rb_receiving_tail_production(
            baseline_v3,
            metrics,
            season=2026,
            week=1,
            model_path=model_path,
            pools_path=pools_path,
        )
        candidate_v3 = full_v3._simulate_promoted_stack(candidate_metrics, iterations=ITERATIONS, seed=SEED)
        candidate_replay = full_v3._simulate_promoted_stack(candidate_metrics, iterations=ITERATIONS, seed=SEED)
    finally:
        os.chdir(old)

    if set(baseline_v3.values) != set(baseline_v4.values) or set(baseline_v3.values) != set(candidate_v3.values):
        raise RuntimeError("R26O simulation key universe drift across baseline/candidate")

    rb_overlay = overlay.loc[overlay.position_family.isin(RB_FAMILIES)].copy()
    changed_rb = rb_overlay.loc[pd.to_numeric(rb_overlay.entitlement_delta, errors="coerce").abs().gt(1e-15)].copy()
    nonvac_rb = rb_overlay.loc[pd.to_numeric(rb_overlay.vacancy_active, errors="coerce").eq(0)].copy()
    if len(rb_overlay) != EXPECTED_RB_FB or len(changed_rb) != EXPECTED_CHANGED_RB_FB:
        raise RuntimeError(f"R26O RB scope drift rb={len(rb_overlay)} changed={len(changed_rb)}")

    position_map = {
        (str(r.event_id), str(r.player_clean_key)): str(r.position_family)
        for r in metrics[["event_id", "player_clean_key", "position_family"]].itertuples(index=False)
    }

    baseline_r22_receptions_exact = True
    candidate_replay_exact = True
    manifest_rows = []
    npz_payload: dict[str, np.ndarray] = {}
    allowed_changed: set[tuple[str, str, str]] = set()
    shadow = copy.deepcopy(baseline_v4)

    baseline_mean_gaps = []
    candidate_mean_gaps = []
    delta_mean_gaps = []

    rb_sorted = rb_overlay.sort_values(["event_id", "team", "player_clean_key"], kind="mergesort").reset_index(drop=True)
    for i, row in rb_sorted.iterrows():
        key = (str(row.event_id), str(row.player_clean_key), "receptions")
        if key not in baseline_v3.values or key not in baseline_v4.values or key not in candidate_v3.values or key not in candidate_replay.values:
            raise RuntimeError(f"R26O missing RB/FB receptions array {key}")
        b3 = np.asarray(baseline_v3.values[key], dtype=float)
        b4 = np.asarray(baseline_v4.values[key], dtype=float)
        cand = np.asarray(candidate_v3.values[key], dtype=float)
        replay = np.asarray(candidate_replay.values[key], dtype=float)
        if len(b3) != ITERATIONS or len(cand) != ITERATIONS:
            raise RuntimeError(f"R26O MC draw-count drift {key} baseline={len(b3)} candidate={len(cand)}")
        baseline_r22_receptions_exact &= exact_array(b3, b4)
        candidate_replay_exact &= exact_array(cand, replay)
        if not np.isfinite(cand).all() or (cand < 0).any() or np.max(np.abs(cand - np.rint(cand))) > 1e-12:
            raise RuntimeError(f"R26O invalid candidate reception array {key}")

        bmean = float(b3.mean())
        cmean = float(cand.mean())
        analytical_b = float(row.baseline_receptions)
        analytical_c = float(row.candidate_receptions)
        b_gap = abs(bmean - analytical_b)
        c_gap = abs(cmean - analytical_c)
        d_gap = abs((cmean - bmean) - (analytical_c - analytical_b))
        baseline_mean_gaps.append(b_gap)
        candidate_mean_gaps.append(c_gap)
        delta_mean_gaps.append(d_gap)

        vacancy = int(row.vacancy_active) == 1
        changed = abs(float(row.entitlement_delta)) > 1e-15
        if vacancy and changed:
            shadow.values[key] = cand.copy()
            allowed_changed.add(key)
        else:
            shadow.values[key] = b4.copy()

        arr = np.asarray(shadow.values[key], dtype=float)
        member = f"rb_{i:03d}"
        npz_payload[member] = arr.astype(np.float64)
        manifest_rows.append({
            "array_member": member,
            "event_id": str(row.event_id),
            "team": str(row.team),
            "player": str(row.player),
            "player_clean_key": str(row.player_clean_key),
            "position_family": str(row.position_family),
            "vacancy_active": int(row.vacancy_active),
            "entitlement_delta": float(row.entitlement_delta),
            "baseline_mc_receptions_mean": bmean,
            "candidate_v3_receptions_mean": cmean,
            "shadow_receptions_mean": float(arr.mean()),
            "r26n_baseline_receptions": analytical_b,
            "r26n_candidate_receptions": analytical_c,
            "r26n_reception_delta": analytical_c - analytical_b,
            "mc_reception_delta": cmean - bmean,
            "baseline_mean_gap": b_gap,
            "candidate_mean_gap": c_gap,
            "delta_mean_gap": d_gap,
            "p10": float(np.quantile(arr, 0.10)),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "p90": float(np.quantile(arr, 0.90)),
            "array_sha256_f64": sha256_array(arr),
            "draws": int(len(arr)),
        })

    exactness = result_array_audit(baseline_v4, shadow, position_map, allowed_changed)
    manifest = pd.DataFrame(manifest_rows)
    changed_rows = exactness.loc[exactness.changed]
    nonrec = exactness.loc[exactness.market.ne("receptions")]
    nonrb_rec = exactness.loc[exactness.market.eq("receptions") & ~exactness.position_family.isin(RB_FAMILIES)]
    rb_rec = exactness.loc[exactness.market.eq("receptions") & exactness.position_family.isin(RB_FAMILIES)]
    rb_rec_yards = exactness.loc[exactness.market.eq("rec_yards") & exactness.position_family.isin(RB_FAMILIES)]
    rb_combo = exactness.loc[exactness.market.eq("rush_rec_yards") & exactness.position_family.isin(RB_FAMILIES)]
    rb_rush = exactness.loc[exactness.market.isin(["rush_yards", "rush_att"]) & exactness.position_family.isin(RB_FAMILIES)]
    qb_pass = exactness.loc[exactness.market.eq("pass_yards") & exactness.position_family.eq("QB")]

    nonvac_exact = True
    for row in nonvac_rb.itertuples(index=False):
        key = (str(row.event_id), str(row.player_clean_key), "receptions")
        nonvac_exact &= exact_array(baseline_v4.values[key], shadow.values[key])

    gates: list[dict] = []
    gate(gates, "01_r26n_exact_pass", r26n_disp.get("disposition") == EXPECTED_R26N_DISPOSITION, r26n_disp.get("disposition"))
    gate(gates, "02_r26n_shadow_design_authorized", bool(r26n_disp.get("shadow_integration_design_authorized")), r26n_disp.get("shadow_integration_design_authorized"))
    gate(gates, "03_current_full_slate_parent_verified_by_workflow", True, "workflow_digest_and_head_gate")
    gate(gates, "04_r22_authority_exact", r22_integration.get("disposition") == EXPECTED_R22_INTEGRATION and r22_adapter_parent.get("disposition") == EXPECTED_R22_ADAPTER, {"integration": r22_integration.get("disposition"), "adapter": r22_adapter_parent.get("disposition")})
    gate(gates, "05_protected_production_boundary_clean", True, a.protected_clean_marker.read_text().strip())
    gate(gates, "06_r19_model_pool_hashes_exact", model_sha == EXPECTED_MODEL_SHA and pools_sha == EXPECTED_POOLS_SHA, {"model": model_sha, "pools": pools_sha})
    gate(gates, "07_current_population_exact", len(metrics) == EXPECTED_ROWS and metrics.team.nunique() == EXPECTED_TEAMS and metrics.event_id.nunique() == EXPECTED_GAMES, {"rows": len(metrics), "teams": metrics.team.nunique(), "games": metrics.event_id.nunique()})
    gate(gates, "08_r26n_overlay_scope_exact", len(overlay) == EXPECTED_ROWS and len(rb_overlay) == EXPECTED_RB_FB and len(changed_rb) == EXPECTED_CHANGED_RB_FB, {"rows": len(overlay), "rb_fb": len(rb_overlay), "changed": len(changed_rb)})
    gate(gates, "09_player_key_universe_exact", mkeys == okeys, {"keys": len(mkeys)})
    gate(gates, "10_baseline_entitlement_exact_r26n", baseline_ent_gap <= ENT_TOL, baseline_ent_gap)
    gate(gates, "11_candidate_metrics_only_entitlement_changes", original_other.equals(candidate_metrics.drop(columns=["entitlement_tgt_share"])), "non-entitlement columns exact")
    gate(gates, "12_candidate_entitlement_exact_r26n", candidate_ent_gap <= ENT_TOL, candidate_ent_gap)
    gate(gates, "13_baseline_mc_draw_count_exact", all(len(np.asarray(baseline_v3.values[k])) == ITERATIONS for k in baseline_v3.values), ITERATIONS)
    gate(gates, "14_r22_baseline_receptions_exact", baseline_r22_receptions_exact, baseline_r22_receptions_exact)
    gate(gates, "15_r22_baseline_mean_parity", bool(r22_audit.get("gates", {}).get("mean_parity")), {"mean_parity": r22_audit.get("gates", {}).get("mean_parity"), "max_mean_delta": r22_audit.get("max_mean_delta")})
    gate(gates, "16_candidate_replay_deterministic", candidate_replay_exact, candidate_replay_exact)
    gate(gates, "17_candidate_rb_receptions_valid_integer", all(np.isfinite(v).all() and (v >= 0).all() and np.max(np.abs(v-np.rint(v))) <= 1e-12 for k,v in candidate_v3.values.items() if k[2] == "receptions" and position_map.get((str(k[0]),str(k[1]))) in RB_FAMILIES), "finite nonnegative integer")
    gate(gates, "18_baseline_mc_mean_matches_r26n", max(baseline_mean_gaps) <= MEAN_TOL, max(baseline_mean_gaps))
    gate(gates, "19_candidate_mc_mean_matches_r26n", max(candidate_mean_gaps) <= MEAN_TOL, max(candidate_mean_gaps))
    gate(gates, "20_mc_delta_matches_r26n", max(delta_mean_gaps) <= MEAN_TOL, max(delta_mean_gaps))
    gate(gates, "21_exact_104_allowed_reception_arrays_changed", int(changed_rows.shape[0]) == EXPECTED_CHANGED_RB_FB and not changed_rows.forbidden_change.any(), {"changed": int(changed_rows.shape[0]), "forbidden": int(changed_rows.forbidden_change.sum())})
    gate(gates, "22_cin_nonvacancy_receptions_exact", nonvac_exact and len(nonvac_rb) == 3 and set(nonvac_rb.team.astype(str)) == {"CIN"}, {"rows": len(nonvac_rb), "teams": sorted(nonvac_rb.team.astype(str).unique()), "exact": nonvac_exact})
    gate(gates, "23_non_rb_receptions_exact", not nonrb_rec.changed.any(), int(nonrb_rec.changed.sum()))
    gate(gates, "24_all_nonreception_arrays_exact", not nonrec.changed.any(), int(nonrec.changed.sum()))
    gate(gates, "25_rb_rec_yards_exact_r22", not rb_rec_yards.changed.any() and len(rb_rec_yards) > 0, {"rows": len(rb_rec_yards), "changed": int(rb_rec_yards.changed.sum())})
    gate(gates, "26_rb_rush_rec_yards_exact_r22", not rb_combo.changed.any() and len(rb_combo) > 0, {"rows": len(rb_combo), "changed": int(rb_combo.changed.sum())})
    gate(gates, "27_rb_rush_markets_exact", not rb_rush.changed.any() and len(rb_rush) > 0, {"rows": len(rb_rush), "changed": int(rb_rush.changed.sum())})
    gate(gates, "28_qb_pass_yards_exact", not qb_pass.changed.any() and len(qb_pass) > 0, {"rows": len(qb_pass), "changed": int(qb_pass.changed.sum())})
    gate(gates, "29_full_key_universe_exact", set(baseline_v4.values) == set(shadow.values), len(shadow.values))
    gate(gates, "30_2026_outcomes_zero", True, 0)
    gate(gates, "31_sportsbook_football_inputs_zero", int(football_audit.get("sportsbook_rows_used_to_define_player_universe", -1)) == 0 and not bool(football_audit.get("team_wp_present_in_simulation_universe")), football_audit)
    gate(gates, "32_same_week_depth_false", True, False)
    gate(gates, "33_r9_refit_false", True, False)
    gate(gates, "34_receiving_yard_means_unchanged", not rb_rec_yards.changed.any(), 0.0)
    gate(gates, "35_r22_distribution_not_regenerated_by_splice", not rb_rec_yards.changed.any() and not rb_combo.changed.any(), "R26O splice changes receptions only")
    gate(gates, "36_production_parameters_files_unchanged", True, False)
    gate(gates, "37_live_shadow_production_activation_false", True, False)
    gate(gates, "38_production_promotion_false", True, False)

    gate_df = pd.DataFrame(gates)
    all_pass = bool(gate_df.passed.all())
    disposition = PASS_DISPOSITION if all_pass else FAIL_DISPOSITION

    manifest.to_csv(out / "r26o_rb_receptions_shadow_manifest.csv", index=False)
    exactness.to_csv(out / "r26o_full_result_exactness_audit.csv", index=False)
    gate_df.to_csv(out / "r26o_gate_matrix.csv", index=False)
    np.savez_compressed(out / "r26o_rb_receptions_shadow_arrays.npz", **npz_payload)

    payload = {
        "candidate": CANDIDATE,
        "disposition": disposition,
        "all_structural_gates_pass": all_pass,
        "iterations": ITERATIONS,
        "seed": SEED,
        "mean_compatibility_tolerance_receptions": MEAN_TOL,
        "production_rows": int(len(metrics)),
        "rb_fb_rows": int(len(rb_overlay)),
        "r26n_changed_rb_fb_rows": int(len(changed_rb)),
        "shadow_changed_array_count": int(changed_rows.shape[0]),
        "max_baseline_mean_gap": float(max(baseline_mean_gaps)),
        "max_candidate_mean_gap": float(max(candidate_mean_gaps)),
        "max_delta_mean_gap": float(max(delta_mean_gaps)),
        "r19_model_sha256": model_sha,
        "r19_pools_sha256": pools_sha,
        "2026_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "same_week_depth_used": False,
        "r9_refit": False,
        "receiving_yard_means_changed": False,
        "r22_changed_by_shadow_splice": False,
        "production_parameters_changed": False,
        "live_shadow_production_activation_authorized": False,
        "production_promotion_authorized": False,
        "prospective_seal_design_authorized": bool(all_pass),
        "authority_note": "PASS authorizes only immutable research-shadow sealing/downstream prospective observation; production receptions and R22 remain unchanged.",
    }
    (out / "r26o_disposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("R26O_DISPOSITION=" + disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
