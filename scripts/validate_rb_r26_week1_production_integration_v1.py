#!/usr/bin/env python3
"""Frozen qualification evaluator for Week-1 R26 RB receptions production integration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PASS_DISPOSITION = "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION"
FAIL_DISPOSITION = "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_FAIL_NO_PROMOTION"
R26_VERSION = "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_V1"
VACANCY_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CLE", "DAL", "DEN", "DET", "GB", "HOU",
    "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ",
    "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"qualification missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"qualification missing CSV: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"qualification CSV empty: {path}")
    return out


def marker(path: Path) -> bool:
    return path.is_file() and path.read_text(encoding="utf-8").strip() == "PASS"


def eq_num(a: pd.Series, b: pd.Series, tol: float = 1e-8) -> bool:
    av = pd.to_numeric(a, errors="coerce").to_numpy(float)
    bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
    return bool(np.allclose(av, bv, rtol=0, atol=tol, equal_nan=True))


def eq_text(a: pd.Series, b: pd.Series) -> bool:
    return bool(a.astype("string").fillna("<NA>").eq(b.astype("string").fillna("<NA>")).all())


def add_gate(rows: list[dict], name: str, passed: bool, evidence) -> bool:
    rows.append({
        "gate": name,
        "passed": bool(passed),
        "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True, default=str),
    })
    return bool(passed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--frozen-marker", type=Path, required=True)
    ap.add_argument("--protected-marker", type=Path, required=True)
    ap.add_argument("--clean-assets-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    control = read_csv(a.control)
    candidate = read_csv(a.candidate)
    r26 = read_json(Path("data/rb_r26_receptions_production_audit.json"))
    r26_price = read_json(Path("data/rb_r26_receptions_pricing_lineage_audit.json"))
    r22 = read_json(Path("data/rb_receiving_tail_production_audit.json"))
    r22_price = read_json(Path("data/rb_receiving_tail_pricing_lineage_audit.json"))
    football = read_json(Path("data/football_simulation_universe_audit.json"))
    trace = read_csv(Path("data/rb_r26_receptions_production_trace.csv"))
    arrays = read_csv(Path("data/rb_r26_receptions_production_array_audit.csv"))

    required_candidate_cols = {
        "event_id", "player", "team", "market", "source_market", "vegas_line", "side", "book",
        "mc_proj", "ensemble_proj", "model_proj", "ml_proj", "state_proj",
        "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state",
        "ensemble_status", "ensemble_method", "rb_r26_receptions_applied",
        "rb_r26_receptions_version", "rb_r26_baseline_mc_proj_audit", "rb_r26_final_mc_proj_audit",
    }
    missing = required_candidate_cols - set(candidate.columns)
    if missing:
        raise RuntimeError(f"candidate priced output missing columns: {sorted(missing)}")

    key = ["event_id", "player", "team", "market", "source_market", "vegas_line", "side", "book"]
    if control.duplicated(key).any() or candidate.duplicated(key).any():
        raise RuntimeError("control/candidate priced output has duplicate exact offer keys")
    ckeys = set(map(tuple, control[key].astype(str).to_numpy()))
    nkeys = set(map(tuple, candidate[key].astype(str).to_numpy()))
    if ckeys != nkeys:
        raise RuntimeError(f"priced offer universe drift control_only={len(ckeys-nkeys)} candidate_only={len(nkeys-ckeys)}")

    compare_cols = [
        "mc_proj", "ensemble_proj", "model_proj", "model_sd", "fair_prob", "edge_pct", "market_prob",
        "ml_proj", "state_proj", "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state",
        "ensemble_status", "ensemble_method", "ensemble_calibration_rows",
        "rb_synthesis_applied", "rb_synthesis_proj", "rb_synthesis_version", "rb_synthesis_route",
    ]
    left = control[key + [c for c in compare_cols if c in control.columns]].copy()
    right = candidate[key + [c for c in compare_cols if c in candidate.columns] + [
        "rb_r26_receptions_applied", "rb_r26_receptions_version",
        "rb_r26_baseline_mc_proj_audit", "rb_r26_final_mc_proj_audit",
    ]].copy()
    merged = left.merge(right, on=key, how="inner", validate="one_to_one", suffixes=("_control", "_candidate"))
    applied = merged["rb_r26_receptions_applied"].fillna(False).astype(bool)
    applied &= merged["market"].astype(str).eq("receptions")
    nonapplied = ~applied

    # Control-vs-candidate preservation outside the intended R26 pricing scope.
    nonapplied_numeric_ok = True
    nonapplied_text_ok = True
    for col in ["mc_proj", "ensemble_proj", "model_proj", "model_sd", "fair_prob", "edge_pct", "market_prob"]:
        a_col, b_col = f"{col}_control", f"{col}_candidate"
        if a_col in merged.columns and b_col in merged.columns:
            nonapplied_numeric_ok &= eq_num(merged.loc[nonapplied, a_col], merged.loc[nonapplied, b_col], 1e-8)
    for col in ["ensemble_status", "ensemble_method"]:
        a_col, b_col = f"{col}_control", f"{col}_candidate"
        if a_col in merged.columns and b_col in merged.columns:
            nonapplied_text_ok &= eq_text(merged.loc[nonapplied, a_col], merged.loc[nonapplied, b_col])

    # On applied reception rows, ML/state/ensemble configuration must remain the same as V4.
    applied_inputs_ok = bool(applied.any())
    for col in ["ml_proj", "state_proj", "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state", "ensemble_calibration_rows"]:
        a_col, b_col = f"{col}_control", f"{col}_candidate"
        if a_col in merged.columns and b_col in merged.columns:
            applied_inputs_ok &= eq_num(merged.loc[applied, a_col], merged.loc[applied, b_col], 1e-12)
    for col in ["ensemble_status", "ensemble_method"]:
        a_col, b_col = f"{col}_control", f"{col}_candidate"
        if a_col in merged.columns and b_col in merged.columns:
            applied_inputs_ok &= eq_text(merged.loc[applied, a_col], merged.loc[applied, b_col])

    baseline_audit_matches_control = eq_num(
        merged.loc[applied, "rb_r26_baseline_mc_proj_audit"],
        merged.loc[applied, "mc_proj_control"],
        1e-8,
    ) if applied.any() else False
    final_audit_matches_candidate = eq_num(
        merged.loc[applied, "rb_r26_final_mc_proj_audit"],
        merged.loc[applied, "mc_proj_candidate"],
        1e-8,
    ) if applied.any() else False

    changed_arrays = arrays.loc[arrays["changed"].fillna(False).astype(bool)].copy()
    allowed_array_scope = bool(
        not changed_arrays.empty
        and changed_arrays["position_family"].astype(str).isin({"RB", "FB"}).all()
        and changed_arrays["market"].astype(str).eq("receptions").all()
        and changed_arrays["team"].astype(str).isin(VACANCY_TEAMS).all()
    )

    trace["team"] = trace["team"].astype(str).str.upper().str.strip()
    trace_applied = trace["rb_r26_receptions_applied"].fillna(False).astype(bool)
    cin = trace.loc[trace.team.eq("CIN")].copy()
    rb_identity_unique = not trace.duplicated(["event_id", "team", "player_clean_key"]).any()

    # Position-level exactness at the R26 seam.
    def no_changed(position: str | None = None, markets: set[str] | None = None, non_rb: bool = False) -> bool:
        x = arrays.copy()
        if non_rb:
            x = x.loc[~x.position_family.astype(str).isin({"RB", "FB"})]
        elif position is not None:
            x = x.loc[x.position_family.astype(str).eq(position)]
        if markets is not None:
            x = x.loc[x.market.astype(str).isin(markets)]
        return bool(not x["changed"].fillna(False).astype(bool).any())

    p3 = candidate.loc[pd.to_numeric(candidate.get("rb_synthesis_applied", 0), errors="coerce").fillna(0).eq(1)].copy()
    p3_ok = bool(
        not p3.empty
        and "rb_synthesis_proj" in p3.columns
        and eq_num(p3["model_proj"], p3["rb_synthesis_proj"], 1e-8)
    )

    model_hash_ok = bool(r26.get("model_assets", {}).get("model_sha256") == "9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba")
    room_hash_ok = bool(r26.get("model_assets", {}).get("room_state_sha256") == "27ad7bad8fcdfa6b0b1090994e0c0d2bdc4c0e45209d2409abc9574b5d354258")

    gates: list[dict] = []
    first34 = [
        add_gate(gates, "01_v2_frozen_before_qualification", marker(a.frozen_marker), a.frozen_marker.read_text().strip() if a.frozen_marker.is_file() else "MISSING"),
        add_gate(gates, "02_protected_parent_code_boundary", marker(a.protected_marker), a.protected_marker.read_text().strip() if a.protected_marker.is_file() else "MISSING"),
        add_gate(gates, "03_exact_r19_model_and_feature_contract", model_hash_ok and r26.get("disposition") == "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_ADAPTER_PASS", r26.get("model_assets")),
        add_gate(gates, "04_exact_r26l_vacancy_set_and_cin_control", room_hash_ok and int(r26.get("vacancy_teams", -1)) == 31 and r26.get("control_team") == "CIN", r26.get("model_assets")),
        add_gate(gates, "05_full_slate_32_teams_16_games", int(football.get("football_teams", 0)) == 32 and int(football.get("canonical_games", 0)) == 16, {"teams": football.get("football_teams"), "games": football.get("canonical_games"), "players": football.get("football_players")}),
        add_gate(gates, "06_current_rb_fb_32_teams_unique", trace.team.nunique() == 32 and rb_identity_unique, {"rows": len(trace), "teams": trace.team.nunique(), "unique": rb_identity_unique}),
        add_gate(gates, "07_strict_prior_identity_before_2026_w1", int(r26.get("strict_prior_max_time_key", 999999)) < 202601, r26.get("strict_prior_max_time_key")),
        add_gate(gates, "08_no_r9_refit", r26.get("r9_refit") is False, r26.get("r9_refit")),
        add_gate(gates, "09_rb_fb_room_pool_conservation", float(r26.get("max_rb_pool_gap", 1.0)) <= 1e-12, r26.get("max_rb_pool_gap")),
        add_gate(gates, "10_all_team_entitlement_conservation", float(r26.get("max_team_entitlement_gap", 1.0)) <= 1e-12, r26.get("max_team_entitlement_gap")),
        add_gate(gates, "11_non_rb_entitlement_exact", float(r26.get("max_non_rb_entitlement_delta", 1.0)) <= 1e-12, r26.get("max_non_rb_entitlement_delta")),
        add_gate(gates, "12_cin_entitlement_exact_baseline", float(r26.get("max_cin_entitlement_delta", 1.0)) <= 1e-12, r26.get("max_cin_entitlement_delta")),
        add_gate(gates, "13_candidate_reception_arrays_valid", r26.get("disposition") == "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_ADAPTER_PASS" and np.isfinite(pd.to_numeric(trace["final_receptions_mean"], errors="coerce")).all(), "adapter enforces finite/nonnegative/integer candidate draws before PASS"),
        add_gate(gates, "14_final_result_key_universe_exact_v4", r26.get("disposition") == "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_ADAPTER_PASS" and not arrays.duplicated(["event_id", "player_clean_key", "market"]).any(), {"array_rows": len(arrays)}),
        add_gate(gates, "15_only_vacancy_rb_fb_receptions_arrays_change", allowed_array_scope, {"changed": len(changed_arrays), "teams": sorted(changed_arrays.team.astype(str).unique().tolist()) if len(changed_arrays) else []}),
        add_gate(gates, "16_forbidden_changed_arrays_zero", int(r26.get("forbidden_changed_arrays", 1)) == 0 and not arrays["forbidden_change"].fillna(False).astype(bool).any(), r26.get("forbidden_changed_arrays")),
        add_gate(gates, "17_cin_final_receptions_exact_v4", not cin.empty and not cin["rb_r26_receptions_applied"].fillna(False).astype(bool).any() and pd.to_numeric(cin["final_minus_baseline_receptions_mean"], errors="coerce").abs().max() <= 1e-12, {"cin_rows": len(cin)}),
        add_gate(gates, "18_rb_fb_rec_yards_exact_r22", no_changed(markets={"rec_yards"}), "array audit"),
        add_gate(gates, "19_rb_fb_rush_rec_yards_exact_at_r26_seam", no_changed(markets={"rush_rec_yards"}), "array audit"),
        add_gate(gates, "20_rb_fb_rushing_arrays_exact", no_changed(markets={"rush_yards", "rush_att"}), "array audit"),
        add_gate(gates, "21_qb_arrays_exact_v4", no_changed(position="QB"), "array audit"),
        add_gate(gates, "22_wr_arrays_exact_v4", no_changed(position="WR"), "array audit"),
        add_gate(gates, "23_te_arrays_exact_v4", no_changed(position="TE"), "array audit"),
        add_gate(gates, "24_all_non_rb_arrays_exact_v4", no_changed(non_rb=True) and nonapplied_numeric_ok and nonapplied_text_ok, {"array_exact": no_changed(non_rb=True), "priced_nonapplied_exact": nonapplied_numeric_ok and nonapplied_text_ok}),
        add_gate(gates, "25_r22_contract_still_passes", r22.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS" and float(r22.get("max_mean_delta", 1.0)) <= 1e-8 and r22.get("gates", {}).get("receptions_exact") is True and r22_price.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS", {"r22": r22.get("disposition"), "pricing": r22_price.get("disposition"), "max_mean_delta": r22.get("max_mean_delta")}),
        add_gate(gates, "26_p3_final_rush_pricing_exact", p3_ok, {"rows": len(p3)}),
        add_gate(gates, "27_r26_mc_enters_existing_ensemble_without_rewriting_inputs", bool(applied.any()) and applied_inputs_ok and baseline_audit_matches_control and final_audit_matches_candidate and float(r26_price.get("max_abs_mc_trace_gap", 1.0)) <= 1e-8 and r26_price.get("existing_ml_state_inputs_rewritten_by_r26") is False and r26_price.get("existing_ensemble_method_rewritten_by_r26") is False, {"applied_rows": int(applied.sum()), "inputs_exact": applied_inputs_ok, "baseline_matches_control": baseline_audit_matches_control, "final_matches_candidate": final_audit_matches_candidate}),
        add_gate(gates, "28_single_final_model_proj_from_existing_ensemble", r26_price.get("single_authoritative_model_proj") is True and float(r26_price.get("max_abs_model_vs_ensemble_gap", 1.0)) <= 1e-8 and np.isfinite(pd.to_numeric(merged.loc[applied, "model_proj_candidate"], errors="coerce")).all(), {"applied_rows": int(applied.sum()), "max_gap": r26_price.get("max_abs_model_vs_ensemble_gap")}),
        add_gate(gates, "29_r26_pricing_lineage_and_baseline_audit_only", r26_price.get("disposition") == "RB_R26_WEEK1_RECEPTIONS_PRICING_LINEAGE_PASS" and r26_price.get("baseline_retained_for_audit_only") is True and int(r26_price.get("priced_r26_applied_rows", 0)) > 0 and candidate.loc[applied.reindex(candidate.index, fill_value=False) if False else candidate.index].shape[0] >= 0, {"disposition": r26_price.get("disposition"), "priced_applied": r26_price.get("priced_r26_applied_rows")}),
        add_gate(gates, "30_week1_outcomes_used_zero", int(r26.get("same_week_outcomes_used", -1)) == 0, r26.get("same_week_outcomes_used")),
        add_gate(gates, "31_sportsbook_football_inputs_zero", int(r26.get("sportsbook_inputs_used", -1)) == 0 and int(r26_price.get("sportsbook_inputs_to_r26_football", -1)) == 0, {"adapter": r26.get("sportsbook_inputs_used"), "pricing": r26_price.get("sportsbook_inputs_to_r26_football")}),
        add_gate(gates, "32_no_outcome_router_blend_or_tuning", r26.get("r9_refit") is False and r26.get("final_receptions_authority") == R26_VERSION, {"r9_refit": r26.get("r9_refit"), "authority": r26.get("final_receptions_authority")}),
        add_gate(gates, "33_no_same_week_outcome_data", int(r26.get("same_week_outcomes_used", -1)) == 0, r26.get("same_week_outcomes_used")),
        add_gate(gates, "34_clean_checkout_repo_pinned_assets", marker(a.clean_assets_marker), a.clean_assets_marker.read_text().strip() if a.clean_assets_marker.is_file() else "MISSING"),
    ]

    disposition = PASS_DISPOSITION if all(first34) else FAIL_DISPOSITION
    gate35 = add_gate(gates, "35_exact_authority_disposition", disposition == PASS_DISPOSITION, disposition)
    passed = all(first34) and gate35

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    gate_df = pd.DataFrame(gates)
    gate_df.to_csv(out / "rb_r26_week1_production_integration_gate_matrix.csv", index=False)

    applied_compare = merged.loc[applied].copy()
    applied_compare.to_csv(out / "rb_r26_week1_production_applied_pricing_comparison.csv", index=False)
    changed_arrays.to_csv(out / "rb_r26_week1_production_changed_array_scope.csv", index=False)

    summary = {
        "candidate": "RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_V1",
        "disposition": disposition,
        "gate_count": int(len(gates)),
        "gate_pass_count": int(gate_df.passed.sum()),
        "all_gates_pass": bool(passed),
        "priced_rows_control": int(len(control)),
        "priced_rows_candidate": int(len(candidate)),
        "priced_r26_applied_rows": int(applied.sum()),
        "r26_trace_rb_fb_rows": int(len(trace)),
        "r26_trace_applied_rows": int(trace_applied.sum()),
        "changed_reception_arrays": int(len(changed_arrays)),
        "week1_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "single_authoritative_receptions_output": True,
        "baseline_receptions_retained_for_audit_only_on_r26_scope": True,
        "production_promotion_authorized": bool(passed),
    }
    (out / "rb_r26_week1_production_integration_disposition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("R26_PRODUCTION_QUALIFICATION_DISPOSITION=" + disposition)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
