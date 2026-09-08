#!/usr/bin/env python3
"""Certified Full Slate stack with TE-R5P + WR-R15 + QB C2.

This is the production-candidate successor to full-roster V2. It preserves all
previously certified contracts and adds the scientifically authorized WR-R15
specialist at the explicit entitlement seam:

  legacy raw evidence -> M38 -> explicit finite team target entitlement
      -> TE-R5P inside conserved TE room
      -> WR-R15 inside conserved WR2+ room with M38 WR1 frozen
      -> canonical joint MC
      -> mean-neutral QB C2 distribution selector
      -> M89/M90 QB mean and RB P3 pricing authorities downstream

Sportsbook offers are never used to define the football universe, entitlement,
participation features, starter selection, C2 selection, or football simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
from scripts.modeling.qb_c2_production_adapter_v1 import (
    AUDIT_JSON as QB_C2_AUDIT_JSON,
    apply_qb_c2_selector,
)
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.simulation_c2_qb_candidate import simulate_with_states
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import simulate as legacy_simulate

DATA = Path("data")
ENTITLEMENT_TRACE = DATA / "target_entitlement_v1_trace.csv"
ENTITLEMENT_AUDIT = DATA / "target_entitlement_v1_audit.json"
ENTITLEMENT_INVARIANCE = DATA / "target_entitlement_v1_projection_invariance.csv"
TE_TRACE = DATA / "te_r5p_full_slate_entitlement_trace.csv"
TE_AUDIT = DATA / "te_r5p_full_slate_entitlement_audit.json"
TE_SIM_DELTA = DATA / "te_r5p_full_slate_simulation_delta.csv"
WR_TRACE = DATA / "wr_r15_full_slate_entitlement_trace.csv"
WR_AUDIT = DATA / "wr_r15_full_slate_entitlement_audit.json"
WR_SIM_DELTA = DATA / "wr_r15_full_slate_simulation_delta.csv"
QB_C2_STATE_PARITY = DATA / "qb_c2_state_capture_parity.csv"
VERSION = "TEAM_TARGET_ENTITLEMENT_V1_PLUS_TE_R5P_PLUS_WR_R15_V1"


def _write_delta(left, right, metrics: pd.DataFrame, path: Path, left_label: str, right_label: str) -> pd.DataFrame:
    if set(left.values) != set(right.values):
        raise RuntimeError(f"{right_label} changed simulation key universe")
    pos_map = {
        (str(r.event_id), str(r.player_clean_key)): str(r.position)
        for r in metrics[["event_id", "player_clean_key", "position"]].drop_duplicates().itertuples(index=False)
    }
    rows = []
    for key in sorted(left.values):
        a = np.asarray(left.values[key], dtype=float)
        b = np.asarray(right.values[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"{right_label} produced invalid simulation arrays key={key}")
        rows.append({
            "event_id": key[0],
            "player_clean_key": key[1],
            "position": pos_map.get((str(key[0]), str(key[1])), ""),
            "market": key[2],
            f"{left_label}_mean": float(a.mean()) if len(a) else np.nan,
            f"{right_label}_mean": float(b.mean()) if len(b) else np.nan,
            "mean_delta": float(b.mean() - a.mean()) if len(a) else np.nan,
            "abs_mean_delta": abs(float(b.mean() - a.mean())) if len(a) else np.nan,
            "max_element_gap": float(np.max(np.abs(a - b))) if len(a) else 0.0,
        })
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _build_with_promoted_entitlement_specialists(pricing_metrics: pd.DataFrame):
    universe, aliases, audit = v2._ORIGINAL_BUILD(pricing_metrics)
    baseline, baseline_trace = materialize_target_entitlement(universe)

    te_only, te_trace, te_audit = apply_te_r5p_entitlement(baseline)
    te_only["te_only_entitlement_tgt_share"] = pd.to_numeric(
        te_only["entitlement_tgt_share"], errors="raise"
    ).astype(float)

    final, wr_trace, wr_audit = apply_wr_r15_entitlement(te_only)

    ENTITLEMENT_TRACE.parent.mkdir(parents=True, exist_ok=True)
    te_trace.to_csv(TE_TRACE, index=False)
    wr_trace.to_csv(WR_TRACE, index=False)
    TE_AUDIT.write_text(json.dumps(te_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WR_AUDIT.write_text(json.dumps(wr_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_by_key = final.set_index(["event_id", "team", "player_clean_key"])
    trace = baseline_trace.copy()
    keys = pd.MultiIndex.from_frame(trace[["event_id", "team", "player_clean_key"]])
    trace["m38_explicit_entitlement_tgt_share"] = trace["entitlement_tgt_share"].astype(float)
    trace["entitlement_tgt_share"] = final_by_key.loc[keys, "entitlement_tgt_share"].to_numpy(float)
    trace["te_r5p_applied"] = final_by_key.loc[keys, "te_r5p_applied"].to_numpy(bool)
    trace["te_r5p_model_version"] = final_by_key.loc[keys, "te_r5p_model_version"].astype(str).to_numpy()
    trace["wr_r15_applied"] = final_by_key.loc[keys, "wr_r15_applied"].to_numpy(bool)
    trace["wr_r15_anchor"] = final_by_key.loc[keys, "wr_r15_anchor"].to_numpy(bool)
    trace["wr_r15_model_version"] = final_by_key.loc[keys, "wr_r15_model_version"].astype(str).to_numpy()
    trace["wr_r15_route"] = final_by_key.loc[keys, "wr_r15_route"].astype(str).to_numpy()
    trace["entitlement_version"] = VERSION
    trace.to_csv(ENTITLEMENT_TRACE, index=False)

    team = trace.drop_duplicates(["event_id", "team"])
    payload = {
        "disposition": "EXPLICIT_TARGET_ENTITLEMENT_MATERIALIZED",
        "version": VERSION,
        "baseline_refactor_version": "TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL",
        "football_players": int(len(final)),
        "teams": int(team["team"].nunique()),
        "games": int(team["event_id"].nunique()),
        "raw_team_sum_min": float(team["raw_team_sum"].min()),
        "raw_team_sum_median": float(team["raw_team_sum"].median()),
        "raw_team_sum_max": float(team["raw_team_sum"].max()),
        "explicit_modeled_sum_min": float(team["modeled_player_sum"].min()),
        "explicit_modeled_sum_max": float(team["modeled_player_sum"].max()),
        "residual_min": float(team["residual_share"].min()),
        "residual_max": float(team["residual_share"].max()),
        "m38_applied_before_entitlement": True,
        "production_specialists": ["TE_R5P_PRODUCTION_MODEL_V1", "WR_R15_PRODUCTION_MODEL_V1"],
        "te_r5p_entitlement_audit": str(TE_AUDIT),
        "te_r5p_entitlement_trace": str(TE_TRACE),
        "te_r5p_team_pool_preserved": bool(te_audit["team_te_pool_preserved"]),
        "te_r5p_non_te_entitlement_preserved": bool(te_audit["non_te_entitlement_preserved"]),
        "wr_r15_entitlement_audit": str(WR_AUDIT),
        "wr_r15_entitlement_trace": str(WR_TRACE),
        "wr_r15_m38_wr1_anchor_preserved": bool(wr_audit["m38_wr1_anchor_preserved"]),
        "wr_r15_wr2plus_pool_preserved": bool(wr_audit["wr2plus_pool_preserved"]),
        "wr_r15_wr_room_mass_preserved": bool(wr_audit["wr_room_mass_preserved"]),
        "wr_r15_non_wr_entitlement_preserved": bool(wr_audit["non_wr_entitlement_preserved"]),
        "sportsbook_inputs_used": False,
        "new_scientific_parameters_introduced": True,
        "trace": str(ENTITLEMENT_TRACE),
    }
    ENTITLEMENT_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.update({
        "explicit_target_entitlement_version": VERSION,
        "explicit_target_entitlement_materialized": True,
        "te_r5p_full_slate_consumed": True,
        "te_r5p_model_version": te_audit["model_version"],
        "wr_r15_full_slate_consumed": True,
        "wr_r15_model_version": wr_audit["model_version"],
        "wr_r15_m38_wr1_anchor_preserved": True,
    })
    return final, aliases, audit


def _simulate_promoted_stack(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    required = {
        "baseline_entitlement_tgt_share",
        "te_only_entitlement_tgt_share",
        "wr_r15_baseline_entitlement_tgt_share",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"promoted entitlement simulation missing columns: {sorted(missing)}")

    legacy_input = metrics.drop(
        columns=[
            c for c in metrics.columns
            if c.startswith("entitlement_")
            or c.startswith("baseline_entitlement_")
            or c.startswith("te_r5p_")
            or c.startswith("te_only_")
            or c.startswith("wr_r15_")
        ],
        errors="ignore",
    )
    baseline_metrics = metrics.copy()
    baseline_metrics["entitlement_tgt_share"] = pd.to_numeric(
        baseline_metrics["baseline_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    legacy = legacy_simulate(legacy_input, iterations=iterations, seed=seed)
    baseline_explicit = explicit_simulate(baseline_metrics, iterations=iterations, seed=seed)
    neutral = v2._compare_results_exact(legacy, baseline_explicit, label="explicit entitlement baseline")
    neutral_rows = []
    for key in sorted(legacy.values):
        a = np.asarray(legacy.values[key], dtype=float)
        b = np.asarray(baseline_explicit.values[key], dtype=float)
        neutral_rows.append({
            "event_id": key[0], "player_clean_key": key[1], "market": key[2],
            "legacy_mean": float(a.mean()) if len(a) else np.nan,
            "baseline_explicit_mean": float(b.mean()) if len(b) else np.nan,
            "mean_gap": abs(float(a.mean()) - float(b.mean())),
            "max_element_gap": float(np.max(np.abs(a - b))) if len(a) else 0.0,
        })
    pd.DataFrame(neutral_rows).to_csv(ENTITLEMENT_INVARIANCE, index=False)
    if neutral["changed_arrays"] != 0 or neutral["max_mean_gap"] > 1e-12 or neutral["max_element_gap"] > 1e-12:
        raise RuntimeError(f"explicit target entitlement baseline is not exactly projection-neutral: {neutral}")

    te_metrics = metrics.copy()
    te_metrics["entitlement_tgt_share"] = pd.to_numeric(
        te_metrics["te_only_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    te_result = explicit_simulate(te_metrics, iterations=iterations, seed=seed)
    final = explicit_simulate(metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace)
    te_delta = _write_delta(baseline_explicit, te_result, metrics, TE_SIM_DELTA, "m38_baseline", "te_r5p")
    wr_delta = _write_delta(te_result, final, metrics, WR_SIM_DELTA, "te_r5p", "wr_r15")

    entitlement = json.loads(ENTITLEMENT_AUDIT.read_text(encoding="utf-8"))
    wr_audit = json.loads(WR_AUDIT.read_text(encoding="utf-8"))
    te_audit = json.loads(TE_AUDIT.read_text(encoding="utf-8"))
    entitlement.update({
        "projection_invariance_scope": "legacy_vs_explicit_m38_baseline_before_position_specialists",
        "projection_invariance_keys": neutral["keys"],
        "projection_invariance_changed_arrays": neutral["changed_arrays"],
        "projection_invariance_max_mean_gap": neutral["max_mean_gap"],
        "projection_invariance_max_element_gap": neutral["max_element_gap"],
        "projection_neutral_gate": "PASS",
        "invariance_audit": str(ENTITLEMENT_INVARIANCE),
        "te_r5p_simulation_delta_audit": str(TE_SIM_DELTA),
        "te_r5p_changed_distribution_keys": int(te_delta["abs_mean_delta"].fillna(0).gt(1e-12).sum()),
        "te_r5p_max_abs_mean_delta": float(te_delta["abs_mean_delta"].max()) if len(te_delta) else 0.0,
        "wr_r15_simulation_delta_audit": str(WR_SIM_DELTA),
        "wr_r15_changed_distribution_keys": int(wr_delta["abs_mean_delta"].fillna(0).gt(1e-12).sum()),
        "wr_r15_max_abs_mean_delta": float(wr_delta["abs_mean_delta"].max()) if len(wr_delta) else 0.0,
        "wr_r15_entitlement_scope_gate": "PASS",
        "wr_r15_m38_wr1_anchor_preserved": bool(wr_audit["m38_wr1_anchor_preserved"]),
        "wr_r15_wr2plus_pool_preserved": bool(wr_audit["wr2plus_pool_preserved"]),
        "wr_r15_wr_room_mass_preserved": bool(wr_audit["wr_room_mass_preserved"]),
        "wr_r15_non_wr_entitlement_preserved": bool(wr_audit["non_wr_entitlement_preserved"]),
        "wr_r15_team_total_player_entitlement_preserved": bool(wr_audit["team_total_player_entitlement_preserved"]),
        "te_r5p_team_pool_preserved": bool(te_audit["team_te_pool_preserved"]),
    })
    ENTITLEMENT_AUDIT.write_text(json.dumps(entitlement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[target_entitlement_v1_plus_te_r5p_plus_wr_r15] " + json.dumps(entitlement, sort_keys=True))

    stateful = simulate_with_states(metrics, iterations=iterations, seed=seed)
    state_parity = v2._compare_results_exact(
        final, stateful, label="QB C2 state-capture seam", out_path=QB_C2_STATE_PARITY
    )
    if state_parity["changed_arrays"] != 0 or state_parity["max_mean_gap"] > 1e-12 or state_parity["max_element_gap"] > 1e-12:
        raise RuntimeError(f"QB C2 state-capture seam is not exact after WR-R15: {state_parity}")

    seasons = pd.to_numeric(metrics["season"], errors="coerce").dropna().astype(int).unique().tolist()
    weeks = pd.to_numeric(metrics["week"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"QB C2 production integration requires one season/week, got seasons={seasons} weeks={weeks}")
    selected, _, qb_payload = apply_qb_c2_selector(
        stateful, metrics, season=int(seasons[0]), week=int(weeks[0])
    )
    qb_payload.update({
        "state_capture_parity_keys": state_parity["keys"],
        "state_capture_changed_arrays": state_parity["changed_arrays"],
        "state_capture_max_mean_gap": state_parity["max_mean_gap"],
        "state_capture_max_element_gap": state_parity["max_element_gap"],
        "state_capture_parity_audit": str(QB_C2_STATE_PARITY),
        "te_r5p_consumed_before_c2": True,
        "wr_r15_consumed_before_c2": True,
        "wr_r15_model_version": "WR_R15_PRODUCTION_MODEL_V1",
        "explicit_entitlement_consumed_before_c2": True,
    })
    QB_C2_AUDIT_JSON.write_text(json.dumps(qb_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected.qb_distribution_audit = qb_payload
    print("[qb_c2_production_state_parity_after_wr_r15] " + json.dumps(state_parity, sort_keys=True))
    return selected


def main() -> int:
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = v2._install_provider_player_aliases_and_validate
    base._build_full_universe = _build_with_promoted_entitlement_specialists
    base.canonical_simulate = _simulate_promoted_stack
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
