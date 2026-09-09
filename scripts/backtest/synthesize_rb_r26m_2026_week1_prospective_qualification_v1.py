#!/usr/bin/env python3
"""Frozen R26M evidence synthesis for 2026 Week-1 RB receiving qualification.

This evaluator reads only immutable parent disposition JSONs and a workflow-created
production-boundary marker. It generates no predictions, fits no model, and loads
no 2026 outcomes, sportsbook data, or same-week depth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CANDIDATE = "RB_R26M_2026_WEEK1_PROSPECTIVE_QUALIFICATION_SYNTHESIS_V1"
QUALIFIED = "2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED"
NOT_QUALIFIED = "2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_NOT_QUALIFIED"

PARENTS = {
    "R26": {
        "run_id": 34356222339,
        "artifact_id": 10106271075,
        "artifact_name": "rb-r26-vacancy-gated-r9-retrospective-v1",
        "digest": "sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e",
        "file": "r26_final_disposition.json",
        "disposition": "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW",
    },
    "R26E": {
        "run_id": 34368268224,
        "artifact_id": 10110785184,
        "artifact_name": "rb-r26e-week1-component-qualification-v1",
        "digest": "sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7",
        "file": "r26e_week1_qualification_disposition.json",
        "disposition": "WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW",
    },
    "R26J": {
        "run_id": 34374987828,
        "artifact_id": 10113466373,
        "artifact_name": "rb-r26j-2020-week1-comparability-source-audit-v1",
        "digest": "sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46",
        "file": "r26j_source_disposition.json",
        "disposition": "2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP",
    },
    "R26K": {
        "run_id": 34376961740,
        "artifact_id": 10114261724,
        "artifact_name": "rb-r26k-week1-allocation-mechanism-atlas-v1",
        "digest": "sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88",
        "file": "r26k_disposition.json",
        "disposition": "2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER",
    },
    "R26L": {
        "run_id": 34389455694,
        "artifact_id": 10119058769,
        "artifact_name": "rb-r26l-2026-week1-regime-transportability-v1",
        "digest": "sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46",
        "file": "r26l_disposition.json",
        "disposition": "2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION",
    },
}


def read_unique_json(root: Path, filename: str) -> dict:
    hits = sorted(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {filename} under {root}, found {len(hits)}")
    return json.loads(hits[0].read_text())


def gate_row(rows: list[dict], condition: str, passed: bool, evidence: str) -> bool:
    rows.append({"condition": condition, "passed": bool(passed), "evidence": str(evidence)})
    return bool(passed)


def false_gate_names(gates: dict) -> list[str]:
    return sorted(str(k) for k, v in gates.items() if not bool(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26e-root", type=Path, required=True)
    ap.add_argument("--r26j-root", type=Path, required=True)
    ap.add_argument("--r26k-root", type=Path, required=True)
    ap.add_argument("--r26l-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.protected_clean_marker.is_file() or args.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26M protected production boundary marker missing or invalid")

    roots = {
        "R26": args.r26_root,
        "R26E": args.r26e_root,
        "R26J": args.r26j_root,
        "R26K": args.r26k_root,
        "R26L": args.r26l_root,
    }
    data = {k: read_unique_json(roots[k], spec["file"]) for k, spec in PARENTS.items()}

    rows: list[dict] = []
    checks: list[bool] = []

    # Exact parent dispositions.
    for key, spec in PARENTS.items():
        got = str(data[key].get("disposition", ""))
        checks.append(gate_row(rows, f"{key}_exact_disposition", got == spec["disposition"], got))

    # R26: preserve failure and structural safety; do not rewrite it as a historical pass.
    r26 = data["R26"]
    r26_gates = dict(r26.get("gates", {}))
    checks.append(gate_row(rows, "R26_frozen_temporal_failure_preserved", r26_gates.get("15_no_season_worsens_more_than_2pct") is False, false_gate_names(r26_gates)))
    r26_fit = list(r26.get("fit_metadata", []))
    r26_structural_ok = bool(r26_fit) and all(
        int(x.get("sportsbook_inputs_used", -1)) == 0
        and int(x.get("future_outcomes_used_in_features", -1)) == 0
        and bool(x.get("strict_prior_fit"))
        and x.get("receiving_yard_mean_changed") is False
        and x.get("r22_changed") is False
        for x in r26_fit
    )
    checks.append(gate_row(rows, "R26_structural_safety_inherited", r26_structural_ok, f"fit_rows={len(r26_fit)}"))
    checks.append(gate_row(rows, "R26_no_shadow_or_production_authority", r26.get("prospective_2026_shadow_authorized") is False and r26.get("production_promotion_authorized") is False, "preserved"))

    # R26E: exact 19/20 Week-1 gate pattern, with 2020 as the sole harmful season.
    r26e = data["R26E"]
    e_gates = dict(r26e.get("gates", {}))
    e_false = false_gate_names(e_gates)
    e_pattern = len(e_gates) == 20 and e_false == ["14_no_w1_season_worsens_more_than_5pct"] and sum(bool(v) for v in e_gates.values()) == 19
    checks.append(gate_row(rows, "R26E_exact_19_of_20_gate_pattern", e_pattern, f"gate_count={len(e_gates)} false={e_false}"))
    e_seasons = list(r26e.get("temporal", {}).get("season_rows", []))
    by_season = {int(x.get("season")): float(x.get("relative_mae_change")) for x in e_seasons if x.get("season") is not None and x.get("relative_mae_change") is not None}
    e_temporal = set(by_season) == set(range(2020, 2026)) and by_season[2020] > 0 and all(by_season[y] < 0 for y in range(2021, 2026)) and int(r26e.get("temporal", {}).get("seasons_improved", -1)) == 5
    checks.append(gate_row(rows, "R26E_2020_only_harmful_week1_season", e_temporal, by_season))
    e_struct = dict(r26e.get("structural", {}))
    e_struct_ok = (
        int(e_struct.get("sportsbook_inputs_used", -1)) == 0
        and int(e_struct.get("future_outcomes_used", -1)) == 0
        and bool(e_struct.get("strict_prior_fit"))
        and float(e_struct.get("max_receiving_yard_mean_delta", 1.0)) == 0.0
        and float(e_struct.get("max_r22_authority_delta", 1.0)) == 0.0
        and r26e.get("r26_full_candidate_failure_preserved") is True
    )
    checks.append(gate_row(rows, "R26E_structural_and_parent_failure_preserved", e_struct_ok, e_struct))
    checks.append(gate_row(rows, "R26E_no_shadow_or_production_authority", r26e.get("prospective_2026_week1_shadow_authorized") is False and r26e.get("production_promotion_authorized") is False, "preserved"))

    # R26J: source-only evidence says 2020 is distinct, not deletable.
    r26j = data["R26J"]
    j_ok = (
        r26j.get("all_integrity_gates_pass") is True
        and int(r26j.get("independent_structurally_distinct_A_D_dimensions", -1)) == 10
        and int(r26j.get("independent_A_D_sections_with_distinction", -1)) == 4
        and r26j.get("exclude_2020_authorized") is False
        and int(r26j.get("future_outcome_features_used", -1)) == 0
        and int(r26j.get("sportsbook_inputs_used", -1)) == 0
        and r26j.get("predictions_regenerated") is False
        and r26j.get("r9_refit") is False
    )
    checks.append(gate_row(rows, "R26J_source_distinctness_and_integrity_inherited", j_ok, f"dims={r26j.get('independent_structurally_distinct_A_D_dimensions')} sections={r26j.get('independent_A_D_sections_with_distinction')}"))

    # R26K: no replicated historical router is justified.
    r26k = data["R26K"]
    k_ok = (
        r26k.get("all_integrity_gates_pass") is True
        and int(r26k.get("qualified_replicated_state_count", -1)) == 0
        and r26k.get("child_candidate_design_authorized") is False
        and r26k.get("exclude_2020_authorized") is False
        and r26k.get("predictions_regenerated") is False
        and r26k.get("r9_refit") is False
        and r26k.get("r22_changed") is False
        and r26k.get("receiving_yard_means_changed") is False
        and r26k.get("production_parameters_changed") is False
    )
    checks.append(gate_row(rows, "R26K_no_replicated_router_inherited", k_ok, f"qualified_replicated_state_count={r26k.get('qualified_replicated_state_count')}"))

    # R26L: actual 2026 source state satisfies the already-frozen modern-like rule.
    r26l = data["R26L"]
    d20 = float(r26l.get("mean_normalized_distance_to_2020", float("nan")))
    dmod = float(r26l.get("mean_normalized_distance_to_modern", float("nan")))
    l_ok = (
        r26l.get("all_integrity_gates_pass") is True
        and r26l.get("all_primary_2026_features_finite") is True
        and int(r26l.get("modern_closer_features", -1)) >= 5
        and dmod <= 0.75 * d20
        and int(r26l.get("beyond_2020_anomalous_direction_features", 99)) <= 2
        and r26l.get("prospective_qualification_design_authorized") is True
        and r26l.get("exclude_2020_authorized") is False
        and r26l.get("prospective_shadow_authorized") is False
        and r26l.get("production_promotion_authorized") is False
        and r26l.get("r9_refit") is False
        and r26l.get("predictions_regenerated") is False
        and int(r26l.get("sportsbook_inputs_used", -1)) == 0
        and r26l.get("same_week_depth_used") is False
        and r26l.get("r22_changed") is False
        and r26l.get("receiving_yard_means_changed") is False
        and r26l.get("production_parameters_changed") is False
    )
    checks.append(gate_row(rows, "R26L_modern_like_prospective_qualification_inherited", l_ok, f"modern_closer={r26l.get('modern_closer_features')} dmodern={dmod} d2020={d20} beyond={r26l.get('beyond_2020_anomalous_direction_features')}"))

    # Cross-parent authority ceiling. Historical negative evidence remains present.
    no_prod = all(x.get("production_promotion_authorized") is False for x in [r26, r26e, r26j, r26k, r26l])
    no_exclusion = r26j.get("exclude_2020_authorized") is False and r26k.get("exclude_2020_authorized") is False and r26l.get("exclude_2020_authorized") is False
    checks.append(gate_row(rows, "cross_parent_no_production_authority", no_prod, "all false"))
    checks.append(gate_row(rows, "cross_parent_2020_exclusion_unauthorized", no_exclusion, "R26J/R26K/R26L false"))
    checks.append(gate_row(rows, "protected_production_boundary_clean", True, args.protected_clean_marker.read_text().strip()))

    qualified = all(checks)
    disposition = QUALIFIED if qualified else NOT_QUALIFIED

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "r26m_parent_evidence_matrix.csv", index=False)

    disposition_json = {
        "candidate": CANDIDATE,
        "scientific_label": "IMMUTABLE_PARENT_EVIDENCE_SYNTHESIS_NO_NEW_PERFORMANCE_FIT",
        "disposition": disposition,
        "all_inherited_qualification_conditions_pass": qualified,
        "parent_artifacts_verified_by_workflow": True,
        "parents": {
            k: {
                "run_id": v["run_id"],
                "artifact_id": v["artifact_id"],
                "artifact_name": v["artifact_name"],
                "digest": v["digest"],
                "required_disposition": v["disposition"],
                "observed_disposition": data[k].get("disposition"),
            }
            for k, v in PARENTS.items()
        },
        "r26_failed_gate_preserved": "15_no_season_worsens_more_than_2pct" in false_gate_names(r26_gates),
        "r26e_failed_gate_count": len(e_false),
        "r26e_failed_gates": e_false,
        "r26e_seasons_improved": r26e.get("temporal", {}).get("seasons_improved"),
        "r26e_2020_relative_mae_change": by_season.get(2020),
        "r26j_distinct_dimensions": r26j.get("independent_structurally_distinct_A_D_dimensions"),
        "r26j_distinct_sections": r26j.get("independent_A_D_sections_with_distinction"),
        "r26k_qualified_replicated_state_count": r26k.get("qualified_replicated_state_count"),
        "r26l_modern_closer_features": r26l.get("modern_closer_features"),
        "r26l_mean_normalized_distance_to_2020": d20,
        "r26l_mean_normalized_distance_to_modern": dmod,
        "r26l_beyond_2020_anomalous_direction_features": r26l.get("beyond_2020_anomalous_direction_features"),
        "r26n_design_authorized": qualified,
        "prospective_shadow_activation_authorized": False,
        "production_promotion_authorized": False,
        "exclude_2020_authorized": False,
        "new_historical_router_authorized": False,
        "r9_refit": False,
        "predictions_regenerated": False,
        "2026_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "same_week_depth_used": False,
        "production_parameters_changed": False,
        "r22_changed": False,
        "receiving_yard_means_changed": False,
    }
    (out / "r26m_disposition.json").write_text(json.dumps(disposition_json, indent=2, sort_keys=True) + "\n")

    print(json.dumps(disposition_json, indent=2, sort_keys=True))
    print("R26M_DISPOSITION=" + disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
