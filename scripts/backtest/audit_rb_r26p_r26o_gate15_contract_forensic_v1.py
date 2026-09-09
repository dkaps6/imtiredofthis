#!/usr/bin/env python3
"""Read-only forensic for R26O Gate-15 R22 audit-contract wiring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PASS = "R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED"
FAIL = "R26P_GATE15_FORENSIC_INCONCLUSIVE_NO_R26O_RERUN"
EXPECTED_R26O = "R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW"
EXPECTED_R22 = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS"
GATE15 = "15_r22_baseline_mean_parity"


def unique(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def add(rows: list[dict], test: str, passed: bool, evidence) -> bool:
    rows.append({
        "test": test,
        "passed": bool(passed),
        "evidence": json.dumps(evidence, sort_keys=True, default=str)
        if not isinstance(evidence, str) else evidence,
    })
    return bool(passed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26o-root", type=Path, required=True)
    ap.add_argument("--r22-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    r26o_disp = json.loads(unique(a.r26o_root.resolve(), "r26o_disposition.json").read_text())
    gates = pd.read_csv(unique(a.r26o_root.resolve(), "r26o_gate_matrix.csv"), low_memory=False)
    r22 = json.loads(unique(a.r22_root.resolve(), "rb_receiving_tail_production_audit.json").read_text())

    plan_path = Path("docs/research/RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1_FROZEN_PLAN.md")
    eval_path = Path("scripts/backtest/evaluate_rb_r26o_2026_week1_receptions_shadow_integration_v1.py")
    r22_path = Path("scripts/modeling/rb_receiving_tail_production_adapter_v1.py")
    for p in (plan_path, eval_path, r22_path):
        if not p.is_file():
            raise RuntimeError(f"missing forensic source {p}")

    plan = plan_path.read_text(encoding="utf-8")
    evaluator = eval_path.read_text(encoding="utf-8")
    r22_source = r22_path.read_text(encoding="utf-8")

    failed = gates.loc[~gates["passed"].astype(bool)].copy()
    gate15_rows = gates.loc[gates["gate"].astype(str).eq(GATE15)].copy()
    if len(gate15_rows) != 1:
        raise RuntimeError(f"expected one Gate 15 row, found {len(gate15_rows)}")

    evidence_raw = str(gate15_rows.iloc[0]["evidence"])
    try:
        evidence = json.loads(evidence_raw)
    except Exception:
        evidence = {}

    r22_max = r22.get("max_mean_delta")
    r22_max_finite = False
    try:
        r22_max_finite = bool(np.isfinite(float(r22_max)))
    except Exception:
        pass

    tests: list[dict] = []
    add(tests, "01_r26o_executed_fail_preserved", r26o_disp.get("disposition") == EXPECTED_R26O, r26o_disp.get("disposition"))
    add(tests, "02_exactly_one_failed_gate_is_15", len(failed) == 1 and str(failed.iloc[0]["gate"]) == GATE15, failed[["gate", "evidence"]].to_dict("records"))
    add(tests, "03_gate15_evidence_is_null_not_numeric_miss", evidence.get("mean_parity") is None and evidence.get("max_mean_delta") is None, evidence)
    add(tests, "04_frozen_plan_requires_r22_baseline_mean_parity", "15. baseline R22 receiving-yard mean-parity gate passes;" in plan, "frozen Gate 15 wording found")
    add(tests, "05_protected_r22_return_order_adapted_trace_payload", "return adapted, trace, payload" in r22_source, "return adapted, trace, payload")
    add(tests, "06_r26o_binds_second_return_as_r22_audit", "baseline_v4, r22_audit, _ = apply_rb_receiving_tail_production(" in evaluator, "baseline_v4, r22_audit, _")
    add(tests, "07_gate15_queries_nested_audit_semantics", "r22_audit.get(\"gates\", {}).get(\"mean_parity\")" in evaluator and "r22_audit.get(\"max_mean_delta\")" in evaluator, "Gate 15 reads audit payload fields")
    add(tests, "08_second_return_is_trace_not_audit_payload", "return adapted, trace, payload" in r22_source and "baseline_v4, r22_audit, _ = apply_rb_receiving_tail_production(" in evaluator, "R26O variable r22_audit receives protected R22 trace DataFrame")
    add(tests, "09_canonical_r22_adapter_pass", r22.get("disposition") == EXPECTED_R22, r22.get("disposition"))
    add(tests, "10_canonical_r22_mean_parity_true", r22.get("gates", {}).get("mean_parity") is True, r22.get("gates", {}).get("mean_parity"))
    add(tests, "11_canonical_r22_max_mean_delta_within_frozen_gate", r22_max_finite and float(r22_max) <= 1e-8, r22_max)
    add(tests, "12_canonical_r22_receptions_exact", r22.get("gates", {}).get("receptions_exact") is True, r22.get("gates", {}).get("receptions_exact"))
    marker_ok = a.protected_clean_marker.is_file() and a.protected_clean_marker.read_text().strip() == "PASS"
    add(tests, "13_protected_r22_code_boundary_clean", marker_ok, a.protected_clean_marker.read_text().strip() if a.protected_clean_marker.is_file() else "missing")
    wiring_only = all(t["passed"] for t in tests[3:13])
    add(tests, "14_correction_requires_evidence_object_only", wiring_only, "candidate/gates/threshold/seed/iterations need no change; only return-object exposure is defective")
    add(tests, "15_2026_outcomes_zero", int(r26o_disp.get("2026_outcomes_used", -1)) == 0, r26o_disp.get("2026_outcomes_used"))
    add(tests, "16_sportsbook_football_inputs_zero", int(r26o_disp.get("sportsbook_football_inputs_used", -1)) == 0, r26o_disp.get("sportsbook_football_inputs_used"))
    add(tests, "17_no_production_change_authority", r26o_disp.get("production_parameters_changed") is False and r26o_disp.get("production_promotion_authorized") is False, {"parameters_changed": r26o_disp.get("production_parameters_changed"), "promotion_authorized": r26o_disp.get("production_promotion_authorized")})

    frame = pd.DataFrame(tests)
    all_pass = bool(frame["passed"].all())
    disposition = PASS if all_pass else FAIL
    frame.to_csv(out / "r26p_forensic_matrix.csv", index=False)

    payload = {
        "candidate": "RB_R26P_R26O_GATE15_CONTRACT_FORENSIC_V1",
        "disposition": disposition,
        "all_forensic_tests_pass": all_pass,
        "tests_passed": int(frame["passed"].sum()),
        "tests_total": int(len(frame)),
        "r26o_executed_disposition_preserved": r26o_disp.get("disposition"),
        "r26o_failed_gate": str(failed.iloc[0]["gate"]) if len(failed) == 1 else None,
        "r26o_gate15_recorded_evidence": evidence,
        "canonical_r22_mean_parity": r22.get("gates", {}).get("mean_parity"),
        "canonical_r22_max_mean_delta": float(r22_max) if r22_max_finite else None,
        "canonical_r22_receptions_exact": r22.get("gates", {}).get("receptions_exact"),
        "mechanical_r26o_rerun_authorized": bool(all_pass),
        "authorized_correction": "expose protected R22 third return audit payload to unchanged frozen R26O Gate-15 lookup; no candidate/gate/threshold/seed/iteration changes" if all_pass else None,
        "2026_outcomes_used": 0,
        "sportsbook_football_inputs_used": 0,
        "production_change_authorized": False,
        "authority_note": "R26P cannot reinterpret the prior R26O run as PASS. A forensic PASS authorizes only a mechanical evidence-wiring correction and exact R26O rerun.",
    }
    (out / "r26p_disposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("R26P_DISPOSITION=" + disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
