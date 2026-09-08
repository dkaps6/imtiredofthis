#!/usr/bin/env python3
"""Certify the QB-only subset of canonical C2 evidence without changing production.

This evaluator is intentionally downstream of the frozen full-stack C2 experiment.
It does not re-score, tune, or reinterpret receiver results. Full C2 remains failed.
The only question is whether its QB distribution component independently clears a
predeclared gate while preserving the promoted M89/M90 mean anchor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--integration-result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    src = json.loads(a.integration_result.read_text(encoding="utf-8"))
    if src.get("migration") != "PASS_RECEIVING_CONSERVATION_INTEGRATION_V1":
        raise SystemExit("FATAL: wrong source migration")
    if src.get("production_changed") is not False or src.get("sportsbook_inputs_used") is not False:
        raise SystemExit("FATAL: source evidence violates frozen independence contract")
    if src.get("disposition") != "CONSERVATION_INTEGRATION_CANDIDATE_FAIL":
        raise SystemExit("FATAL: full C2 source disposition must remain recorded as FAIL")
    if not all(bool(v) for v in src.get("integrity_gates", {}).values()):
        raise SystemExit("FATAL: canonical source integrity gates are not all true")

    q = src["qb"]
    gates = {
        "m89_mean_anchor_gap_le001": float(q["mean_anchor_max_gap"]) <= 0.01,
        "qb_mean_mae_delta_le001": abs(float(q["c2_mean_mae"]) - float(q["b0_mean_mae"])) <= 0.01,
        "qb_crps_improvement_ge025": float(q["crps_improvement"]) >= 0.25,
        "qb_crps_bootstrap_ge090": float(q["bootstrap_probability"]) >= 0.90,
        "qb_80_coverage_error_not_worse_by_gt002": abs(float(q["c2_cover80"]) - 0.80) <= abs(float(q["b0_cover80"]) - 0.80) + 0.02,
        "qb_p90_abs_error_no_worse": float(q["c2_p90_abs_error"]) <= float(q["b0_p90_abs_error"]) + 1e-12,
        "qb_miss100_no_worse": float(q["c2_miss100"]) <= float(q["b0_miss100"]) + 1e-12,
        "canonical_qb_rows_exact_884": int(src.get("qb_rows", 0)) == 884,
    }
    disposition = "QB_C2_DISTRIBUTION_ONLY_CANDIDATE_PASS" if all(gates.values()) else "QB_C2_DISTRIBUTION_ONLY_CANDIDATE_FAIL"
    result = {
        "migration": "QB_C2_DISTRIBUTION_ONLY_V1",
        "disposition": disposition,
        "production_changed": False,
        "sportsbook_inputs_used": False,
        "source_full_c2_disposition": src["disposition"],
        "receiver_outputs_consumed": False,
        "promoted_mean_anchor": "M89_M90_UNCHANGED",
        "qb_rows": int(src["qb_rows"]),
        "qb": q,
        "gates": gates,
        "interpretation": "This certifies only the QB distribution-shape candidate. Full shared QB/receiver C2 remains failed and is not promoted.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
