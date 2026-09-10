#!/usr/bin/env python3
"""Finalize frozen gate 35 after immutable evidence upload metadata exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRE = Path("data/current_player_availability_35gate_preliminary.json")
OUT = Path("data/current_player_availability_35gate_result.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--artifact-digest", required=True)
    args = ap.parse_args()
    if not PRE.is_file():
        raise RuntimeError("preliminary gates 1-34 result missing")
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    if not pre.get("all_1_34_pass") or int(pre.get("passed_1_34", 0)) != 34:
        raise RuntimeError("cannot finalize gate35 when gates 1-34 did not all pass")
    lineage = {
        "branch": args.branch,
        "head": args.head,
        "run": str(args.run),
        "job": args.job,
        "artifact_id": str(args.artifact_id),
        "artifact_digest": args.artifact_digest,
    }
    ok = all(str(v).strip() for v in lineage.values()) and str(args.artifact_digest).startswith("sha256:")
    gates = list(pre["gates"]) + [{
        "gate": 35,
        "name": "result record captures immutable execution lineage and disposition",
        "pass": bool(ok),
        "evidence": lineage,
    }]
    passed = int(sum(bool(g.get("pass")) for g in gates))
    disposition = (
        "CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION"
        if ok and passed == 35
        else "CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION"
    )
    payload = {
        "disposition": disposition,
        "frozen_gate_count": 35,
        "passed_gates": passed,
        "failed_gates": 35 - passed,
        "production_promoted": False,
        "sportsbook_used_to_define_football": False,
        "evidence_lineage": lineage,
        "gates": gates,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if disposition != "CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION":
        raise RuntimeError("first valid 35-gate result is not promotion-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
