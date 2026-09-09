#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

EXPECTED_DISPOSITION = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
EXPECTED_GATES = 28


def one(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boolish(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "pass"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--staged-root", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    args = ap.parse_args()

    src = args.source_root.resolve()
    dst = args.staged_root.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    src_disp = one(src, "r26q_disposition.json")
    src_gates = one(src, "r26q_gate_matrix.csv")
    dst_disp = one(dst, "r26q_disposition.json")

    original = json.loads(src_disp.read_text(encoding="utf-8"))
    gates = pd.read_csv(src_gates, low_memory=False)

    if original.get("disposition") != EXPECTED_DISPOSITION:
        raise RuntimeError(f"unexpected R26Q disposition: {original.get('disposition')}")
    if original.get("all_seal_gates_pass") is not True:
        raise RuntimeError("canonical R26Q all_seal_gates_pass is not true")
    if len(gates) != EXPECTED_GATES:
        raise RuntimeError(f"canonical R26Q gate row count {len(gates)} != {EXPECTED_GATES}")
    if "passed" not in gates.columns or not gates["passed"].map(boolish).all():
        raise RuntimeError("canonical R26Q gate matrix is not 28/28 PASS")

    if "gate_count" in original or "gate_pass_count" in original:
        raise RuntimeError("canonical R26Q unexpectedly already contains compatibility gate-count fields")

    staged = dict(original)
    staged["gate_count"] = EXPECTED_GATES
    staged["gate_pass_count"] = EXPECTED_GATES
    dst_disp.write_text(json.dumps(staged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    changed = {
        k: {"original": original.get(k, "<MISSING>"), "staged": staged.get(k, "<MISSING>")}
        for k in sorted(set(original) | set(staged))
        if original.get(k, "<MISSING>") != staged.get(k, "<MISSING>")
    }
    if set(changed) != {"gate_count", "gate_pass_count"}:
        raise RuntimeError(f"compatibility staging changed unauthorized fields: {sorted(changed)}")

    # All files other than the staged disposition must remain byte-identical.
    src_files = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst) for p in dst.rglob("*") if p.is_file())
    if src_files != dst_files:
        raise RuntimeError("staged compatibility copy file set differs from canonical parent")
    non_disp_mismatches = []
    disp_rel = src_disp.relative_to(src)
    for rel in src_files:
        if rel == disp_rel:
            continue
        if sha256_file(src / rel) != sha256_file(dst / rel):
            non_disp_mismatches.append(str(rel))
    if non_disp_mismatches:
        raise RuntimeError(f"non-disposition files changed during staging: {non_disp_mismatches}")

    audit = {
        "repair": "R26S_RUN1_R26Q_GATECOUNT_CONTRACT_MECHANICAL_REPAIR_V1",
        "canonical_disposition": original.get("disposition"),
        "canonical_all_seal_gates_pass": original.get("all_seal_gates_pass"),
        "canonical_gate_rows": int(len(gates)),
        "canonical_gate_pass_rows": int(gates["passed"].map(boolish).sum()),
        "canonical_disposition_sha256": sha256_file(src_disp),
        "staged_disposition_sha256": sha256_file(dst_disp),
        "changed_fields": changed,
        "non_disposition_files_byte_exact": True,
        "non_disposition_file_count": int(len(src_files) - 1),
        "scientific_evaluator_changed": False,
        "football_values_changed": False,
        "candidate_arrays_changed": False,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
