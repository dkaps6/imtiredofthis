#!/usr/bin/env python3
"""Operational wrapper for one live RB-PD2 prospective lock attempt.

This is research-only orchestration around the frozen §7/§8/§9 primitives.
It selects the capture session produced by the *current* Full Slate invocation,
assembles the immutable candidate lock, and always writes a status receipt.
A research failure is therefore inspectable without needing to abort or mutate
canonical production pricing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.research import assemble_rb_pd2_forward_locks_v1 as assembler

DEFAULT_CAPTURE_ROOT = Path("data/research/rb_pd2_shadow")
STATUS_FILE = "rb_pd2_forward_live_lock_status.json"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_capture_session(
    root: Path,
    *,
    workflow_run_id: str,
    code_sha: str,
) -> Path:
    """Return exactly one finalized capture session for this workflow invocation."""
    root = Path(root)
    matches: list[Path] = []
    for receipt_path in sorted(root.glob("*/session_receipt.json")):
        try:
            receipt = _read_json(receipt_path)
        except Exception:
            continue
        provenance = receipt.get("provenance") or {}
        if str(provenance.get("workflow_run_id") or "").strip() != str(workflow_run_id).strip():
            continue
        if str(provenance.get("code_sha") or "").strip().lower() != str(code_sha).strip().lower():
            continue
        matches.append(receipt_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one RB PD2 capture session for run={workflow_run_id} sha={code_sha}; "
            f"found={len(matches)}"
        )
    return matches[0]


def run(
    *,
    history_root: Path,
    prospective_start_utc: str,
    out_dir: Path,
    capture_root: Path = DEFAULT_CAPTURE_ROOT,
    workflow_run_id: str,
    code_sha: str,
    history_run_id: str,
) -> tuple[int, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "version": "RB_PD2_LIVE_LOCK_OPERATION_V1",
        "valid": False,
        "workflow_run_id": str(workflow_run_id),
        "production_code_sha": str(code_sha),
        "history_run_id": str(history_run_id),
        "prospective_start_utc": str(prospective_start_utc),
        "production_changed": False,
        "sportsbook_inputs_used_in_candidate": False,
        "outcome_present_at_lock": False,
    }
    rc = 2
    try:
        session_dir = find_capture_session(
            capture_root,
            workflow_run_id=str(workflow_run_id),
            code_sha=str(code_sha),
        )
        history_root = Path(history_root)
        history_state = history_root / "history" / "rb_pd2_forward_history_state.csv"
        history_manifest = history_root / "history" / "rb_pd2_forward_history_manifest.json"
        if not history_state.exists() or not history_manifest.exists():
            raise RuntimeError(f"forward-history artifact is incomplete under {history_root}")

        result = assembler.assemble(
            session_dir=session_dir,
            history_state_path=history_state,
            history_manifest_path=history_manifest,
            prospective_start_utc=prospective_start_utc,
            out_dir=out_dir,
        )
        status.update({
            "source_capture_session_dir": str(session_dir),
            "lock_receipt": str(out_dir / assembler.LOCK_RECEIPT),
            "locked_rows": int(result.get("locked_rows", 0)),
            "capture_rows": int(result.get("capture_rows", 0)),
            "population_exclusions": int(len(result.get("population_exclusions") or [])),
            "integrity_failures": int(len(result.get("integrity_failures") or [])),
            "valid": bool(result.get("valid")),
        })
        rc = 0 if status["valid"] else 2
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        rc = 2

    (out_dir / STATUS_FILE).write_text(
        json.dumps(status, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return rc, status


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history-root", type=Path, required=True)
    p.add_argument("--history-run-id", required=True)
    p.add_argument("--prospective-start-utc", required=True)
    p.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    run_id = str(os.getenv("GITHUB_RUN_ID") or "").strip()
    code_sha = str(os.getenv("GITHUB_SHA") or "").strip()
    if not run_id or not code_sha:
        raise SystemExit("GITHUB_RUN_ID and GITHUB_SHA are required for live lock provenance")

    rc, status = run(
        history_root=a.history_root,
        prospective_start_utc=a.prospective_start_utc,
        out_dir=a.out_dir,
        capture_root=a.capture_root,
        workflow_run_id=run_id,
        code_sha=code_sha,
        history_run_id=a.history_run_id,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
