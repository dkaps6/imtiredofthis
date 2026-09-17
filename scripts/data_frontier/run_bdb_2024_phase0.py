"""One-command Phase-0 BDB 2024 contact benchmark orchestration.

Order is deliberately fail-closed:
source normalization/QA -> frozen contact features -> geometry enrichment ->
structural integrity -> fidelity report.

No predictive metrics, model tuning, sportsbook logic, or production integration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PIPELINE_VERSION = "BDB_2024_PHASE0_PIPELINE_V1"


def _run(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Phase-0 stage failed closed: {module} exit={completed.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run isolated BDB 2024 Phase-0 contact benchmark pipeline.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        ("scripts.data_frontier.bdb_2024_artifact_qa", ["--input-dir", str(args.input_dir), "--out-dir", str(args.out_dir)]),
        ("scripts.data_frontier.bdb_2024_contact_enrichment", ["--artifact-dir", str(args.out_dir)]),
        ("scripts.data_frontier.bdb_2024_artifact_integrity", ["--artifact-dir", str(args.out_dir)]),
        ("scripts.data_frontier.bdb_2024_contact_fidelity", ["--artifact-dir", str(args.out_dir)]),
    ]
    completed: list[str] = []
    try:
        for module, stage_args in stages:
            _run(module, stage_args)
            completed.append(module)
    except RuntimeError as exc:
        status = {
            "pipeline_version": PIPELINE_VERSION,
            "passed": False,
            "completed_stages": completed,
            "failure": str(exc),
            "contact_detector_changed": False,
        }
        (args.out_dir / "phase0_pipeline_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(status, sort_keys=True), file=sys.stderr)
        return 2

    status = {
        "pipeline_version": PIPELINE_VERSION,
        "passed": True,
        "completed_stages": completed,
        "contact_detector_changed": False,
        "fidelity_report": str(args.out_dir / "contact_fidelity_report_v1.json"),
    }
    (args.out_dir / "phase0_pipeline_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
