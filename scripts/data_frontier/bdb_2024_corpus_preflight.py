"""Fail-closed file-level preflight for the official BDB 2024 corpus.

This checks only corpus packaging/completeness. It does not inspect outcomes,
change the frozen contact detector, or compute predictive metrics.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPECTED_TRACKING_WEEKS = tuple(range(1, 10))
REQUIRED_STATIC_FILES = ("plays.csv", "tackles.csv")
_TRACKING_RE = re.compile(r"^tracking_week_(\d+)\.csv$")


def inspect_corpus(input_dir: Path) -> dict:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"BDB 2024 input directory not found: {input_dir}")

    missing_static = [name for name in REQUIRED_STATIC_FILES if not (input_dir / name).is_file()]
    tracking_files = sorted(p for p in input_dir.iterdir() if p.is_file() and _TRACKING_RE.match(p.name))
    weeks = sorted(int(_TRACKING_RE.match(p.name).group(1)) for p in tracking_files)
    missing_weeks = sorted(set(EXPECTED_TRACKING_WEEKS) - set(weeks))
    unexpected_weeks = sorted(set(weeks) - set(EXPECTED_TRACKING_WEEKS))
    duplicate_weeks = sorted({w for w in weeks if weeks.count(w) > 1})
    zero_byte = sorted(
        p.name for p in [*(input_dir / n for n in REQUIRED_STATIC_FILES), *tracking_files]
        if p.exists() and p.stat().st_size == 0
    )
    passed = not (missing_static or missing_weeks or unexpected_weeks or duplicate_weeks or zero_byte)
    return {
        "contract": "BDB_2024_OFFICIAL_CORPUS_FILESET_V1",
        "passed": passed,
        "expected_tracking_weeks": list(EXPECTED_TRACKING_WEEKS),
        "observed_tracking_weeks": weeks,
        "missing_static_files": missing_static,
        "missing_tracking_weeks": missing_weeks,
        "unexpected_tracking_weeks": unexpected_weeks,
        "duplicate_tracking_weeks": duplicate_weeks,
        "zero_byte_files": zero_byte,
        "contact_detector_changed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate BDB 2024 corpus packaging before Phase-0 execution.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    try:
        report = inspect_corpus(args.input_dir)
    except FileNotFoundError as exc:
        report = {"contract": "BDB_2024_OFFICIAL_CORPUS_FILESET_V1", "passed": False, "failure": str(exc), "contact_detector_changed": False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
