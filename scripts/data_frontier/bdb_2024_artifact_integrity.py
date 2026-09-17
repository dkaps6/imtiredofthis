"""Fail-closed structural integrity gate for BDB 2024 Phase-0 artifacts.

Data engineering only. This module does not alter the frozen contact detector,
tune thresholds, run predictive metrics, or touch production/sportsbook logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTEGRITY_VERSION = "BDB_2024_ARTIFACT_INTEGRITY_V1"
PLAY_KEY = ["gameId", "playId"]
# Enrichment V1 emits `defenderId`; keep the gate bound to the persisted schema.
DEFENDER_KEY = ["gameId", "playId", "defenderId"]


def _duplicate_count(df: pd.DataFrame, key: list[str]) -> int:
    if df.empty:
        return 0
    if any(c not in df.columns for c in key):
        return -1
    return int(df.duplicated(key, keep=False).sum())


def _key_set(df: pd.DataFrame, key: list[str]) -> set[tuple]:
    if df.empty or any(c not in df.columns for c in key):
        return set()
    return set(map(tuple, df[key].drop_duplicates().itertuples(index=False, name=None)))


def _finite_failures(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        out[col] = int((x.notna() & ~np.isfinite(x)).sum())
    return out


def validate_artifacts(enriched: pd.DataFrame, defenders: pd.DataFrame, dispositions: pd.DataFrame, qa_summary: dict) -> dict:
    failures: list[dict] = []
    required = {"enriched": (enriched, PLAY_KEY), "dispositions": (dispositions, PLAY_KEY)}
    for name, (df, key) in required.items():
        missing = [c for c in key if c not in df.columns]
        if missing:
            failures.append({"check": f"{name}_required_key", "detail": f"missing columns: {missing}"})
            continue
        dupes = _duplicate_count(df, key)
        if dupes:
            failures.append({"check": f"{name}_key_unique", "detail": f"duplicate rows participating in duplicate keys: {dupes}"})

    scoreable = dispositions.loc[dispositions.get("benchmark_disposition", pd.Series(index=dispositions.index, dtype=object)).eq("SCOREABLE")]
    if not scoreable.empty and all(c in scoreable.columns for c in PLAY_KEY) and all(c in enriched.columns for c in PLAY_KEY):
        scoreable_keys, enriched_keys = _key_set(scoreable, PLAY_KEY), _key_set(enriched, PLAY_KEY)
        missing_enriched, extra_enriched = scoreable_keys - enriched_keys, enriched_keys - scoreable_keys
        if missing_enriched or extra_enriched:
            failures.append({"check": "scoreable_enriched_key_parity", "detail": f"missing_enriched={len(missing_enriched)} extra_enriched={len(extra_enriched)}"})

    if not defenders.empty:
        missing_def_key = [c for c in DEFENDER_KEY if c not in defenders.columns]
        if missing_def_key:
            failures.append({"check": "defender_required_key", "detail": f"missing columns: {missing_def_key}"})
        else:
            dupes = _duplicate_count(defenders, DEFENDER_KEY)
            if dupes:
                failures.append({"check": "defender_key_unique", "detail": f"duplicate rows participating in duplicate keys: {dupes}"})
            orphan_keys = _key_set(defenders, PLAY_KEY) - _key_set(enriched, PLAY_KEY)
            if orphan_keys:
                failures.append({"check": "defender_play_referential_integrity", "detail": f"orphan play keys: {len(orphan_keys)}"})

    range_specs = {
        "firstContactCarrierSidelineDistanceYards": (0.0, 26.65),
        "firstContactMinPursuitAngleErrorDeg": (0.0, 180.0),
        "firstContactMeanPursuitAngleErrorDeg": (0.0, 180.0),
    }
    for col, (lo, hi) in range_specs.items():
        if col in enriched.columns:
            x = pd.to_numeric(enriched[col], errors="coerce")
            bad = int((x.notna() & ((x < lo) | (x > hi))).sum())
            if bad:
                failures.append({"check": f"range_{col}", "detail": f"out_of_range_rows={bad} allowed=[{lo},{hi}]"})
    if "pursuitAngleErrorDeg" in defenders.columns:
        x = pd.to_numeric(defenders["pursuitAngleErrorDeg"], errors="coerce")
        bad = int((x.notna() & ((x < 0.0) | (x > 180.0))).sum())
        if bad:
            failures.append({"check": "range_defender_pursuitAngleErrorDeg", "detail": f"out_of_range_rows={bad}"})

    finite = {}
    finite.update({f"enriched.{k}": v for k, v in _finite_failures(enriched, ["firstContactCarrierSidelineDistanceYards", "firstContactMaxClosingSpeedProxyYdsPerSec", "firstContactMeanClosingSpeedProxyYdsPerSec", "firstContactMinPursuitAngleErrorDeg", "firstContactMeanPursuitAngleErrorDeg"]).items()})
    finite.update({f"defenders.{k}": v for k, v in _finite_failures(defenders, ["closingSpeedProxyYdsPerSec", "pursuitAngleErrorDeg"]).items()})
    for col, count in finite.items():
        if count:
            failures.append({"check": f"finite_{col}", "detail": f"nonfinite_rows={count}"})

    qa_feature_version = qa_summary.get("feature_version")
    if not qa_feature_version:
        failures.append({"check": "qa_feature_version_present", "detail": "qa_summary.feature_version missing"})
    if qa_summary.get("contact_detector_changed") is not False:
        failures.append({"check": "contact_detector_immutability", "detail": "qa_summary.contact_detector_changed must be false"})
    normalized = qa_summary.get("normalized_artifacts", {})
    for table in ("tracking", "plays", "tackles"):
        meta = normalized.get(table)
        if not isinstance(meta, dict) or "rows" not in meta or "columns" not in meta:
            failures.append({"check": f"normalized_manifest_{table}", "detail": "missing rows/columns manifest metadata"})

    return {"integrity_version": INTEGRITY_VERSION, "scope": "structural_data_integrity_only", "passed": not failures, "failure_count": len(failures), "failures": failures, "counts": {"enriched_rows": int(len(enriched)), "defender_rows": int(len(defenders)), "disposition_rows": int(len(dispositions)), "scoreable_rows": int(len(scoreable))}, "feature_version": qa_feature_version, "contact_detector_changed": False, "guardrail": "A failed integrity report is a hard stop for downstream fidelity reporting; do not repair by dropping or imputing corrupt rows silently."}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed BDB 2024 Phase-0 artifact integrity gate.")
    ap.add_argument("--artifact-dir", type=Path, required=True)
    args = ap.parse_args()
    enriched = pd.read_csv(args.artifact_dir / "rb_contact_features_enriched_v1.csv")
    defenders_path = args.artifact_dir / "rb_contact_defender_geometry_v1.csv"
    defenders = pd.read_csv(defenders_path) if defenders_path.exists() else pd.DataFrame()
    dispositions = pd.read_csv(args.artifact_dir / "benchmark_dispositions.csv")
    qa_summary = json.loads((args.artifact_dir / "qa_summary.json").read_text(encoding="utf-8"))
    report = validate_artifacts(enriched, defenders, dispositions, qa_summary)
    out = args.artifact_dir / "artifact_integrity_v1.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"integrity_version": INTEGRITY_VERSION, "passed": report["passed"], "failure_count": report["failure_count"], "output": str(out)}, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
