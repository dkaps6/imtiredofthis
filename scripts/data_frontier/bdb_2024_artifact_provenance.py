"""Cryptographic provenance manifest for BDB 2024 Phase-0 artifacts.

Data-engineering only. SHA-256 binds downstream QA/fidelity output to exact
normalized inputs and upstream artifacts. The official-corpus preflight report
is part of the seal, so its source-file byte hashes are transitively bound to
every fidelity result. No predictive science or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROVENANCE_VERSION = "BDB_2024_ARTIFACT_PROVENANCE_V1"
REQUIRED_FILES = (
    "corpus_preflight_v1.json",
    "normalized/tracking.csv",
    "normalized/plays.csv",
    "normalized/tackles.csv",
    "rb_contact_features.csv",
    "benchmark_dispositions.csv",
    "qa_summary.json",
    "source_manifest.json",
    "rb_contact_features_enriched_v1.csv",
    "artifact_integrity_v1.json",
)
OPTIONAL_FILES = ("rb_contact_defender_geometry_v1.csv",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(artifact_dir: Path) -> dict:
    missing = [rel for rel in REQUIRED_FILES if not (artifact_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"required provenance artifacts missing: {missing}")
    integrity = json.loads((artifact_dir / "artifact_integrity_v1.json").read_text(encoding="utf-8"))
    if integrity.get("passed") is not True:
        raise RuntimeError("refusing provenance seal: structural integrity did not pass")
    corpus_preflight = json.loads((artifact_dir / "corpus_preflight_v1.json").read_text(encoding="utf-8"))
    if corpus_preflight.get("passed") is not True or not corpus_preflight.get("source_files"):
        raise RuntimeError("refusing provenance seal: official-corpus preflight/fingerprint did not pass")
    files = {}
    for rel in (*REQUIRED_FILES, *OPTIONAL_FILES):
        path = artifact_dir / rel
        if path.is_file():
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_canonical = json.dumps(corpus_preflight["source_files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "provenance_version": PROVENANCE_VERSION,
        "scope": "phase0_data_fidelity_only",
        "hash_algorithm": "sha256",
        "passed_structural_integrity": True,
        "passed_official_corpus_preflight": True,
        "contact_detector_changed": False,
        "files": files,
        "source_corpus_sha256": hashlib.sha256(source_canonical).hexdigest(),
        "artifact_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "guardrail": "Any source-corpus or upstream artifact byte change invalidates this seal; rerun preflight, integrity, provenance, and fidelity before reporting.",
    }


def verify_manifest(artifact_dir: Path, manifest: dict) -> dict:
    failures = []
    for rel, meta in manifest.get("files", {}).items():
        path = artifact_dir / rel
        if not path.is_file():
            failures.append({"file": rel, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != meta.get("sha256"):
            failures.append({"file": rel, "reason": "sha256_mismatch", "expected": meta.get("sha256"), "actual": actual})
    return {"passed": not failures, "failure_count": len(failures), "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser(description="Seal or verify BDB 2024 Phase-0 artifact provenance.")
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    out = args.artifact_dir / "artifact_provenance_v1.json"
    if args.verify:
        if not out.is_file():
            print(json.dumps({"passed": False, "failure": "provenance manifest missing"}, sort_keys=True))
            return 2
        report = verify_manifest(args.artifact_dir, json.loads(out.read_text(encoding="utf-8")))
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 2
    try:
        manifest = build_manifest(args.artifact_dir)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "failure": str(exc)}, sort_keys=True))
        return 2
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": True, "provenance_version": PROVENANCE_VERSION, "source_corpus_sha256": manifest["source_corpus_sha256"], "artifact_set_sha256": manifest["artifact_set_sha256"], "output": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
