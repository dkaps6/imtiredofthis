import json

import pytest

from scripts.data_frontier.bdb_2024_artifact_provenance import build_manifest, verify_manifest


def _write_required(root):
    files = [
        "normalized/tracking.csv", "normalized/plays.csv", "normalized/tackles.csv",
        "rb_contact_features.csv", "benchmark_dispositions.csv", "qa_summary.json",
        "source_manifest.json", "rb_contact_features_enriched_v1.csv",
    ]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel + "\n", encoding="utf-8")
    (root / "corpus_preflight_v1.json").write_text(
        json.dumps({"passed": True, "source_files": [{"name": "plays.csv", "size_bytes": 1, "sha256": "a" * 64}]}),
        encoding="utf-8",
    )
    (root / "artifact_integrity_v1.json").write_text(json.dumps({"passed": True}), encoding="utf-8")


def test_manifest_detects_post_seal_mutation(tmp_path):
    _write_required(tmp_path)
    manifest = build_manifest(tmp_path)
    assert verify_manifest(tmp_path, manifest)["passed"] is True
    (tmp_path / "rb_contact_features.csv").write_text("mutated\n", encoding="utf-8")
    report = verify_manifest(tmp_path, manifest)
    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "sha256_mismatch"


def test_manifest_binds_official_corpus_preflight(tmp_path):
    _write_required(tmp_path)
    manifest = build_manifest(tmp_path)
    assert "corpus_preflight_v1.json" in manifest["files"]
    assert manifest["source_corpus_sha256"]
    (tmp_path / "corpus_preflight_v1.json").write_text(
        json.dumps({"passed": True, "source_files": [{"name": "plays.csv", "size_bytes": 2, "sha256": "b" * 64}]}),
        encoding="utf-8",
    )
    report = verify_manifest(tmp_path, manifest)
    assert report["passed"] is False
    assert any(
        failure["file"] == "corpus_preflight_v1.json" and failure["reason"] == "sha256_mismatch"
        for failure in report["failures"]
    )


def test_manifest_refuses_failed_integrity(tmp_path):
    _write_required(tmp_path)
    (tmp_path / "artifact_integrity_v1.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="structural integrity"):
        build_manifest(tmp_path)


def test_manifest_refuses_failed_corpus_preflight(tmp_path):
    _write_required(tmp_path)
    (tmp_path / "corpus_preflight_v1.json").write_text(json.dumps({"passed": False, "source_files": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="official-corpus preflight"):
        build_manifest(tmp_path)


def test_manifest_requires_normalized_sources(tmp_path):
    _write_required(tmp_path)
    (tmp_path / "normalized/tracking.csv").unlink()
    with pytest.raises(FileNotFoundError, match="normalized/tracking.csv"):
        build_manifest(tmp_path)
