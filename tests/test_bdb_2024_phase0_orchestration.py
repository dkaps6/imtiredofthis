from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data_frontier import run_bdb_2024_phase0 as phase0


def _patch_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        phase0.argparse.ArgumentParser,
        "parse_args",
        lambda self: phase0.argparse.Namespace(input_dir=tmp_path / "input", out_dir=tmp_path / "out"),
    )


def test_run_raises_on_nonzero_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 7

    monkeypatch.setattr(phase0.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="failed closed"):
        phase0._run("scripts.data_frontier.fake_stage", [])


def test_preflight_failure_prevents_all_downstream_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run(module: str, args: list[str]) -> None:
        calls.append(module)
        if module == "scripts.data_frontier.bdb_2024_corpus_preflight":
            raise RuntimeError("Phase-0 stage failed closed: corpus preflight")

    monkeypatch.setattr(phase0, "_run", fake_run)
    _patch_args(tmp_path, monkeypatch)
    assert phase0.main() == 2
    assert calls == ["scripts.data_frontier.bdb_2024_corpus_preflight"]


def test_integrity_failure_prevents_provenance_and_fidelity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run(module: str, args: list[str]) -> None:
        calls.append(module)
        if module == "scripts.data_frontier.bdb_2024_artifact_integrity":
            raise RuntimeError("Phase-0 stage failed closed: integrity")

    monkeypatch.setattr(phase0, "_run", fake_run)
    _patch_args(tmp_path, monkeypatch)
    assert phase0.main() == 2
    assert calls == [
        "scripts.data_frontier.bdb_2024_corpus_preflight",
        "scripts.data_frontier.bdb_2024_artifact_qa",
        "scripts.data_frontier.bdb_2024_contact_enrichment",
        "scripts.data_frontier.bdb_2024_artifact_integrity",
    ]
    assert "scripts.data_frontier.bdb_2024_artifact_provenance" not in calls
    assert "scripts.data_frontier.bdb_2024_contact_fidelity" not in calls
    status = json.loads((tmp_path / "out" / "phase0_pipeline_status.json").read_text())
    assert status["passed"] is False
    assert status["contact_detector_changed"] is False
    assert status["completed_stages"] == calls[:3]


def test_success_runs_preflight_seal_fidelity_then_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    artifact_hash = "a" * 64

    def fake_run(module: str, args: list[str]) -> None:
        calls.append((module, tuple(args)))
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        if module == "scripts.data_frontier.bdb_2024_artifact_provenance" and "--verify" not in args:
            (out / "artifact_provenance_v1.json").write_text(json.dumps({"artifact_set_sha256": artifact_hash}), encoding="utf-8")
        if module == "scripts.data_frontier.bdb_2024_contact_fidelity":
            (out / "contact_fidelity_report_v1.json").write_text(json.dumps({"upstream_artifact_set_sha256": artifact_hash}), encoding="utf-8")

    monkeypatch.setattr(phase0, "_run", fake_run)
    _patch_args(tmp_path, monkeypatch)
    assert phase0.main() == 0
    modules = [module for module, _ in calls]
    assert modules == [
        "scripts.data_frontier.bdb_2024_corpus_preflight",
        "scripts.data_frontier.bdb_2024_artifact_qa",
        "scripts.data_frontier.bdb_2024_contact_enrichment",
        "scripts.data_frontier.bdb_2024_artifact_integrity",
        "scripts.data_frontier.bdb_2024_artifact_provenance",
        "scripts.data_frontier.bdb_2024_contact_fidelity",
        "scripts.data_frontier.bdb_2024_artifact_provenance",
    ]
    assert "--verify" not in calls[4][1]
    assert "--verify" in calls[6][1]
    status = json.loads((tmp_path / "out" / "phase0_pipeline_status.json").read_text())
    assert status["passed"] is True
    assert status["artifact_set_sha256"] == artifact_hash
    assert status["post_fidelity_provenance_verified"] is True
    assert status["contact_detector_changed"] is False


def test_fidelity_provenance_sha_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed_hash = "b" * 64

    def fake_run(module: str, args: list[str]) -> None:
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        if module == "scripts.data_frontier.bdb_2024_artifact_provenance" and "--verify" not in args:
            (out / "artifact_provenance_v1.json").write_text(json.dumps({"artifact_set_sha256": sealed_hash}), encoding="utf-8")
        if module == "scripts.data_frontier.bdb_2024_contact_fidelity":
            (out / "contact_fidelity_report_v1.json").write_text(json.dumps({"upstream_artifact_set_sha256": "c" * 64}), encoding="utf-8")

    monkeypatch.setattr(phase0, "_run", fake_run)
    _patch_args(tmp_path, monkeypatch)
    assert phase0.main() == 2
    status = json.loads((tmp_path / "out" / "phase0_pipeline_status.json").read_text())
    assert status["passed"] is False
    assert "does not match" in status["failure"]
    assert status["contact_detector_changed"] is False
