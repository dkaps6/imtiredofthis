from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data_frontier import run_bdb_2024_phase0 as phase0


def test_run_raises_on_nonzero_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 7

    monkeypatch.setattr(phase0.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="failed closed"):
        phase0._run("scripts.data_frontier.fake_stage", [])


def test_integrity_failure_prevents_fidelity_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_run(module: str, args: list[str]) -> None:
        calls.append(module)
        if module == "scripts.data_frontier.bdb_2024_artifact_integrity":
            raise RuntimeError("Phase-0 stage failed closed: integrity")

    monkeypatch.setattr(phase0, "_run", fake_run)
    monkeypatch.setattr(
        phase0.argparse.ArgumentParser,
        "parse_args",
        lambda self: phase0.argparse.Namespace(input_dir=tmp_path / "input", out_dir=tmp_path / "out"),
    )

    assert phase0.main() == 2
    assert calls == [
        "scripts.data_frontier.bdb_2024_artifact_qa",
        "scripts.data_frontier.bdb_2024_contact_enrichment",
        "scripts.data_frontier.bdb_2024_artifact_integrity",
    ]
    assert "scripts.data_frontier.bdb_2024_contact_fidelity" not in calls

    status = json.loads((tmp_path / "out" / "phase0_pipeline_status.json").read_text())
    assert status["passed"] is False
    assert status["contact_detector_changed"] is False
    assert status["completed_stages"] == calls[:2]


def test_success_runs_fidelity_only_after_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(phase0, "_run", lambda module, args: calls.append(module))
    monkeypatch.setattr(
        phase0.argparse.ArgumentParser,
        "parse_args",
        lambda self: phase0.argparse.Namespace(input_dir=tmp_path / "input", out_dir=tmp_path / "out"),
    )

    assert phase0.main() == 0
    assert calls == [
        "scripts.data_frontier.bdb_2024_artifact_qa",
        "scripts.data_frontier.bdb_2024_contact_enrichment",
        "scripts.data_frontier.bdb_2024_artifact_integrity",
        "scripts.data_frontier.bdb_2024_contact_fidelity",
    ]
    status = json.loads((tmp_path / "out" / "phase0_pipeline_status.json").read_text())
    assert status["passed"] is True
    assert status["contact_detector_changed"] is False
