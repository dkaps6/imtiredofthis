"""Guards for the replay-input preservation.

Its whole purpose is to catch the stripping regression that cost the week-7 parity
replay, on the run that causes it. So that is the central test.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.operations import preserve_replay_input_set_v1 as P


def _complete_artifact(root: Path) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    for rel, cols in P.REQUIRED.items():
        pd.DataFrame({c: [1.0, 2.0] for c in cols}).to_csv(root / rel, index=False)
    return root


def test_required_set_names_the_columns_that_broke_the_w7_replay():
    ml = P.REQUIRED["data/model_ml_diagnostics.csv"]
    assert "ml_pass_yards" in ml and "ml_available" in ml
    assert "state_available" in P.REQUIRED["data/model_state_diagnostics.csv"]


def test_complete_artifact_archives_and_reports_complete(tmp_path):
    art = _complete_artifact(tmp_path / "art")
    out = tmp_path / "replay_inputs"
    m = P.preserve(art, out, "2026_wk08", run_id="123", git_sha="abc", week="8")
    assert m["complete"] is True
    assert not m["required_missing"] and not m["required_column_stripped"]
    assert len(m["files"]) == len(P.REQUIRED)
    for rel, info in m["files"].items():
        gz = out / "2026_wk08" / info["archived_as"]
        assert gz.exists()
        # round-trips to the same bytes
        with gzip.open(gz, "rb") as fh:
            assert len(fh.read()) == info["bytes_raw"]
        assert len(info["sha256_raw"]) == 64


def test_a_STRIPPED_required_file_is_refused_and_flagged(tmp_path):
    """The exact week-7 regression: identity columns only."""
    art = _complete_artifact(tmp_path / "art")
    pd.DataFrame({"player": ["x"], "player_clean_key": ["x"], "team": ["KC"],
                  "season": [2026], "week": [7], "position": ["WR"]}
                 ).to_csv(art / "data/model_ml_diagnostics.csv", index=False)
    m = P.preserve(art, tmp_path / "out", "2026_wk07", run_id="1")
    assert m["complete"] is False
    s = m["required_column_stripped"]["data/model_ml_diagnostics.csv"]
    assert "ml_pass_yards" in s["missing_columns"]
    assert s["columns_present"] == 6
    assert "data/model_ml_diagnostics.csv" not in m["files"], "must not archive a stripped file"


def test_a_missing_required_file_is_recorded_not_silently_skipped(tmp_path):
    art = _complete_artifact(tmp_path / "art")
    (art / "data/metrics_ready.csv").unlink()
    m = P.preserve(art, tmp_path / "out", "2026_wk08", run_id="1")
    assert "data/metrics_ready.csv" in m["required_missing"]
    assert m["complete"] is False


def test_optional_files_are_taken_when_present_and_skipped_when_not(tmp_path):
    art = _complete_artifact(tmp_path / "art")
    pd.DataFrame({"market": ["rec_yards"]}).to_csv(art / "data/model_ensemble_weights.csv", index=False)
    m = P.preserve(art, tmp_path / "out", "2026_wk08", run_id="1")
    assert "data/model_ensemble_weights.csv" in m["files"]
    assert "data/weather_week.csv" not in m["files"]
    assert m["complete"] is True, "a missing OPTIONAL file must not mark the set incomplete"


def test_the_directory_warns_it_is_not_research_safe(tmp_path):
    art = _complete_artifact(tmp_path / "art")
    out = tmp_path / "replay_inputs"
    m = P.preserve(art, out, "2026_wk08", run_id="1")
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "NOT denylist-stripped" in readme
    assert "Do not use anything here to construct a projection" in readme
    assert "never a projection input" in m["warning"]


def test_manifest_carries_provenance_for_a_replay_to_pin(tmp_path):
    art = _complete_artifact(tmp_path / "art")
    out = tmp_path / "out"
    P.preserve(art, out, "2026_wk08", run_id="40551229604", git_sha="deadbeef", week="8")
    m = json.loads((out / "2026_wk08" / "manifest.json").read_text(encoding="utf-8"))
    assert m["source_run_id"] == "40551229604" and m["source_git_sha"] == "deadbeef"
    assert m["version"] == P.VERSION


def test_the_archive_workflow_actually_calls_it():
    wf = Path(".github/workflows/archive-market-track-record-v1.yml").read_text(encoding="utf-8")
    assert "preserve_replay_input_set_v1.py" in wf
    assert "--out-dir data/market_track_record/replay_inputs" in wf
