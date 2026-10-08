"""Frozen Gate-0 contract checks: never grade historical or prospective outcomes."""
import json
from pathlib import Path

import pytest

from scripts.research.audit_player_trajectory_historical_oos_gate_v1 import gate, read_model


def model(tmp_path: Path, name: str, version: str, seasons: list[int]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"model_version": version, "training_seasons": seasons}), encoding="utf-8")
    return path


def pair(tmp_path, years=(2022, 2023, 2024, 2025)):
    te = model(tmp_path, "te.json", "TE_R5P_PRODUCTION_MODEL_V1", list(years))
    wr = model(tmp_path, "wr.json", "WR_R15_PRODUCTION_MODEL_V1", list(years))
    return te, wr


def test_overlapping_production_seasons_fail_closed(tmp_path):
    te, wr = pair(tmp_path)
    result = gate(te_path=te, wr_path=wr)
    assert result["status"] == "HISTORICAL_INTEGRATION_DIAGNOSTIC_ONLY__SPECIALIST_TRAINING_OVERLAP"
    assert [r["season"] for r in result["season_dispositions"]] == [2023, 2024, 2025]
    assert all(r["overlapping_specialists"] == ["te", "wr"] for r in result["season_dispositions"])
    assert all(not r["independent_oos_validated"] for r in result["season_dispositions"])
    assert result["historical_grading_executed"] is False
    assert result["independent_oos_certified"] is False


def test_mixed_overlap_cannot_earn_oos(tmp_path):
    te = model(tmp_path, "te.json", "TE_R5P_PRODUCTION_MODEL_V1", [2022, 2023])
    wr = model(tmp_path, "wr.json", "WR_R15_PRODUCTION_MODEL_V1", [2022, 2024])
    result = gate(te_path=te, wr_path=wr, target_seasons=(2023, 2024, 2025))
    assert result["status"].startswith("HISTORICAL_INTEGRATION_DIAGNOSTIC_ONLY")
    assert result["season_dispositions"][0]["overlapping_specialists"] == ["te"]
    assert result["season_dispositions"][1]["overlapping_specialists"] == ["wr"]
    assert result["season_dispositions"][2]["declared_training_membership_nonoverlap"] is True
    assert not result["independent_oos_certified"]


def test_nonoverlap_still_requires_asof_and_source_parity(tmp_path):
    te, wr = pair(tmp_path, years=(2022, 2023))
    result = gate(te_path=te, wr_path=wr, target_seasons=(2024, 2025))
    assert result["status"] == "SOURCE_PARITY_AND_AS_OF_GATE_REQUIRED__MEMBERSHIP_NONOVERLAP"
    assert result["independent_oos_certified"] is False


def test_zero_outcome_and_production_contract(tmp_path):
    te, wr = pair(tmp_path)
    result = gate(te_path=te, wr_path=wr)
    assert result["sportsbook_inputs_used"] is False
    assert result["target_outcomes_read"] is False
    assert result["week5_2026_outcomes_read"] is False
    assert result["production_changed"] is False
    assert result["parameters_fit"] == 0


def test_missing_wrong_or_malformed_specialist_fails_closed(tmp_path):
    te, wr = pair(tmp_path)
    with pytest.raises(RuntimeError, match="missing frozen"):
        gate(te_path=tmp_path / "missing.json", wr_path=wr)
    bad = model(tmp_path, "bad.json", "WRONG_VERSION", [2022])
    with pytest.raises(RuntimeError, match="version mismatch"):
        gate(te_path=te, wr_path=bad)
    bad.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid frozen"):
        gate(te_path=te, wr_path=bad)


def test_invalid_training_seasons_and_target_seasons_rejected(tmp_path):
    te, wr = pair(tmp_path)
    with pytest.raises(ValueError, match="invalid target"):
        gate(te_path=te, wr_path=wr, target_seasons=(2023, 2023))
    with pytest.raises(ValueError, match="invalid target"):
        gate(te_path=te, wr_path=wr, target_seasons=())
    bad = model(tmp_path, "bad.json", "TE_R5P_PRODUCTION_MODEL_V1", [2023, 2023])
    with pytest.raises(RuntimeError, match="invalid model training"):
        read_model(bad, "TE_R5P_PRODUCTION_MODEL_V1")
