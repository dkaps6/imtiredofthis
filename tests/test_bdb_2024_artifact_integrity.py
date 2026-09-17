import pandas as pd

from scripts.data_frontier.bdb_2024_artifact_integrity import validate_artifacts


def _qa():
    return {
        "feature_version": "BDB_2024_CONTACT_GEOMETRY_V1",
        "contact_detector_changed": False,
        "normalized_artifacts": {
            "tracking": {"rows": 10, "columns": ["gameId", "playId"]},
            "plays": {"rows": 1, "columns": ["gameId", "playId"]},
            "tackles": {"rows": 1, "columns": ["gameId", "playId", "nflId"]},
        },
    }


def test_integrity_passes_clean_artifacts():
    enriched = pd.DataFrame([{ "gameId": 1, "playId": 2, "firstContactCarrierSidelineDistanceYards": 10.0,
        "firstContactMinPursuitAngleErrorDeg": 15.0, "firstContactMeanPursuitAngleErrorDeg": 20.0 }])
    defenders = pd.DataFrame([{ "gameId": 1, "playId": 2, "defenderNflId": 99,
        "closingSpeedProxyYdsPerSec": 2.0, "pursuitAngleErrorDeg": 15.0 }])
    dispositions = pd.DataFrame([{ "gameId": 1, "playId": 2, "benchmark_disposition": "SCOREABLE" }])
    report = validate_artifacts(enriched, defenders, dispositions, _qa())
    assert report["passed"] is True
    assert report["failure_count"] == 0


def test_integrity_fails_duplicate_and_orphan_defender():
    enriched = pd.DataFrame([{ "gameId": 1, "playId": 2 }, { "gameId": 1, "playId": 2 }])
    defenders = pd.DataFrame([{ "gameId": 1, "playId": 999, "defenderNflId": 99 }])
    dispositions = pd.DataFrame([{ "gameId": 1, "playId": 2, "benchmark_disposition": "SCOREABLE" }])
    report = validate_artifacts(enriched, defenders, dispositions, _qa())
    checks = {x["check"] for x in report["failures"]}
    assert report["passed"] is False
    assert "enriched_key_unique" in checks
    assert "defender_play_referential_integrity" in checks


def test_integrity_fails_scoreable_parity_and_geometry_range():
    enriched = pd.DataFrame([{ "gameId": 1, "playId": 3, "firstContactCarrierSidelineDistanceYards": 40.0,
        "firstContactMinPursuitAngleErrorDeg": 181.0 }])
    defenders = pd.DataFrame()
    dispositions = pd.DataFrame([{ "gameId": 1, "playId": 2, "benchmark_disposition": "SCOREABLE" }])
    report = validate_artifacts(enriched, defenders, dispositions, _qa())
    checks = {x["check"] for x in report["failures"]}
    assert "scoreable_enriched_key_parity" in checks
    assert "range_firstContactCarrierSidelineDistanceYards" in checks
    assert "range_firstContactMinPursuitAngleErrorDeg" in checks


def test_integrity_fails_version_and_detector_guardrails():
    enriched = pd.DataFrame([{ "gameId": 1, "playId": 2 }])
    defenders = pd.DataFrame()
    dispositions = pd.DataFrame([{ "gameId": 1, "playId": 2, "benchmark_disposition": "SCOREABLE" }])
    qa = _qa()
    qa.pop("feature_version")
    qa["contact_detector_changed"] = True
    report = validate_artifacts(enriched, defenders, dispositions, qa)
    checks = {x["check"] for x in report["failures"]}
    assert "qa_feature_version_present" in checks
    assert "contact_detector_immutability" in checks
