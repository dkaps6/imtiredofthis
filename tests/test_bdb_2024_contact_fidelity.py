from __future__ import annotations

import pandas as pd

from scripts.data_frontier.bdb_2024_contact_fidelity import build_fidelity_report


def test_fidelity_report_separates_abstention_overlap_and_geometry():
    enriched = pd.DataFrame([
        {"contactGeometryDefendersResolved": 1, "first_contact_primary_tackle_overlap": 1, "first_contact_assist_overlap": 0, "first_contact_missed_tackle_overlap": 0, "first_contact_any_source_label_overlap": 1, "firstContactCarrierSidelineDistanceYards": 8.0, "firstContactMaxClosingSpeedProxyYdsPerSec": 4.0, "firstContactMeanClosingSpeedProxyYdsPerSec": 4.0, "firstContactMinPursuitAngleErrorDeg": 12.0, "firstContactMeanPursuitAngleErrorDeg": 12.0},
        {"contactGeometryDefendersResolved": 0, "first_contact_primary_tackle_overlap": 0, "first_contact_assist_overlap": 1, "first_contact_missed_tackle_overlap": 0, "first_contact_any_source_label_overlap": 1, "firstContactCarrierSidelineDistanceYards": 3.0},
    ])
    defenders = pd.DataFrame([{"closingSpeedProxyYdsPerSec": 4.0, "pursuitAngleErrorDeg": 12.0}])
    dispositions = pd.DataFrame([
        {"benchmark_disposition": "SCOREABLE", "benchmark_reason": "ok"},
        {"benchmark_disposition": "SCOREABLE", "benchmark_reason": "ok"},
        {"benchmark_disposition": "ABSTAIN", "benchmark_reason": "missing_carrier_frame"},
    ])
    qa = {"source_window_diagnostics": {"tracking_plays": 3, "event_column_present": True}}
    report = build_fidelity_report(enriched, defenders, dispositions, qa)
    assert report["attempted_plays"] == 3
    assert report["scoreable_plays"] == 2
    assert report["geometry_resolution"]["plays_with_one_or_more_resolved_contact_defenders"] == 1
    assert report["source_label_overlap_rates"]["first_contact_primary_tackle_overlap"] == 0.5
    assert report["source_label_overlap_rates"]["first_contact_assist_overlap"] == 0.5
    assert report["abstention_reason_counts"] == {"missing_carrier_frame": 1}
    assert report["contact_detector_changed"] is False


def test_fidelity_report_preserves_missing_geometry_as_missing():
    enriched = pd.DataFrame([{"contactGeometryDefendersResolved": 0, "first_contact_any_source_label_overlap": 0}])
    dispositions = pd.DataFrame([{"benchmark_disposition": "SCOREABLE"}])
    report = build_fidelity_report(enriched, pd.DataFrame(), dispositions, {})
    assert report["geometry_resolution"]["rate_over_enriched_plays"] == 0.0
    assert report["source_label_overlap_rates"]["first_contact_any_source_label_overlap"] == 0.0
    assert report["geometry_distribution_sanity"] == {}
