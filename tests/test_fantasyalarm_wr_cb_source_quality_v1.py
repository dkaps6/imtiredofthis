"""Fail-closed provider bridge isolation for public WR/CB source-quality audits."""

import pandas as pd

from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _provider_bridge


def test_provider_bridge_ignores_after_kickoff_and_schedule_mismatched_anchors():
    rows = pd.DataFrame(
        [
            # The same provider ID must not become ambiguous because of a quarantined row.
            {"source_id": "A", "gsis_id": "GS1", "schedule_match": True,
             "publication_timing_status": "PRE_KICKOFF"},
            {"source_id": "A", "gsis_id": "GS2", "schedule_match": True,
             "publication_timing_status": "AFTER_KICKOFF"},
            # An after-kickoff-only provider ID cannot be used as a bridge.
            {"source_id": "B", "gsis_id": "GS3", "schedule_match": True,
             "publication_timing_status": "AFTER_KICKOFF"},
            # A schedule-mismatched provider ID cannot be used as a bridge.
            {"source_id": "C", "gsis_id": "GS4", "schedule_match": False,
             "publication_timing_status": "PRE_KICKOFF"},
            # Real pre-kickoff collisions must remain quarantined.
            {"source_id": "D", "gsis_id": "GS5", "schedule_match": True,
             "publication_timing_status": "PRE_KICKOFF"},
            {"source_id": "D", "gsis_id": "GS6", "schedule_match": True,
             "publication_timing_status": "PRE_KICKOFF"},
        ]
    )
    mapping, collisions = _provider_bridge(rows, "source_id", "gsis_id")
    assert mapping == {"A": "GS1"}
    assert collisions == {"D"}
