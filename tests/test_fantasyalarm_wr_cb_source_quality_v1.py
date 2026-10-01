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


def test_article_content_version_timing_distinguishes_publication_from_edits():
    from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _content_version_timing
    x = pd.DataFrame([
        {"published_at_utc": "2023-10-27T16:00:00Z", "kickoff_utc": "2023-10-29T17:00:00Z",
         "modified_at_utc": "2024-08-21T13:00:00Z", "modification_metadata_status": "UNAMBIGUOUS_MODIFICATION_METADATA"},
        {"published_at_utc": "2023-10-27T16:00:00Z", "kickoff_utc": "2023-10-29T17:00:00Z",
         "modified_at_utc": "2023-10-28T13:00:00Z", "modification_metadata_status": "UNAMBIGUOUS_MODIFICATION_METADATA"},
        {"published_at_utc": "2023-10-27T16:00:00Z", "kickoff_utc": "2023-10-29T17:00:00Z",
         "modified_at_utc": "", "modification_metadata_status": "MISSING_MODIFICATION_METADATA"},
        {"published_at_utc": "2023-10-27T16:00:00Z", "kickoff_utc": "2023-10-29T17:00:00Z",
         "modified_at_utc": "", "modification_metadata_status": "CONFLICTING_MODIFICATION_METADATA"},
    ])
    assert _content_version_timing(x).tolist() == [
        "MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED",
        "METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF",
        "UNVERIFIED_NO_MODIFICATION_METADATA",
        "UNVERIFIED_CONFLICTING_MODIFICATION_METADATA",
    ]


def test_provider_reuse_veto_catches_hidden_wrong_person_bridge():
    from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import _provider_reused_ids
    evidence = pd.DataFrame([
        # Pre-kickoff trusted Goodwin anchor; the reused ID's second person
        # is unresolved and would bypass the old anchored-GSIS collision count.
        {"source_id": "300936", "name_key": "marquisegoodwin", "gsis_id": "GOODWIN",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        {"source_id": "300936", "name_key": "allenrobinsonii", "gsis_id": "",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        # Known exact person aliases remain eligible for bridge.
        {"source_id": "304198", "name_key": "robbieanderson", "gsis_id": "ANDERSON",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        {"source_id": "304198", "name_key": "robbiechosen", "gsis_id": "",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        # Same source can also conflate two corners in a single table cell.
        {"source_id": "305132", "name_key": "marshonlattimore", "gsis_id": "LATTIMORE",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        {"source_id": "305132", "name_key": "marshonlattimore/bradleyroby", "gsis_id": "",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        # Unknown spelling drift is quarantined, not automatically assumed alias.
        {"source_id": "308712", "name_key": "nickwestbrookikhine", "gsis_id": "NICK",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
        {"source_id": "308712", "name_key": "nickwestbrookikhineikhine", "gsis_id": "",
         "schedule_match": True, "publication_timing_status": "PRE_KICKOFF"},
    ])
    reuse = _provider_reused_ids(evidence, "source_id", "name_key")
    assert reuse == {"300936", "305132", "308712"}
    old_map, old_collision = _provider_bridge(evidence, "source_id", "gsis_id")
    assert not old_collision  # demonstrates why the old census misses this
    assert old_map["300936"] == "GOODWIN"
    safe_map = {k: v for k, v in old_map.items() if k not in reuse}
    assert safe_map == {"304198": "ANDERSON"}
