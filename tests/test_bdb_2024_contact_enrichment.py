import pandas as pd

from scripts.data_frontier.bdb_2024_contact_enrichment import enrich_contact_features


def test_enrichment_adds_geometry_and_separate_source_overlap_without_changing_detector():
    features = pd.DataFrame([{
        "gameId": 1, "playId": 2, "ballCarrierId": 10,
        "firstContactFrameId": 5, "firstContactDefenderIds": "20",
        "sourcePrimaryTacklerIds": "20", "sourceAssistIds": "21",
        "sourceMissedTacklerIds": "22", "featureVersion": "BDB_2024_CONTACT_GEOMETRY_V1",
    }])
    tracking = pd.DataFrame([
        {"gameId": 1, "playId": 2, "frameId": 5, "nflId": 10, "x": 50.0, "y": 10.0, "s": 2.0, "dir": 90.0},
        {"gameId": 1, "playId": 2, "frameId": 5, "nflId": 20, "x": 51.0, "y": 10.0, "s": 4.0, "dir": 270.0},
    ])
    enriched, defenders = enrich_contact_features(features, tracking)
    row = enriched.iloc[0]
    assert row["first_contact_primary_tackle_overlap"] == 1
    assert row["first_contact_assist_overlap"] == 0
    assert row["first_contact_missed_tackle_overlap"] == 0
    assert row["firstContactCarrierSidelineDistanceYards"] == 10.0
    assert row["contactDetectorChanged"] == False
    assert row["featureVersion"] == "BDB_2024_CONTACT_GEOMETRY_V1"
    assert len(defenders) == 1
    assert defenders.iloc[0]["isSourcePrimaryTackler"] == 1


def test_enrichment_abstains_on_missing_contact_frame_geometry():
    features = pd.DataFrame([{
        "gameId": 1, "playId": 3, "ballCarrierId": 10,
        "firstContactFrameId": 8, "firstContactDefenderIds": "20",
        "sourcePrimaryTacklerIds": "", "sourceAssistIds": "",
        "sourceMissedTacklerIds": "", "featureVersion": "BDB_2024_CONTACT_GEOMETRY_V1",
    }])
    tracking = pd.DataFrame([{"gameId": 1, "playId": 3, "frameId": 7, "nflId": 10, "x": 1.0, "y": 1.0, "s": 1.0, "dir": 0.0}])
    enriched, defenders = enrich_contact_features(features, tracking)
    assert pd.isna(enriched.iloc[0]["firstContactMaxClosingSpeedProxyYdsPerSec"])
    assert enriched.iloc[0]["contactGeometryDefendersResolved"] == 0
    assert defenders.empty
