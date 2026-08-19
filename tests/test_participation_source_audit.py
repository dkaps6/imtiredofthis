import pandas as pd

from scripts.backtest.audit_participation_source import audit_frames


def test_participation_audit_detects_all_defensive_fields_and_joinability():
    participation = pd.DataFrame([
        {"nflverse_game_id": "2025_01_MIA_BUF", "play_id": 1, "defenders_in_box": 6, "defense_man_zone_type": "ZONE", "defense_coverage_type": "COVER_3"},
        {"nflverse_game_id": "2025_01_MIA_BUF", "play_id": 2, "defenders_in_box": 8, "defense_man_zone_type": "MAN", "defense_coverage_type": "COVER_1"},
    ])
    pbp = pd.DataFrame([
        {"nflverse_game_id": "2025_01_MIA_BUF", "play_id": 1},
        {"nflverse_game_id": "2025_01_MIA_BUF", "play_id": 2},
    ])
    summary, by_week = audit_frames(participation, pbp)
    metrics = summary.set_index("metric")
    assert metrics.loc["field_defenders_in_box", "value"] == 1.0
    assert metrics.loc["field_defense_man_zone_type", "value"] == 1.0
    assert metrics.loc["field_defense_coverage_type", "value"] == 1.0
    assert metrics.loc["pbp_join_match_rate", "value"] == 1.0
    assert metrics.loc["source_recommendation", "value"] == "GO"
    assert by_week.iloc[0]["season"] == 2025
    assert by_week.iloc[0]["week"] == 1


def test_participation_audit_never_claims_missing_field_is_available():
    participation = pd.DataFrame([
        {"nflverse_game_id": "2025_02_BUF_MIA", "play_id": 1, "defenders_in_box": 7},
    ])
    pbp = pd.DataFrame([
        {"nflverse_game_id": "2025_02_BUF_MIA", "play_id": 1},
    ])
    summary, _ = audit_frames(participation, pbp)
    metrics = summary.set_index("metric")
    assert metrics.loc["field_defense_man_zone_type", "status"] == "missing_column"
    assert metrics.loc["field_defense_coverage_type", "status"] == "missing_column"
    assert metrics.loc["source_recommendation", "value"] == "NO_GO"
