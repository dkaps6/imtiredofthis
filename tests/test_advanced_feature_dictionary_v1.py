from scripts.data_frontier.validate_advanced_feature_dictionary_v1 import (
    load_contract,
    validate_contract,
)


def test_advanced_feature_dictionary_v1_contract_passes():
    result = validate_contract(load_contract())
    assert result["disposition"] == "NFL_ADVANCED_FEATURE_DICTIONARY_V1_VALIDATED"
    assert result["field_count"] == 47
    assert result["source_hashes_verified"] == 3
    assert result["predictive_experiments_authorized"] is False
    assert result["production_changes_authorized"] is False


def test_advanced_feature_dictionary_v1_has_all_temporal_classes():
    result = validate_contract(load_contract())
    counts = result["temporal_class_counts"]
    assert counts["PREGAME_HISTORICAL_DERIVABLE"] > 0
    assert counts["TARGET_GAME_POST_KICKOFF"] > 0
    assert counts["RETROSPECTIVE_VALIDATION_ONLY"] > 0


def test_advanced_feature_dictionary_v1_sources_are_all_represented():
    result = validate_contract(load_contract())
    counts = result["source_field_counts"]
    assert counts["BDB2021_ROUTE_GEOMETRY"] > 0
    assert counts["BDB2023_PROTECTION_GEOMETRY"] > 0
    assert counts["BDB2026_THROW_WINDOW"] > 0
