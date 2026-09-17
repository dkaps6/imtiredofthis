import pandas as pd
import pytest

from scripts.data_frontier.advanced_feature_materializer_common_v1 import (
    ALGORITHM_VERSION,
    assert_strict_prior,
    load_dictionary,
    source_contract,
)


def test_materializer_contract_covers_all_three_validated_labs():
    d = load_dictionary()
    expected = {
        "BDB2021_ROUTE_GEOMETRY": 12,
        "BDB2023_PROTECTION_GEOMETRY": 13,
        "BDB2026_THROW_WINDOW": 22,
    }
    for source_key, count in expected.items():
        source, fields = source_contract(source_key, d)
        assert len(fields) == count
        assert len(source["source_hash_sha256"]) == 64
    assert ALGORITHM_VERSION == "NFL_ADVANCED_FEATURE_MATERIALIZER_V1"


def test_strict_prior_accepts_only_earlier_weeks():
    df = pd.DataFrame(
        {
            "target_week": [2, 5, 9],
            "history_max_source_week": [1, 4, 8],
        }
    )
    result = assert_strict_prior(df)
    assert result == {"rows_checked": 3, "violations": 0}


def test_strict_prior_fails_closed_on_same_week_or_future_history():
    df = pd.DataFrame(
        {
            "target_week": [4, 8],
            "history_max_source_week": [4, 9],
        }
    )
    with pytest.raises(ValueError, match="temporal leakage"):
        assert_strict_prior(df)


def test_no_contracted_field_authorizes_predictive_or_production_use():
    d = load_dictionary()
    for field in d["fields"]:
        assert field["production_status"] == "RESEARCH_ONLY_NOT_PROMOTED"
        assert field["predictive_experiment_status"] == "NOT_AUTHORIZED_BY_THIS_CONTRACT"
        temporal = field["temporal_availability"]
        if temporal["classification"] != "PREGAME_HISTORICAL_DERIVABLE":
            assert temporal["target_game_pregame_allowed"] is False
