import numpy as np
import pandas as pd
import pytest

from scripts.research.grade_empirical_fair_prob_v1 import (
    _assert_same_base_cohort,
    _canon_keys,
    empirical_over_probability,
    rescale_outcomes,
)


def test_rescale_outcomes_matches_production_mean_alignment():
    raw = np.array([0.0, 10.0, 20.0, 30.0])
    adjusted = rescale_outcomes(raw, 30.0)
    assert np.isclose(adjusted.mean(), 30.0)
    assert np.all(adjusted >= 0.0)


def test_rescale_outcomes_preserves_shape_by_single_multiplier():
    raw = np.array([2.0, 4.0, 8.0])
    adjusted = rescale_outcomes(raw, 14.0)
    ratios = adjusted / raw
    assert np.allclose(ratios, ratios[0])


def test_empirical_over_probability_uses_strict_over_semantics():
    outcomes = np.array([9.5, 10.0, 10.5, 11.0])
    assert empirical_over_probability(outcomes, 10.0) == 0.5


def test_same_row_contract_allows_translator_to_change_strong_coverage():
    comparison = pd.DataFrame(
        [
            {
                "market": "ALL_MARKETS",
                "tier": "ALL_NO_FILTER",
                "matched_rows_legacy": 100,
                "matched_rows_empirical": 100,
            },
            {
                "market": "ALL_MARKETS",
                "tier": "STRONG_ONLY_PLAY_TIER",
                "matched_rows_legacy": 90,
                "matched_rows_empirical": 35,
            },
        ]
    )
    # Tier membership is an experimental result, not a cohort-identity invariant.
    _assert_same_base_cohort(comparison)


def test_same_row_contract_rejects_unfiltered_cohort_mismatch():
    comparison = pd.DataFrame(
        [
            {
                "market": "ALL_MARKETS",
                "tier": "ALL_NO_FILTER",
                "matched_rows_legacy": 100,
                "matched_rows_empirical": 99,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="unfiltered cohort"):
        _assert_same_base_cohort(comparison)


def test_distribution_identity_requires_opponent():
    missing_opponent = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "team": "KC",
                "player_clean_key": "patrickmahomes",
                "market": "pass_yards",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="missing columns"):
        _canon_keys(missing_opponent)
