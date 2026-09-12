import numpy as np

from scripts.research.grade_empirical_fair_prob_v1 import (
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
