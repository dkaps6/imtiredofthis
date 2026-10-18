import numpy as np

from scripts.research.evaluate_prism_rb_te_stage1_player_specific_mechanism_authority_bridge_v1 import (
    personalize_probability,
)


def test_low_meta_support_preserves_universal_stage1_candidate():
    pp, fallback = personalize_probability(
        np.array([0.90]),
        np.array([0.60]),
        np.array([2]),
        np.array([-1.0]),
    )
    assert np.isclose(pp[0], 0.60)
    assert not fallback[0]


def test_supported_positive_utility_preserves_candidate():
    pp, fallback = personalize_probability(
        np.array([0.90]),
        np.array([0.60]),
        np.array([3]),
        np.array([0.001]),
    )
    assert np.isclose(pp[0], 0.60)
    assert not fallback[0]


def test_supported_zero_or_negative_utility_reverts_to_family_baseline():
    pp, fallback = personalize_probability(
        np.array([0.90, 0.80]),
        np.array([0.60, 0.55]),
        np.array([3, 10]),
        np.array([0.0, -0.1]),
    )
    assert np.allclose(pp, np.array([0.90, 0.80]))
    assert fallback.tolist() == [True, True]


def test_authority_rule_does_not_require_target_outcome():
    pb = np.array([0.75, 0.80, 0.85])
    pc = np.array([0.65, 0.70, 0.72])
    n = np.array([2, 3, 8])
    h = np.array([-0.5, 0.1, -0.01])
    pp, fallback = personalize_probability(pb, pc, n, h)
    assert np.allclose(pp, np.array([0.65, 0.70, 0.85]))
    assert fallback.tolist() == [False, False, True]
