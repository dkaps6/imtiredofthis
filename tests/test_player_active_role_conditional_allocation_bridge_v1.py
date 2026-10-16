import numpy as np

from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    discrete_crps,
    enumerate_active_states,
    mixture_binom_cdf,
)


def test_active_state_weights_and_state_mass_conserve():
    w, r, r0 = enumerate_active_states(
        np.array([0.4, 0.3]),
        np.array([0.8, 0.7]),
    )
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert np.allclose(r.sum(axis=1) + r0, 1.0, atol=1e-12)
    assert np.all(r >= 0)
    assert np.all(r0 >= 0)


def test_all_players_certain_reproduces_canonical_entitlement():
    q = np.array([0.4, 0.3])
    # Probabilities must remain inside the open interval, so use the closest
    # deterministic scientific limit rather than exactly one.
    p = np.array([1.0 - 1e-12, 1.0 - 1e-12])
    w, r, r0 = enumerate_active_states(q, p)
    expected = w @ r
    expected0 = w @ r0
    assert np.allclose(expected, q, atol=2e-12)
    assert np.isclose(expected0, 0.3, atol=2e-12)


def test_single_player_activity_lowers_expected_named_share_without_new_mean_fit():
    q = np.array([0.6])
    p = np.array([0.8])
    w, r, r0 = enumerate_active_states(q, p)
    expected = float(w @ r[:, 0])
    expected0 = float(w @ r0)
    # If active, canonical 0.6/0.4 is preserved. If inactive, complement gets 1.
    assert np.isclose(expected, 0.48)
    assert np.isclose(expected0, 0.52)


def test_degenerate_all_off_fallback_goes_to_anonymous_complement():
    q = np.array([1.0])
    p = np.array([0.5])
    w, r, r0 = enumerate_active_states(q, p)
    # One state is inactive -> complement 1; one state is active -> player 1.
    assert np.isclose(w.sum(), 1.0)
    pairs = sorted((round(float(ri[0]), 6), round(float(ci), 6), round(float(wi), 6))
                   for wi, ri, ci in zip(w, r, r0))
    assert pairs == [(0.0, 1.0, 0.5), (1.0, 0.0, 0.5)]


def test_mixture_cdf_and_crps_are_finite():
    N = 10
    weights = np.array([0.25, 0.75])
    probs = np.array([0.0, 0.6])
    cdf = mixture_binom_cdf(N, probs, weights)
    assert np.isfinite(cdf).all()
    assert np.all(np.diff(cdf) >= -1e-12)
    assert np.isclose(cdf[-1], 1.0)
    score = discrete_crps(cdf, 5)
    assert np.isfinite(score)
    assert score >= 0
