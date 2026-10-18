import numpy as np

from scripts.research.evaluate_prism_rb_stage2_player_specific_mechanism_authority_bridge_v1 import (
    enumerate_states,
)


def test_deterministic_active_player_is_supported_and_conserves():
    q = np.array([0.6, 0.2])
    p = np.array([1.0, 0.5])
    w, r, r0 = enumerate_states(q, p)
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert np.allclose(r.sum(axis=1) + r0, 1.0, atol=1e-12)
    # Player 1 is deterministically active in every positive-weight state.
    assert np.all(r[:, 0] > 0)


def test_all_deterministic_reproduces_canonical_entitlement():
    q = np.array([0.5, 0.3])
    p = np.array([1.0, 1.0])
    w, r, r0 = enumerate_states(q, p)
    assert len(w) == 1
    assert np.isclose(w[0], 1.0)
    assert np.allclose(r[0], q, atol=1e-12)
    assert np.isclose(r0[0], 0.2, atol=1e-12)


def test_disabling_one_players_uncertainty_changes_room_coherently():
    q = np.array([0.6, 0.2])
    # Player 1 retains uncertainty.
    w0, r0, c0 = enumerate_states(q, np.array([0.5, 0.5]))
    # Player 1's own Stage-2 uncertainty is disabled.
    w1, r1, c1 = enumerate_states(q, np.array([1.0, 0.5]))
    e0 = w0 @ r0
    e1 = w1 @ r1
    assert np.isclose(w0.sum(), 1.0)
    assert np.isclose(w1.sum(), 1.0)
    assert np.isclose(float(e0.sum()) + float(w0 @ c0), 1.0)
    assert np.isclose(float(e1.sum()) + float(w1 @ c1), 1.0)
    # Deterministically active player 1 receives more expected room share.
    assert e1[0] > e0[0]


def test_zero_complement_all_off_fallback_still_conserves():
    q = np.array([0.7, 0.3])
    p = np.array([1.0, 0.0])
    w, r, r0 = enumerate_states(q, p)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(r.sum(axis=1) + r0, 1.0)
    assert np.isfinite(r).all()
    assert np.isfinite(r0).all()
