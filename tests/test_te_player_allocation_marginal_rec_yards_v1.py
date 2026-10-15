import numpy as np
import pandas as pd

from scripts.research.evaluate_te_player_allocation_marginal_rec_yards_v1 import (
    DRAW_COUNT,
    _empirical_crps,
    _overlay,
    _stable_seed,
    _standard_noise,
)


def test_standard_noise_is_exactly_centered_and_scaled():
    z = _standard_noise(DRAW_COUNT, 12345)
    assert len(z) == DRAW_COUNT
    assert abs(float(z.mean())) <= 1e-12
    assert abs(float(z.std(ddof=0)) - 1.0) <= 1e-12


def test_overlay_preserves_nonnegative_mean():
    base = np.linspace(0.0, 80.0, DRAW_COUNT)
    z = _standard_noise(DRAW_COUNT, 222)
    out = _overlay(base, 18.0, z)
    assert np.isfinite(out).all()
    assert (out >= 0).all()
    assert abs(float(out.mean()) - float(base.mean())) <= 1e-10


def test_overlay_zero_sigma_is_identity():
    base = np.linspace(0.0, 50.0, DRAW_COUNT)
    z = _standard_noise(DRAW_COUNT, 333)
    out = _overlay(base, 0.0, z)
    assert np.array_equal(out, base)


def test_empirical_crps_degenerate_truth_zero():
    x = np.zeros(DRAW_COUNT)
    assert _empirical_crps(x, 0.0) == 0.0


def test_stable_seed_repeatable():
    r = pd.Series({
        "season": 2025,
        "week": 8,
        "team": "KC",
        "player_clean_key": "testplayer",
    })
    assert _stable_seed(r) == _stable_seed(r)
