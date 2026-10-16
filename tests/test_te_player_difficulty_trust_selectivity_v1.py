import numpy as np
import pandas as pd

from scripts.research.evaluate_te_player_difficulty_trust_selectivity_v1 import (
    _corr,
    _rank_pct,
)


def test_rank_pct_average_ties():
    s = pd.Series([1.0, 2.0, 2.0, 4.0])
    r = _rank_pct(s)
    assert np.allclose(r.to_numpy(float), [0.25, 0.625, 0.625, 1.0])


def test_corr_positive_monotonic():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)
    assert abs(_corr(x, y) - 1.0) <= 1e-12


def test_corr_negative_monotonic():
    x = np.arange(10, dtype=float)
    y = -np.arange(10, dtype=float)
    assert abs(_corr(x, y) + 1.0) <= 1e-12


def test_corr_constant_is_nan():
    assert np.isnan(_corr(np.ones(10), np.arange(10, dtype=float)))
