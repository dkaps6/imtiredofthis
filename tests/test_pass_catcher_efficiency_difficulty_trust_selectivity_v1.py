import numpy as np
import pandas as pd

from scripts.research.evaluate_pass_catcher_efficiency_difficulty_trust_selectivity_v1 import (
    corr,
    partial_corr,
    rank_pct,
)


def test_rank_pct_average_ties():
    s = pd.Series([1.0, 2.0, 2.0, 4.0])
    got = rank_pct(s).to_numpy(float)
    want = np.array([0.25, 0.625, 0.625, 1.0])
    assert np.allclose(got, want)


def test_partial_corr_removes_shared_control():
    z = np.linspace(-2.0, 2.0, 500)
    x = z + 0.1 * np.sin(np.arange(500))
    y = z + 0.1 * np.cos(np.arange(500))
    assert corr(x, y) > 0.95
    assert abs(partial_corr(x, y, z)) < 0.1


def test_partial_corr_keeps_incremental_signal():
    rng = np.random.default_rng(20261015)
    z = rng.normal(size=2000)
    x = 0.7 * z + rng.normal(size=2000)
    y = 0.8 * x + 0.4 * z + rng.normal(size=2000)
    assert partial_corr(x, y, z) > 0.4
