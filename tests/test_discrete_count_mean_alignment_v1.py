import numpy as np

from scripts.modeling.discrete_count_alignment_v1 import (
    VERSION,
    align_outcomes,
)


def test_receptions_alignment_is_integer_and_near_target():
    raw = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    target = 2.91
    out, meta = align_outcomes(
        raw,
        market="receptions",
        mc_proj=float(raw.mean()),
        target_mean=target,
    )
    assert meta["discrete_count_alignment_applied"] == 1
    assert meta["discrete_count_alignment_version"] == VERSION
    assert np.array_equal(out, np.rint(out))
    assert np.min(out) >= 0
    assert abs(float(out.mean()) - target) <= 0.5 / len(out) + 1e-12


def test_rush_att_alignment_is_deterministic():
    raw = np.array([1, 2, 2, 3, 4, 6, 1, 2], dtype=float)
    kwargs = dict(
        market="rush_att",
        mc_proj=float(raw.mean()),
        target_mean=3.125,
    )
    a, ma = align_outcomes(raw, **kwargs)
    b, mb = align_outcomes(raw, **kwargs)
    assert np.array_equal(a, b)
    assert ma == mb


def test_zero_mc_is_exact_noop_even_with_nonzero_ensemble_target():
    raw = np.zeros(2000, dtype=float)
    out, meta = align_outcomes(
        raw,
        market="rush_att",
        mc_proj=0.0,
        target_mean=2.5,
    )
    assert np.array_equal(out, raw)
    assert meta["discrete_count_alignment_applied"] == 0
    assert meta["discrete_count_alignment_version"] == ""


def test_noncount_market_is_exact_legacy_continuous_alignment():
    raw = np.array([0.0, 5.0, 10.0, 20.0], dtype=float)
    mc = float(raw.mean())
    target = 13.25
    expected = raw * (target / mc)
    out, meta = align_outcomes(
        raw,
        market="rush_yards",
        mc_proj=mc,
        target_mean=target,
    )
    assert np.array_equal(out, expected)
    assert meta["discrete_count_alignment_applied"] == 0


def test_noncount_zero_mc_is_exact_legacy_noop():
    raw = np.zeros(8, dtype=float)
    out, meta = align_outcomes(
        raw,
        market="rec_yards",
        mc_proj=0.0,
        target_mean=17.5,
    )
    assert np.array_equal(out, raw)
    assert meta["discrete_count_alignment_applied"] == 0
