"""Characterization tests for the ensemble's inability to correct a level drift.

docs/research/PRISM_RB_ENSEMBLE_REWEIGHT_RESULTS_V1.md

These tests DOCUMENT current behaviour rather than assert it is desirable. The
scale-discard below is a real limitation: NNLS asked for a 1.52x scale-up on mc_proj for
rush_yards and the normalizer divided it away. If someone later changes this on purpose,
these tests should fail and be updated together with the doc -- that is the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.ensemble_v2 import (
    COMPONENTS,
    _normalise_nonnegative,
    fit_market_weights,
)


def test_normalise_discards_scale_information():
    """A coefficient vector asking for a 1.5x scale-up is normalized back to sum 1."""
    raw = np.array([1.5165, 0.0, 0.0])
    w = _normalise_nonnegative(raw)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0])
    # the 52% level correction present in `raw` is absent from `w`
    assert raw.sum() > 1.5


def test_weights_always_form_a_convex_combination():
    for raw in ([3.0, 1.0, 0.0], [0.2, 0.2, 0.2], [10.0, 0.0, 5.0], [-1.0, 2.0, 0.0]):
        w = _normalise_nonnegative(np.array(raw, dtype=float))
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_degenerate_coefficients_fall_back_to_mc_only():
    np.testing.assert_allclose(_normalise_nonnegative(np.array([0.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(_normalise_nonnegative(np.array([np.nan, 0.0, 0.0])), [1.0, 0.0, 0.0])


def test_blend_bias_is_bounded_by_its_components():
    """Convex combination => the blend can never be less biased than its best component.

    This is why reweighting cannot fix a level drift that affects every component, which
    is the structural reason fix B closed negative.
    """
    rng = np.random.default_rng(7)
    actual = rng.normal(50, 20, 500)
    comps = np.column_stack([actual - 16.0, actual - 10.0, actual - 4.0])  # all biased low
    for _ in range(200):
        w = rng.dirichlet([1, 1, 1])
        bias = float(np.mean(comps @ w - actual))
        assert bias <= -4.0 + 1e-9   # never better than the least-biased component


def test_squared_error_fit_prefers_the_biased_but_sharper_component():
    """The mechanism behind C1 landing on 100% MC.

    A component with a large constant bias but low noise wins on squared error against an
    unbiased but noisier one, so NNLS keeps selecting it.
    """
    rng = np.random.default_rng(11)
    n = 2000
    actual = rng.normal(50, 20, n)
    biased_sharp = actual - 16.0 + rng.normal(0, 2, n)
    unbiased_noisy = actual + rng.normal(0, 25, n)
    cal = pd.DataFrame({"market": "rush_yards", "actual": actual,
                        "mc_proj": biased_sharp, "ml_proj": unbiased_noisy,
                        "state_proj": actual + rng.normal(0, 40, n)})
    w = fit_market_weights(cal).iloc[0]
    assert w["mc_weight"] > w["ml_weight"]
    assert w["mc_weight"] > 0.8


def test_components_tuple_is_the_expected_three():
    assert COMPONENTS == ("mc_proj", "ml_proj", "state_proj")
