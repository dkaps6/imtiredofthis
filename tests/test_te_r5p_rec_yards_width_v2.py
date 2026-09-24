"""Focused tests for TE-R5P receiving-yard width V2."""
from __future__ import annotations

import numpy as np

from scripts.research.te_r5p_rec_yards_width_v2 import empirical_crps, _widen


def test_empirical_crps_zero_for_degenerate_exact_forecast():
    draws = np.array([5.0, 5.0, 5.0, 5.0])
    assert empirical_crps(draws, 5.0) == 0.0


def test_empirical_crps_matches_two_point_closed_form():
    # F={0,2}, y=1: E|X-y|=1, 0.5 E|X-X'|=0.5 => CRPS=0.5.
    draws = np.array([0.0, 2.0])
    assert abs(empirical_crps(draws, 1.0) - 0.5) < 1e-12


def test_widen_preserves_mean_and_multiplies_sample_sd():
    draws = np.array([1.0, 2.0, 3.0, 4.0])
    mean = float(draws.mean())
    k = 1.75
    widened = _widen(draws, mean, k)
    assert abs(float(widened.mean()) - mean) < 1e-12
    assert abs(float(widened.std(ddof=1)) / float(draws.std(ddof=1)) - k) < 1e-12
