"""Tests for heterogeneous-price significance in graded track-record slicing."""
from __future__ import annotations

from scripts.research.slice_graded_track_record_v1 import poisson_binomial_tail


def test_poisson_binomial_tail_matches_manual_heterogeneous_example():
    # P(X >= 2) for p=[.1,.3,.8]:
    # exactly two = .006 + .056 + .216; all three = .024.
    assert abs(poisson_binomial_tail(2, [0.1, 0.3, 0.8]) - 0.302) < 1e-12


def test_poisson_binomial_tail_handles_boundaries():
    assert poisson_binomial_tail(0, [0.2, 0.7]) == 1.0
    assert poisson_binomial_tail(3, [0.2, 0.7]) == 0.0
