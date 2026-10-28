"""Contract tests for the RB carry-concentration transform.

docs/research/PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import simulation_v2 as sim


def _team(positions):
    return pd.DataFrame({"position": positions})


def test_inert_by_default():
    """Not promoted => identity. This is the production-safety contract."""
    assert sim.RB_CARRY_CONCENTRATION_PROMOTED is False
    assert sim.rb_carry_concentration_gamma() == 1.0
    shares = np.array([0.50, 0.25, 0.15, 0.10])
    out = sim._sharpen_rb_rush_shares(_team(["RB", "RB", "RB", "QB"]), shares)
    np.testing.assert_allclose(out, shares)


def test_candidate_gamma_not_applied_while_unpromoted():
    """The candidate constant alone must never move a projection."""
    assert sim.RB_CARRY_CONCENTRATION_GAMMA_CANDIDATE > 1.0
    assert sim.rb_carry_concentration_gamma() == sim.RB_CARRY_CONCENTRATION_GAMMA_NEUTRAL


@pytest.fixture
def promoted(monkeypatch):
    monkeypatch.setattr(sim, "RB_CARRY_CONCENTRATION_PROMOTED", True)
    return sim.rb_carry_concentration_gamma()


def test_promoted_gamma_is_the_fitted_value(promoted):
    assert promoted == pytest.approx(1.75)


def test_mass_is_preserved_exactly(promoted):
    """Hard requirement: team RB carry volume must not move."""
    shares = np.array([0.40, 0.22, 0.11, 0.30, 0.05])
    team = _team(["RB", "RB", "HB", "QB", "WR"])
    out = sim._sharpen_rb_rush_shares(team, shares)
    rb = [0, 1, 2]
    assert out[rb].sum() == pytest.approx(shares[rb].sum())


def test_non_rb_shares_are_untouched(promoted):
    shares = np.array([0.40, 0.22, 0.30, 0.05])
    team = _team(["RB", "RB", "QB", "WR"])
    out = sim._sharpen_rb_rush_shares(team, shares)
    assert out[2] == pytest.approx(0.30)
    assert out[3] == pytest.approx(0.05)


def test_concentration_increases_and_ordering_is_preserved(promoted):
    shares = np.array([0.40, 0.25, 0.20, 0.15])
    team = _team(["RB", "RB", "RB", "RB"])
    out = sim._sharpen_rb_rush_shares(team, shares)
    assert out[0] > shares[0]                      # top back gains
    assert out[-1] < shares[-1]                    # deep back loses
    assert list(np.argsort(-out)) == list(np.argsort(-shares))  # monotone


def test_single_back_is_a_noop(promoted):
    shares = np.array([0.55, 0.30])
    out = sim._sharpen_rb_rush_shares(_team(["RB", "QB"]), shares)
    np.testing.assert_allclose(out, shares)


def test_zero_rb_mass_is_a_noop(promoted):
    shares = np.array([0.0, 0.0, 0.33])
    out = sim._sharpen_rb_rush_shares(_team(["RB", "RB", "QB"]), shares)
    np.testing.assert_allclose(out, shares)


def test_rb_label_variants_are_recognised():
    for label in ("RB", "rb", "HB", "TB", "FB", "RB1", "RB2"):
        assert sim._is_rb_position(label), label
    for label in ("QB", "WR", "TE", "", "SWR"):
        assert not sim._is_rb_position(label), label


def test_nan_and_inf_are_sanitised(promoted):
    shares = np.array([np.nan, np.inf, 0.30, -0.10])
    out = sim._sharpen_rb_rush_shares(_team(["RB", "RB", "RB", "RB"]), shares)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_gamma_is_clamped_to_a_sane_band(monkeypatch):
    monkeypatch.setattr(sim, "RB_CARRY_CONCENTRATION_PROMOTED", True)
    monkeypatch.setattr(sim, "RB_CARRY_CONCENTRATION_GAMMA_CANDIDATE", 99.0)
    assert sim.rb_carry_concentration_gamma() == 4.0
    monkeypatch.setattr(sim, "RB_CARRY_CONCENTRATION_GAMMA_CANDIDATE", 0.2)
    assert sim.rb_carry_concentration_gamma() == 1.0


def test_gamma_moves_concentration_by_a_material_amount(promoted):
    """gamma=1.75 must lift top-1 share substantially, not trivially.

    The population claim (production 0.5173 -> 0.6283 against the 2013-2025 target
    0.6277) is a mean over 25 real team vectors and is verified by
    scripts/research/measure_production_rb_concentration_v1.py -- a single synthetic
    vector cannot reproduce it, because the transform's effect depends on the whole
    vector's shape. What is asserted here is the per-vector property: a material,
    correctly-signed move in concentration.
    """
    shares = np.array([0.5334, 0.2733, 0.1933])
    out = sim._sharpen_rb_rush_shares(_team(["RB", "RB", "RB"]), shares)
    top1_before = shares[0] / shares.sum()
    top1_after = out[0] / out.sum()
    assert top1_after - top1_before > 0.08
    assert top1_after < 1.0


def test_wr_target_path_is_unaffected():
    """The RB change must not perturb the promoted WR hierarchy sharpening."""
    team = _team(["WR", "WR", "WR"])
    shares = np.array([0.30, 0.20, 0.10])
    out = sim._sharpen_wr_target_shares(team, shares)
    assert out.sum() == pytest.approx(shares.sum())
