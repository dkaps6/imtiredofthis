import numpy as np
import pandas as pd

import scripts.research.lock_player_target_depth_distribution_shadow_v1 as m


def test_anchor_scale_is_one():
    assert abs(m.depth_scale("WR", m.DEPTH_ANCHOR["WR"]) - 1.0) < 1e-15
    assert abs(m.depth_scale("TE", m.DEPTH_ANCHOR["TE"]) - 1.0) < 1e-15


def test_scale_tracks_depth_dispersion_monotonically():
    low = m.depth_scale("WR", m.DEPTH_ANCHOR["WR"] * 0.64)
    high = m.depth_scale("WR", m.DEPTH_ANCHOR["WR"] * 1.44)
    assert abs(low - 0.8) < 1e-12
    assert abs(high - 1.2) < 1e-12
    assert low < 1.0 < high


def test_unavailable_or_non_wr_te_is_no_change():
    assert m.depth_scale("WR", np.nan) == 1.0
    assert m.depth_scale("TE", None) == 1.0
    assert m.depth_scale("RB", 10.0) == 1.0


def test_mean_neutral_shadow_preserves_exact_mean():
    draws = np.array([0.0, 5.0, 12.0, 20.0, 45.0, 80.0], dtype=float)
    baseline, candidate = m.mean_neutral_distribution_shadow(
        draws, exact_mean=31.25, scale=1.25
    )
    assert np.isfinite(baseline).all()
    assert np.isfinite(candidate).all()
    assert (baseline >= 0).all()
    assert (candidate >= 0).all()
    assert abs(float(baseline.mean()) - 31.25) <= 1e-10
    assert abs(float(candidate.mean()) - 31.25) <= 1e-10
    assert float(candidate.std(ddof=0)) > float(baseline.std(ddof=0))


def test_scale_one_is_exact_noop_after_baseline_alignment():
    draws = np.array([1.0, 3.0, 8.0, 21.0], dtype=float)
    baseline, candidate = m.mean_neutral_distribution_shadow(
        draws, exact_mean=12.0, scale=1.0
    )
    assert np.array_equal(baseline, candidate)
    assert abs(float(candidate.mean()) - 12.0) <= 1e-10


def test_contraction_reduces_width_with_mean_invariant():
    draws = np.array([0.0, 4.0, 10.0, 18.0, 38.0, 70.0], dtype=float)
    baseline, candidate = m.mean_neutral_distribution_shadow(
        draws, exact_mean=25.0, scale=0.8
    )
    assert abs(float(candidate.mean()) - 25.0) <= 1e-10
    assert float(candidate.std(ddof=0)) < float(baseline.std(ddof=0))


def test_feature_for_is_strictly_prior_and_requires_support():
    rows = []
    for week in range(1, 5):
        for j, air in enumerate([4.0 + week, 8.0 + week, 12.0 + week]):
            rows.append({
                "season": 2026,
                "week": week,
                "game_id": f"g{week}",
                "receiver_id": "p1",
                "air_yards": air,
            })
    # A target-game row must never be consumed.
    rows.append({
        "season": 2026,
        "week": 5,
        "game_id": "g5",
        "receiver_id": "p1",
        "air_yards": 999.0,
    })
    events = pd.DataFrame(rows)
    idx = m._event_index(events)
    f = m.feature_for(idx, "p1")
    assert f is not None
    assert f["prior_receiver_games"] == 4
    assert f["prior_finite_air_targets"] == 12
    assert f["feature_max_week"] == 4
    expected = np.std(
        [5.0, 9.0, 13.0, 6.0, 10.0, 14.0, 7.0, 11.0, 15.0, 8.0, 12.0, 16.0],
        ddof=0,
    )
    assert abs(f["prior8_target_depth_sd"] - expected) < 1e-12
