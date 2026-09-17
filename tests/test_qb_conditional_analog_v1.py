"""Synthetic-fixture tests for scripts/research/qb_conditional_analog_v1.py.

No real 2024/2025 outcome data is read anywhere in this file. Every fixture
below is fabricated purely to exercise the leakage-safe mechanism described
in docs/research/QB_CONDITIONAL_ANALOG_V1_PLAN.md; it proves the code is
correct, not that the architecture is actionable on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import qb_conditional_analog_v1 as m


def _feature_row(**overrides):
    row = {c: 0.0 for c in m.FEATURE_COLUMNS}
    row.update(overrides)
    return row


def test_join_vegas_and_features_inner_join_on_keys():
    vegas_df = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "player_clean_key": "pmahomes", "actual_pass_yards": 291},
        {"season": 2024, "week": 2, "team": "KC", "player_clean_key": "pmahomes", "actual_pass_yards": 305},
        {"season": 2025, "week": 1, "team": "KC", "player_clean_key": "pmahomes", "actual_pass_yards": 260},
    ])
    features_df = pd.DataFrame([
        {**_feature_row(component_range=10.0), "season": 2024, "week": 1, "team": "KC", "player_clean_key": "pmahomes"},
        {**_feature_row(component_range=12.0), "season": 2025, "week": 1, "team": "KC", "player_clean_key": "pmahomes"},
        {**_feature_row(component_range=99.0), "season": 2024, "week": 9, "team": "KC", "player_clean_key": "pmahomes"},
    ])
    merged = m.join_vegas_and_features(vegas_df, features_df)
    assert len(merged) == 2
    assert set(merged["week"]) == {1}
    assert "component_range" in merged.columns


def test_join_raises_on_missing_feature_column():
    vegas_df = pd.DataFrame([{"season": 2024, "week": 1, "team": "KC", "player_clean_key": "x"}])
    features_df = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "KC", "player_clean_key": "x", "component_range": 1.0}
    ])
    with pytest.raises(RuntimeError, match="missing frozen feature columns"):
        m.join_vegas_and_features(vegas_df, features_df)


def test_exclude_missing_drops_only_null_rows_and_never_imputes():
    df = pd.DataFrame([
        _feature_row(component_range=1.0),
        _feature_row(component_range=np.nan),
        _feature_row(qb_prior_attempts=np.nan),
        _feature_row(component_range=2.0, qb_prior_attempts=3.0),
    ])
    clean, dropped = m.exclude_missing(df)
    assert dropped == 2
    assert len(clean) == 2
    assert clean.isna().sum().sum() == 0


def test_scaler_is_fit_on_reference_only_and_frozen_for_evaluation():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {c: rng.normal(loc=10.0, scale=2.0, size=200) for c in m.FEATURE_COLUMNS}
    )
    evaluation = pd.DataFrame(
        {c: rng.normal(loc=50.0, scale=2.0, size=5) for c in m.FEATURE_COLUMNS}
    )
    scaler = m.fit_frozen_scaler(reference)
    ref_std = m.standardize(reference, scaler)
    eval_std = m.standardize(evaluation, scaler)

    assert np.allclose(ref_std.mean(axis=0), 0.0, atol=0.2)
    assert np.allclose(ref_std.std(axis=0), 1.0, atol=0.2)
    # Evaluation pool is drawn from a shifted distribution; the frozen (reference-fit)
    # scaler must NOT re-center on it -- eval_std should be far from zero-mean.
    assert np.all(eval_std.mean(axis=0) > 5.0)


def test_pairwise_euclidean_matches_known_distances():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [0.0, 4.0]])
    d = m.pairwise_euclidean(a, b)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(4.0)
    assert d[1, 0] == pytest.approx(5.0)
    assert d[1, 1] == pytest.approx(3.0)


def test_k_nearest_returns_sorted_closest_indices():
    points = np.array([[0.0], [1.0], [5.0], [2.0]])
    query = np.array([[0.0]])
    dists = m.pairwise_euclidean(query, points)
    idx, d = m.k_nearest(dists, k=2)
    assert list(idx[0]) == [0, 1]
    assert list(d[0]) == pytest.approx([0.0, 1.0])


def test_leave_one_out_density_threshold_is_feature_space_only():
    rng = np.random.default_rng(1)
    tight_cluster = rng.normal(loc=0.0, scale=0.1, size=(60, 3))
    threshold = m.leave_one_out_density_threshold(tight_cluster, k=5, percentile=90.0)
    assert threshold > 0.0
    assert threshold < 2.0  # a tight cluster must yield a small density threshold


def test_density_gate_pass_flags_far_points_as_fail():
    kth_distance = np.array([0.5, 1.5, 10.0])
    passed = m.density_gate_pass(kth_distance, threshold=2.0)
    assert list(passed) == [True, True, False]


def test_classify_row_evidence_all_three_branches():
    density_pass = np.array([False, True, True])
    directionally_supported = np.array([True, True, False])
    labels = m.classify_row_evidence(density_pass, directionally_supported)
    assert labels == [
        m.EVIDENCE_NO_ANALOG_SUPPORT,
        m.EVIDENCE_SUPPORTED,
        m.EVIDENCE_DESCRIPTIVE_ONLY,
    ]


def test_analog_direction_consistency_majority_vote():
    reference_direction = np.array([1, 1, 0, 0, 1])
    neighbor_indices = np.array([
        [0, 1, 4],  # 1,1,1 -> majority True
        [2, 3, 4],  # 0,0,1 -> majority False
    ])
    out = m.analog_direction_consistency(reference_direction, neighbor_indices)
    assert list(out) == [True, False]


def test_score_architecture_gate_passes_when_all_conditions_hold():
    reference_bucket_supported = np.array([True] * 60 + [False] * 40)
    reference_realized_positive = np.array([True] * 45 + [False] * 15 + [False] * 40)
    evaluation_bucket_supported = np.array([True] * 50 + [False] * 50)
    evaluation_realized_roi = np.array([0.10] * 50 + [-0.05] * 50)
    result = m.score_architecture_gate(
        reference_bucket_supported=reference_bucket_supported,
        reference_realized_positive=reference_realized_positive,
        evaluation_bucket_supported=evaluation_bucket_supported,
        evaluation_realized_roi=evaluation_realized_roi,
        evaluation_baseline_roi=-0.02,
        min_n=40,
    )
    assert result["disposition"] == m.DISPOSITION_SUPPORTED
    assert result["n_2025_supported"] == 50
    assert result["gate_n_ge_min"] is True
    assert result["gate_roi_2025_positive"] is True
    assert result["gate_roi_2025_beats_baseline"] is True
    assert result["gate_2024_directionally_consistent"] is True
    assert result["rescue_authorized"] is False


def test_score_architecture_gate_fails_closed_on_insufficient_n():
    reference_bucket_supported = np.array([True] * 60)
    reference_realized_positive = np.array([True] * 60)
    evaluation_bucket_supported = np.array([True] * 10 + [False] * 90)
    evaluation_realized_roi = np.array([0.20] * 10 + [0.0] * 90)
    result = m.score_architecture_gate(
        reference_bucket_supported=reference_bucket_supported,
        reference_realized_positive=reference_realized_positive,
        evaluation_bucket_supported=evaluation_bucket_supported,
        evaluation_realized_roi=evaluation_realized_roi,
        evaluation_baseline_roi=0.0,
        min_n=40,
    )
    assert result["disposition"] == m.DISPOSITION_NOT_ACTIONABLE
    assert result["gate_n_ge_min"] is False


def test_score_architecture_gate_fails_closed_on_negative_2025_roi():
    reference_bucket_supported = np.array([True] * 60)
    reference_realized_positive = np.array([True] * 60)
    evaluation_bucket_supported = np.array([True] * 50 + [False] * 50)
    evaluation_realized_roi = np.array([-0.03] * 50 + [0.0] * 50)
    result = m.score_architecture_gate(
        reference_bucket_supported=reference_bucket_supported,
        reference_realized_positive=reference_realized_positive,
        evaluation_bucket_supported=evaluation_bucket_supported,
        evaluation_realized_roi=evaluation_realized_roi,
        evaluation_baseline_roi=-0.02,
        min_n=40,
    )
    assert result["disposition"] == m.DISPOSITION_NOT_ACTIONABLE
    assert result["gate_roi_2025_positive"] is False


def test_end_to_end_pipeline_on_synthetic_fixture_no_real_data():
    """Exercises Sections 1,3-8 back to back on fabricated data only."""
    rng = np.random.default_rng(42)
    n_ref, n_eval = 120, 40

    def make_pool(season, n, key_prefix):
        rows = []
        for i in range(n):
            feats = {c: float(rng.normal(0, 1)) for c in m.FEATURE_COLUMNS}
            rows.append({
                "season": season, "week": (i % 17) + 1, "team": "KC",
                "player_clean_key": f"{key_prefix}{i}", **feats,
            })
        return pd.DataFrame(rows)

    features_df = pd.concat([make_pool(2024, n_ref, "r"), make_pool(2025, n_eval, "e")], ignore_index=True)
    vegas_df = features_df[m.JOIN_KEYS].copy()
    vegas_df["realized_direction"] = rng.integers(0, 2, size=len(vegas_df)).astype(bool)
    vegas_df["realized_roi"] = np.where(vegas_df["realized_direction"], 0.05, -0.05)

    merged = m.join_vegas_and_features(vegas_df, features_df)
    clean, dropped = m.exclude_missing(merged)
    assert dropped == 0
    assert len(clean) == n_ref + n_eval

    reference = clean.loc[clean["season"] == 2024].reset_index(drop=True)
    evaluation = clean.loc[clean["season"] == 2025].reset_index(drop=True)

    scaler = m.fit_frozen_scaler(reference)
    ref_std = m.standardize(reference, scaler)
    eval_std = m.standardize(evaluation, scaler)

    threshold = m.leave_one_out_density_threshold(ref_std, k=m.K_NEIGHBORS, percentile=m.DENSITY_PERCENTILE)

    eval_ref_dist = m.pairwise_euclidean(eval_std, ref_std)
    neighbor_idx, neighbor_dist = m.k_nearest(eval_ref_dist, k=m.K_NEIGHBORS)
    eval_density_pass = m.density_gate_pass(neighbor_dist[:, -1], threshold)

    ref_ref_dist = m.pairwise_euclidean(ref_std, ref_std)
    np.fill_diagonal(ref_ref_dist, np.inf)
    ref_neighbor_idx, ref_neighbor_dist = m.k_nearest(ref_ref_dist, k=m.K_NEIGHBORS)
    ref_density_pass = m.density_gate_pass(ref_neighbor_dist[:, -1], threshold)

    ref_direction_supported = m.analog_direction_consistency(
        reference["realized_direction"].to_numpy(), ref_neighbor_idx
    )
    eval_direction_supported = m.analog_direction_consistency(
        reference["realized_direction"].to_numpy(), neighbor_idx
    )

    ref_evidence = m.classify_row_evidence(ref_density_pass, ref_direction_supported)
    eval_evidence = m.classify_row_evidence(eval_density_pass, eval_direction_supported)
    assert len(ref_evidence) == n_ref
    assert len(eval_evidence) == n_eval
    assert set(ref_evidence) <= {m.EVIDENCE_SUPPORTED, m.EVIDENCE_DESCRIPTIVE_ONLY, m.EVIDENCE_NO_ANALOG_SUPPORT}

    reference_bucket_supported = ref_density_pass & ref_direction_supported
    evaluation_bucket_supported = eval_density_pass & eval_direction_supported
    baseline_roi = float(evaluation["realized_roi"].mean()) if len(evaluation) else 0.0

    result = m.score_architecture_gate(
        reference_bucket_supported=reference_bucket_supported,
        reference_realized_positive=reference["realized_direction"].to_numpy(),
        evaluation_bucket_supported=evaluation_bucket_supported,
        evaluation_realized_roi=evaluation["realized_roi"].to_numpy(),
        evaluation_baseline_roi=baseline_roi,
        min_n=5,
    )
    assert result["disposition"] in {m.DISPOSITION_SUPPORTED, m.DISPOSITION_NOT_ACTIONABLE}
    assert result["rescue_authorized"] is False
