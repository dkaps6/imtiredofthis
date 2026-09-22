"""Tests for heterogeneous-price significance in graded track-record slicing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.slice_graded_track_record_v1 import (
    cluster_score_pvalue,
    poisson_binomial_tail,
    summarize,
)


def test_poisson_binomial_tail_matches_manual_heterogeneous_example():
    # P(X >= 2) for p=[.1,.3,.8]:
    # exactly two = .006 + .056 + .216; all three = .024.
    assert abs(poisson_binomial_tail(2, [0.1, 0.3, 0.8]) - 0.302) < 1e-12


def test_poisson_binomial_tail_handles_boundaries():
    assert poisson_binomial_tail(0, [0.2, 0.7]) == 1.0
    assert poisson_binomial_tail(3, [0.2, 0.7]) == 0.0



def _clustered_rows():
    rows = []
    for event_id, results in (
        ("g1", ["WIN", "WIN"]),
        ("g2", ["WIN", "LOSS"]),
        ("g3", ["LOSS", "LOSS"]),
        ("g4", ["WIN", "LOSS"]),
        ("g5", ["WIN", "WIN"]),
        ("g6", ["LOSS", "WIN"]),
        ("g7", ["WIN", "LOSS"]),
        ("g8", ["WIN", "WIN"]),
    ):
        for result in results:
            rows.append({
                "season": 2026,
                "week": 1,
                "event_id": event_id,
                "bet_result": result,
                "unit_result": 1.0 if result == "WIN" else -1.0,
                "vegas_odds": -110.0,
                "model_closer_than_vegas": True,
            })
    return pd.DataFrame(rows)


def test_cluster_score_counts_games_not_rows():
    df = _clustered_rows()
    p = np.full(len(df), 110.0 / 210.0)
    value, clusters = cluster_score_pvalue(df, p)
    assert clusters == 8
    assert np.isfinite(value)


def test_cluster_score_fails_closed_without_enough_games():
    df = _clustered_rows().loc[lambda x: x["event_id"].isin(["g1", "g2", "g3"])].copy()
    p = np.full(len(df), 110.0 / 210.0)
    value, clusters = cluster_score_pvalue(df, p)
    assert clusters == 3
    assert np.isnan(value)


def test_summarize_uses_cluster_p_for_discovery_but_keeps_independent_reference():
    df = _clustered_rows()
    out = summarize(df, "ALL", "overall")
    assert out["clusters"] == 8
    assert np.isfinite(out["p_value"])
    assert np.isfinite(out["independent_p_value"])
