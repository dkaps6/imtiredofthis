import numpy as np
import pandas as pd
import pytest

from scripts.research.audit_qb_m89_opportunity_efficiency_v1 import (
    TOL,
    _summary_rows,
    build_decomposition,
)


def _trace():
    return pd.DataFrame([
        {
            "season": 2024,
            "week": 1,
            "team": "BUF",
            "player_clean_key": "QB One",
            "actual_pass_yards": 240.0,
            "football_synthesis": 270.0,
            "pred_attempts": 30.0,
            "pred_ypa": 8.0,
        },
        {
            "season": 2025,
            "week": 2,
            "team": "KC",
            "player_clean_key": "QB Two",
            "actual_pass_yards": 210.0,
            "football_synthesis": 190.0,
            "pred_attempts": 28.0,
            "pred_ypa": 7.0,
        },
    ])


def _logs():
    return pd.DataFrame([
        {
            "season": 2024,
            "week": 1,
            "team": "BUF",
            "player_clean_key": "qb one",
            "pass_att": 32,
            "pass_yards": 240,
        },
        {
            "season": 2025,
            "week": 2,
            "team": "KC",
            "player_clean_key": "qb two",
            "pass_att": 30,
            "pass_yards": 210,
        },
    ])


def test_shapley_identity_and_oracles():
    x = build_decomposition(_trace(), _logs())
    assert len(x) == 2
    assert float(x["decomposition_identity_gap"].abs().max()) <= TOL
    assert np.allclose(
        x["both_primitives_oracle_proj"],
        x["actual_pass_yards"] + x["nonfactor_residual"],
        rtol=0,
        atol=TOL,
    )

    row = x.iloc[0]
    # P=30, A=32, Yp=8, Ya=7.5, M=270, D=240, R=30.
    assert row["nonfactor_residual"] == pytest.approx(30.0)
    assert row["opportunity_contribution"] == pytest.approx((30 - 32) * (8 + 7.5) / 2)
    assert row["efficiency_contribution"] == pytest.approx((8 - 7.5) * (30 + 32) / 2)
    assert (
        row["opportunity_contribution"]
        + row["efficiency_contribution"]
        + row["nonfactor_residual"]
    ) == pytest.approx(row["total_error"])


def test_summary_has_pooled_and_seasons():
    x = build_decomposition(_trace(), _logs())
    s = _summary_rows(x)
    assert ((s.dimension == "POOLED") & (s.bucket == "ALL")).sum() == 1
    assert set(s.loc[s.dimension.eq("SEASON"), "bucket"]) == {"2024", "2025"}


def test_actual_identity_mismatch_fails_closed():
    bad = _logs()
    bad.loc[0, "pass_yards"] = 241
    with pytest.raises(RuntimeError, match="actual passing-yard mismatch"):
        build_decomposition(_trace(), bad)


def test_missing_primitive_fails_closed():
    bad = _trace()
    bad.loc[0, "pred_attempts"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite M89 decomposition primitives"):
        build_decomposition(bad, _logs())


def test_duplicate_trace_identity_fails_closed():
    bad = pd.concat([_trace(), _trace().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate M89"):
        build_decomposition(bad, _logs())
