"""Tests for full-board backtest outcome provenance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.operations.backtest_full_report_v1 import _apply_verified_zero_outcomes


def test_verified_zero_provenance_uses_prefill_missingness():
    d = pd.DataFrame(
        [
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed": True,
                "actual": 12.0,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed": True,
                "actual": np.nan,
            },
            {
                "identity_status": "UNRESOLVED_IDENTITY",
                "roster_confirmed": False,
                "actual": np.nan,
            },
        ]
    )

    out = _apply_verified_zero_outcomes(d)

    assert out.loc[0, "actual_source"] == "stats_table"
    assert out.loc[0, "actual"] == 12.0

    assert out.loc[1, "actual_source"] == "roster_confirmed_verified_zero"
    assert out.loc[1, "actual"] == 0.0

    assert out.loc[2, "actual_source"] == "unresolved"
    assert pd.isna(out.loc[2, "actual"])
