from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest.build_authority_exact_vegas_projection_trace_v1 import (
    QB_BASE_MAE,
    QB_SYNTH_MAE,
    validate_qb,
)


def _qb_frame() -> pd.DataFrame:
    rows = []
    for season, n in [(2024, 444), (2025, 440)]:
        for i in range(n):
            rows.append({
                "season": season,
                "week": (i % 18) + 1,
                "team": f"T{i:03d}",
                "opponent": "OPP",
                "player_clean_key": f"qb_{season}_{i}",
                "actual_pass_yards": 0.0,
                "base_proj": QB_BASE_MAE,
                "football_synthesis": QB_SYNTH_MAE,
            })
    return pd.DataFrame(rows)


def test_qb_authority_parity_accepts_exact_frozen_contract():
    result = validate_qb(_qb_frame())
    assert result["status"] == "PASS"
    assert result["rows"] == 884
    assert result["candidate_metric"] == pytest.approx(QB_SYNTH_MAE)


def test_qb_authority_parity_fails_closed_on_metric_drift():
    df = _qb_frame()
    df.loc[0, "football_synthesis"] += 1.0
    with pytest.raises(RuntimeError, match="QB synthesis MAE drifted"):
        validate_qb(df)
