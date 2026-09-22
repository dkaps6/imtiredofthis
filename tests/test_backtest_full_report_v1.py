"""Tests for postgame settlement/provenance used by the full-board replay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.operations.grade_market_track_record_gsis_v1 import apply_postgame_settlement


def _rows():
    return pd.DataFrame(
        [
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": True,
                "book": "draftkings",
                "actual": 12.0,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": True,
                "book": "fanduel",
                "actual": np.nan,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "INA",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
            {
                "identity_status": "RESOLVED_GSIS",
                "roster_confirmed_this_team_week": True,
                "roster_status": "ACT",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
            {
                "identity_status": "UNRESOLVED_IDENTITY",
                "roster_confirmed_this_team_week": False,
                "roster_status": "",
                "snap_participated": False,
                "book": "draftkings",
                "actual": np.nan,
            },
        ]
    )


def test_postgame_settlement_distinguishes_stats_zero_dnp_and_unknown():
    out = apply_postgame_settlement(_rows())

    assert out.loc[0, "actual_source"] == "stats_table"
    assert out.loc[0, "settlement_status"] == "SETTLED"
    assert out.loc[0, "actual"] == 12.0

    assert out.loc[1, "actual_source"] == "snap_confirmed_verified_zero"
    assert out.loc[1, "settlement_status"] == "SETTLED"
    assert out.loc[1, "actual"] == 0.0

    assert out.loc[2, "actual_source"] == "sportsbook_void_dnp"
    assert out.loc[2, "settlement_status"] == "VOID"
    assert pd.isna(out.loc[2, "actual"])

    # Active roster status without any snap evidence does not prove action.
    assert out.loc[3, "actual_source"] == "unresolved"
    assert out.loc[3, "settlement_status"] == "UNRESOLVED"
    assert pd.isna(out.loc[3, "actual"])

    assert out.loc[4, "actual_source"] == "unresolved"
    assert out.loc[4, "settlement_status"] == "UNRESOLVED"


def test_dnp_void_rule_is_fail_closed_for_unknown_book():
    row = _rows().iloc[[2]].copy()
    row["book"] = "unknownbook"
    out = apply_postgame_settlement(row)
    assert out.iloc[0]["settlement_status"] == "UNRESOLVED"
