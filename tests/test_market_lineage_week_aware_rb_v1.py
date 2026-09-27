import pandas as pd
import pytest

from scripts.audit_market_model_lineage_v1 import _rb_market_lineage_specs


def _frame(*, week3=True, p3_applied=0, v2_applied=1, v2_version="RB_RUSH_REC_CONSERVATION_V2"):
    rows = [
        {
            "source_market": "player_rush_yds",
            "position_family": "RB/FB",
            "rb_synthesis_applied": p3_applied,
            "rb_rush_rec_conservation_v2_applied": 0,
            "rb_rush_rec_conservation_v2_version": "",
        },
        {
            "source_market": "player_rush_reception_yds",
            "position_family": "RB/FB",
            "rb_synthesis_applied": 0,
            "rb_rush_rec_conservation_v2_applied": v2_applied,
            "rb_rush_rec_conservation_v2_version": v2_version,
        },
    ]
    return pd.DataFrame(rows)


def test_week3_rb_lineage_uses_generic_rushing_and_conservation_v2():
    rush, combo = _rb_market_lineage_specs(_frame(), 3)
    assert rush[0] == "canonical calibrated rush-yards ensemble + joint MC"
    assert rush[2] is False
    assert rush[3] == "GENERIC_CANONICAL_ACTIVE"
    assert "Week-1-only" in rush[5]

    assert combo[0].startswith("RB_RUSH_REC_CONSERVATION_V2")
    assert combo[2] is True
    assert combo[3] == "PROMOTED_RB_RUSH_REC_CONSERVATION_V2_ACTIVE"
    assert "P3" not in combo[0]


def test_week3_rb_lineage_fails_if_p3_claims_to_be_applied():
    with pytest.raises(RuntimeError, match="unexpectedly claim P3"):
        _rb_market_lineage_specs(_frame(p3_applied=1), 3)


def test_week3_rb_lineage_fails_if_conservation_v2_is_missing():
    with pytest.raises(RuntimeError, match="do not all consume"):
        _rb_market_lineage_specs(_frame(v2_applied=0), 3)


def test_week3_rb_lineage_fails_on_conservation_v2_version_drift():
    with pytest.raises(RuntimeError, match="version drift"):
        _rb_market_lineage_specs(_frame(v2_version="WRONG"), 3)


def test_week1_rb_lineage_preserves_frozen_p3_semantics():
    rush, combo = _rb_market_lineage_specs(_frame(p3_applied=1, v2_applied=0, v2_version=""), 1)
    assert rush[0] == "RB_P3_SYNTHESIS_V1 / WEEK1_STACK_OVERRIDE"
    assert rush[3] == "PROMOTED_SPECIALIST_ACTIVE"
    assert combo[0].startswith("RB P3-conserved rushing component")
