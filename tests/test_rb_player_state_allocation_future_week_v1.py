from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.lock_rb_player_state_allocation_future_week_v1 import (
    build,
    certify_parent_result,
)


def parent_meta():
    return {
        "version":"PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1",
        "season":2026,
        "target_week":6,
        "disposition":"PLAYER_STATE_FUTURE_WEEK_READY",
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "candidate_models_fit":0,
        "production_changed":False,
        "prior_week_source_completion":{"complete":True},
    }


def rows():
    return pd.DataFrame([
        {
            "team":"IND","opponent":"HOU","state_key":"gsis:A","player":"A",
            "gsis_id":"A","pfr_id":"PA","position_group":"RB","room_size":2,
            "last3_room_opportunity_share":0.75,"last3_room_snap_fraction":0.50,
            "chronology_valid":True,"last3_source_max_week":5,"snap_source_max_week":5,
        },
        {
            "team":"IND","opponent":"HOU","state_key":"gsis:B","player":"B",
            "gsis_id":"B","pfr_id":"PB","position_group":"RB","room_size":2,
            "last3_room_opportunity_share":0.25,"last3_room_snap_fraction":0.50,
            "chronology_valid":True,"last3_source_max_week":5,"snap_source_max_week":5,
        },
    ])


def test_parent_must_be_ready_and_complete():
    certify_parent_result(parent_meta(),2026,6)
    bad=parent_meta()
    bad["prior_week_source_completion"]={"complete":False}
    with pytest.raises(RuntimeError):
        certify_parent_result(bad,2026,6)


def test_future_week_builder_preserves_exact_frozen_50_50_rule():
    out,teams,result=build(rows(),2026,6,parent_meta())
    got=dict(zip(out["player"],out["shadow_player_state_share"]))
    assert abs(got["A"]-0.625)<1e-12
    assert abs(got["B"]-0.375)<1e-12
    assert abs(float(out["shadow_player_state_share"].sum())-1.0)<1e-12
    assert abs(float(out["control_recent_carry_share"].sum())-1.0)<1e-12
    assert result["scientific_formula_changed_from_week5"] is False
    assert result["target_week_outcomes_read"]==0
    assert result["sportsbook_inputs_used"]==0
    assert result["parameters_fit"]==0


def test_future_week_builder_rejects_same_week_features():
    x=rows()
    x.loc[0,"last3_source_max_week"]=6
    with pytest.raises(RuntimeError):
        build(x,2026,6,parent_meta())
