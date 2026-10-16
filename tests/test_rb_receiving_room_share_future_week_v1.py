from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.lock_rb_receiving_room_share_future_week_v1 import (
    certify_parents,
    normalize_receiving_room_share,
)


def rb_meta():
    return {
        "season":2026,"week":6,"status":"WEEK6_PREGAME_LOCK_FROZEN",
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,
        "production_changed":False,
    }


def live_meta():
    return {
        "season":2026,"target_week":6,"disposition":"PLAYER_STATE_FUTURE_WEEK_READY",
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,
        "production_changed":False,
        "prior_week_source_completion":{"complete":True},
    }


def test_parents_must_match_and_be_clean():
    certify_parents(season=2026,week=6,rb_lock_meta=rb_meta(),live_state_meta=live_meta())
    bad=live_meta()
    bad["sportsbook_inputs_used"]=1
    with pytest.raises(RuntimeError):
        certify_parents(season=2026,week=6,rb_lock_meta=rb_meta(),live_state_meta=bad)


def test_receiving_rule_is_exact_within_room_normalization():
    x=pd.DataFrame([
        {
            "team":"IND","prior_rb_room_share":0.6,
            "shadow_player_state_share":0.7,
        },
        {
            "team":"IND","prior_rb_room_share":0.3,
            "shadow_player_state_share":0.3,
        },
        {
            "team":"IND","prior_rb_room_share":0.1,
            "shadow_player_state_share":0.0,
        },
    ])
    out,teams=normalize_receiving_room_share(x)
    got=out["candidate_rb_receiving_room_share"].tolist()
    assert all(abs(a-b)<1e-12 for a,b in zip(got,[0.6,0.3,0.1]))
    assert abs(float(out["candidate_rb_receiving_room_share"].sum())-1.0)<1e-12
    assert abs(float(teams.iloc[0]["candidate_sum"])-1.0)<1e-12


def test_receiving_rule_rejects_single_player_room():
    x=pd.DataFrame([{
        "team":"IND","prior_rb_room_share":1.0,
        "shadow_player_state_share":1.0,
    }])
    with pytest.raises(RuntimeError):
        normalize_receiving_room_share(x)
