from __future__ import annotations

import pandas as pd

from scripts.research.lock_player_opportunity_share_volatility_future_week_v1 import (
    _feature,
    lock_from_frames,
)


def parent_result():
    return {
        "version":"PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1",
        "season":2026,
        "target_week":6,
        "disposition":"PLAYER_STATE_FUTURE_WEEK_READY",
        "target_week_roster_source_ready":True,
        "prior_week_source_completion":{"complete":True},
        "chronology_violations":0,
    }


def parent_rows():
    return pd.DataFrame([
        {
            "team":"IND","opponent":"HOU","player":"Back A","position_group":"RB",
            "gsis_id":"00-rb","name_key":"backa","roster_source_week":6,
        },
        {
            "team":"IND","opponent":"HOU","player":"Wide A","position_group":"WR",
            "gsis_id":"00-wr","name_key":"widea","roster_source_week":6,
        },
    ])


def weekly():
    rows=[]
    rb_targets=[0.10,0.00,0.20,0.10,0.15]
    rb_rush=[0.40,0.00,0.50,0.30,0.45]
    wr_targets=[0.05,0.40,0.10,0.35,0.20]
    for week in range(1,6):
        rows.append({
            "season":2026,"week":week,"team":"IND",
            "player_identity_key":"gsis:00-rb",
            "tgt_share_game":rb_targets[week-1],
            "rush_share_game":rb_rush[week-1],
            "team_targets":40.0,
            "team_rushes":30.0,
        })
        rows.append({
            "season":2026,"week":week,"team":"IND",
            "player_identity_key":"gsis:00-wr",
            "tgt_share_game":wr_targets[week-1],
            "rush_share_game":0.0,
            "team_targets":40.0,
            "team_rushes":30.0,
        })
    return pd.DataFrame(rows)


def test_feature_keeps_zero_share_games():
    got=_feature(
        pd.Series([0.4,0.0,0.5,0.3]),
        pd.Series([1,2,3,4]),
        pd.Series([20.0,20.0,20.0,20.0]),
    )
    assert got["supported"] is True
    assert got["prior_games"]==4
    assert abs(got["mean"]-0.30)<1e-12
    assert got["min"]==0.0
    assert abs(got["sampling_var"]-0.0105)<1e-12
    assert abs(got["excess_var"]-0.0245)<1e-12


def test_feature_fails_closed_under_four_games():
    got=_feature(pd.Series([0.1,0.2,0.3]),pd.Series([1,2,3]))
    assert got["supported"] is False
    assert got["prior_games"]==3


def test_lock_preserves_position_specific_confirmed_features():
    rows,result=lock_from_frames(
        parent_rows=parent_rows(),
        parent_result=parent_result(),
        weekly=weekly(),
        season=2026,
        target_week=6,
    )
    rb=rows.loc[rows["position"].eq("RB")].iloc[0]
    wr=rows.loc[rows["position"].eq("WR")].iloc[0]
    assert bool(rb["rush_share_supported"]) is True
    assert bool(rb["target_share_supported"]) is True
    assert bool(wr["target_share_supported"]) is True
    assert bool(wr["rush_share_supported"]) is False
    assert rb["rush_share_prior_games"]==5
    assert rb["prior_rush_excess_share_var"]>0
    assert wr["prior_target_excess_share_var"]>0
    assert bool(rb["rb_carry_betabinomial_eligible"]) is True
    assert bool(wr["wr_target_betabinomial_eligible"]) is True
    assert result["rb_carry_betabinomial_eligible"]==1
    assert result["wr_target_betabinomial_eligible"]==1
    assert result["rb_target_betabinomial_authorized"] is False
    assert result["te_target_betabinomial_authorized"] is False
    assert result["allocation_uncertainty_candidate_cells"]==[
        "RB_CARRY_SHARE","WR_TARGET_SHARE"
    ]
    assert result["projection_changes"]==0
    assert result["target_week_outcomes_read"]==0
    assert result["chronology_violations"]==0
