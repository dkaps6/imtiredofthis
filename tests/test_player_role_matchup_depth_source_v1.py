from __future__ import annotations

import pandas as pd

from scripts.research.audit_player_role_matchup_depth_source_v1 import (
    BANDS,
    defense_state,
    depth_band,
    player_state,
)


def test_frozen_depth_bands():
    assert depth_band(-1.0)=="BEHIND"
    assert depth_band(0.0)=="SHORT"
    assert depth_band(9.99)=="SHORT"
    assert depth_band(10.0)=="INTERMEDIATE"
    assert depth_band(19.99)=="INTERMEDIATE"
    assert depth_band(20.0)=="DEEP"


def test_player_state_requires_four_games_and_ten_targets():
    rows=[]
    for week in range(1,5):
        rows.append({
            "time_key":202400+week,
            "game_id":f"g{week}",
            "targets":3,
            "air_sum":15.0,
            "air_sq_sum":75.0,
            "targets_behind":0,
            "targets_short":2,
            "targets_intermediate":1,
            "targets_deep":0,
        })
    got=player_state(pd.DataFrame(rows))
    assert got["ready"] is True
    assert got["prior_games"]==4
    assert got["prior_targets"]==12
    assert abs(
        got["share_behind"]+
        got["share_short"]+
        got["share_intermediate"]+
        got["share_deep"]-1.0
    )<1e-12
    assert got["mass_gap"]<=1e-12


def test_player_state_fails_below_target_floor():
    rows=[]
    for week in range(1,5):
        rows.append({
            "time_key":202400+week,
            "game_id":f"g{week}",
            "targets":2,
            "air_sum":8.0,
            "air_sq_sum":32.0,
            "targets_behind":0,
            "targets_short":2,
            "targets_intermediate":0,
            "targets_deep":0,
        })
    got=player_state(pd.DataFrame(rows))
    assert got["ready"] is False
    assert got["prior_targets"]==8


def test_defense_state_requires_all_four_bands_supported():
    rows=[]
    for week in range(1,5):
        row={"time_key":202400+week,"game_id":f"g{week}"}
        for band in BANDS:
            key=band.lower()
            row[f"targets_{key}"]=2
            row[f"completions_{key}"]=1
            row[f"yards_{key}"]=10.0
        rows.append(row)
    got=defense_state(pd.DataFrame(rows))
    assert got["ready"] is True
    for band in BANDS:
        key=band.lower()
        assert got[f"targets_{key}"]==8
        assert abs(got[f"comp_{key}"]-0.5)<1e-12
        assert abs(got[f"ypt_{key}"]-5.0)<1e-12

    sparse=pd.DataFrame(rows).copy()
    sparse["targets_deep"]=1
    sparse["completions_deep"]=0
    sparse["yards_deep"]=0.0
    got2=defense_state(sparse)
    assert got2["ready"] is False


def test_player_state_zero_prior_history_is_unsupported_not_error():
    got=player_state(pd.DataFrame())
    assert got=={"ready":False,"prior_games":0,"prior_targets":0}
