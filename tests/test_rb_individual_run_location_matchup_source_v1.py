from __future__ import annotations

import pandas as pd

from scripts.research.audit_rb_individual_run_location_matchup_source_v1 import (
    LOCATIONS,
    defense_state,
    player_state,
    run_location,
)


def test_run_location_canonicalization():
    assert run_location("left")=="LEFT"
    assert run_location("middle")=="MIDDLE"
    assert run_location("mid")=="MIDDLE"
    assert run_location("right")=="RIGHT"
    assert run_location("guard")==""


def test_player_state_requires_four_games_and_twenty_located_rushes():
    rows=[]
    for week in range(1,5):
        rows.append({
            "time_key":202400+week,"game_id":f"g{week}",
            "rushes":5,"yards":25.0,
            "rushes_left":2,"rushes_middle":2,"rushes_right":1,
        })
    got=player_state(pd.DataFrame(rows))
    assert got["ready"] is True
    assert got["located_rushes"]==20
    assert abs(got["share_left"]+got["share_middle"]+got["share_right"]-1.0)<1e-12
    assert got["mass_gap"]<=1e-12


def test_player_state_below_rush_floor_fails_closed():
    rows=[]
    for week in range(1,5):
        rows.append({
            "time_key":202400+week,"game_id":f"g{week}",
            "rushes":4,"yards":20.0,
            "rushes_left":2,"rushes_middle":1,"rushes_right":1,
        })
    got=player_state(pd.DataFrame(rows))
    assert got["ready"] is False
    assert got["located_rushes"]==16


def test_zero_history_player_is_unsupported():
    assert player_state(pd.DataFrame())=={"ready":False,"prior_games":0,"located_rushes":0}


def test_defense_state_requires_each_location_floor_and_complete_success():
    rows=[]
    for week in range(1,5):
        row={"time_key":202400+week,"game_id":f"g{week}"}
        for loc in LOCATIONS:
            k=loc.lower()
            row[f"rushes_{k}"]=3
            row[f"yards_{k}"]=15.0
            row[f"success_n_{k}"]=3
            row[f"success_sum_{k}"]=1.5
        rows.append(row)
    got=defense_state(pd.DataFrame(rows))
    assert got["ready"] is True
    for loc in LOCATIONS:
        k=loc.lower()
        assert got[f"rushes_{k}"]==12
        assert abs(got[f"ypc_{k}"]-5.0)<1e-12
        assert abs(got[f"success_{k}"]-0.5)<1e-12

    sparse=pd.DataFrame(rows).copy()
    sparse["rushes_right"]=2
    sparse["success_n_right"]=2
    got2=defense_state(sparse)
    assert got2["ready"] is False
