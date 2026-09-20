import numpy as np
import pandas as pd

from scripts.football_context import qualify_ol_roster_continuity_v1 as ol
from scripts.football_context.qualify_defensive_front_pairwise_cohesion_v1 import (
    CANDIDATE, IMMEDIATE, LOOKBACK_GAMES,
    _redundancy_design, build_front_sets, materialize,
)


def test_front_eligibility_uses_position_or_depth_and_excludes_db():
    schedule=ol.normalize_schedule(pd.DataFrame([
        {"season":2024,"week":1,"team":"A"},
    ]))
    roster=pd.DataFrame([
        {"season":2024,"week":1,"team":"A","gsis_id":"de","position":"DE","depth_chart_position":"DE"},
        {"season":2024,"week":1,"team":"A","gsis_id":"lb","position":"LB","depth_chart_position":"ILB"},
        {"season":2024,"week":1,"team":"A","gsis_id":"edge","position":"LB","depth_chart_position":"EDGE"},
        {"season":2024,"week":1,"team":"A","gsis_id":"cb","position":"CB","depth_chart_position":"CB"},
    ])
    sets,report=build_front_sets(roster,schedule)
    assert sets[(2024,1,"A")]=={"de","lb","edge"}
    assert report["stable_id_coverage"]==1.0


def test_crossseason_pair_history_and_immediate_continuity():
    schedule=ol.normalize_schedule(pd.DataFrame([
        {"season":2023,"week":18,"team":"A"},
        {"season":2024,"week":1,"team":"A"},
    ]))
    sets={(2023,18,"A"):{"a","b","x"},(2024,1,"A"):{"a","b","y"}}
    out,integrity=materialize(schedule,sets)
    r=out[(out.season==2024)&(out.week==1)].iloc[0]
    assert abs(r[IMMEDIATE]-(2/3))<1e-12
    # current pairs ab, ay, by; only ab shared in sole prior game
    assert abs(r[CANDIDATE]-(1/3))<1e-12
    assert integrity["chronology_violations"]==0


def test_first_horizon_game_unknown_not_zero():
    schedule=ol.normalize_schedule(pd.DataFrame([{"season":2019,"week":1,"team":"A"}]))
    out,_=materialize(schedule,{(2019,1,"A"):{"a","b"}})
    assert out.iloc[0].cohesion_state=="UNKNOWN_NO_PRIOR_HISTORY"
    assert np.isnan(out.iloc[0][CANDIDATE])


def test_lookback_capped_at_twenty():
    schedule=ol.normalize_schedule(pd.DataFrame([
        {"season":2024,"week":w,"team":"A"} for w in range(1,23)
    ]))
    sets={(2024,w,"A"):{"a","b"} for w in range(1,23)}
    out,_=materialize(schedule,sets)
    r=out[out.week==22].iloc[0]
    assert LOOKBACK_GAMES==20
    assert r.prior_scheduled_games_in_window==20
    assert abs(r[CANDIDATE]-1.0)<1e-12


def test_redundancy_floor_fail_closed():
    rows=[]
    for i in range(1499):
        tr=i<999
        rows.append({
            "season":2020 if tr else 2024,"week":2+(i%16),"team":f"T{i%8}",
            CANDIDATE:.5,IMMEDIATE:.8,
            "prior_pressure_rate_generated":.2,"prior_success_rate_def":.5,
            "prior_def_pass_epa":0.0,"prior_explosive_play_rate_allowed":.1,
        })
    Xtr,ytr,Xte,yte,ntr,nte=_redundancy_design(pd.DataFrame(rows))
    assert ntr==999 and nte==500
    assert Xtr.size==0 and Xte.size==0
