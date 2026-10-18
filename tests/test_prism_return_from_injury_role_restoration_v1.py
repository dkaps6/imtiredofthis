import pandas as pd
import pytest

from scripts.research.audit_prism_return_from_injury_role_restoration_v1 import (
    NO_RETURN,
    RETURN_REDUCED,
    RETURN_RESTORED,
    build_episodes,
)


def _h(week, pos, pct, injury, state):
    return {
        "season":2024,
        "week":week,
        "team":"CAR",
        "stable_id":"P1",
        "family":"WR",
        "positive_participation":pos,
        "offense_snaps":(50*pct if pos else 0),
        "offense_pct":(pct if pos else 0.0),
        "injury_reported":injury,
        "observation_state":state,
    }


def test_partial_then_dnp_then_restored_return_is_one_episode():
    health=pd.DataFrame([
        _h(1,True,0.80,False,"NO_REPORT_PARTICIPATED"),
        _h(2,True,0.82,False,"NO_REPORT_PARTICIPATED"),
        _h(3,True,0.78,False,"NO_REPORT_PARTICIPATED"),
        _h(4,True,0.30,True,"REPORTED_INJURY_PARTICIPATED"),
        _h(5,False,0.00,True,"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"),
        _h(6,True,0.79,False,"NO_REPORT_PARTICIPATED"),
    ])
    anomaly=pd.DataFrame([{
        "season":2024,"week":4,"team":"CAR","stable_id":"P1",
        "participation_anomaly_state":"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
        "player_prior_snap_pct":0.80,
    }])
    ep,audit=build_episodes(health,anomaly)
    assert len(ep)==1
    r=ep.iloc[0]
    assert r["episode_start_week"]==4
    assert r["return_week"]==6
    assert r["return_state"]==RETURN_RESTORED
    assert r["intervening_no_participation_rows"]==1
    assert r["pre_episode_snap_pct"]==pytest.approx(0.80)
    assert audit["baseline_crosscheck_max_abs_gap"]==pytest.approx(0.0)


def test_first_return_can_remain_reduced():
    health=pd.DataFrame([
        _h(1,True,0.80,False,"NO_REPORT_PARTICIPATED"),
        _h(2,True,0.82,False,"NO_REPORT_PARTICIPATED"),
        _h(3,True,0.78,False,"NO_REPORT_PARTICIPATED"),
        _h(4,False,0.00,True,"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"),
        _h(5,True,0.40,True,"REPORTED_INJURY_PARTICIPATED"),
    ])
    anomaly=pd.DataFrame(columns=[
        "season","week","team","stable_id","participation_anomaly_state",
        "player_prior_snap_pct"
    ])
    ep,_=build_episodes(health,anomaly)
    assert len(ep)==1
    r=ep.iloc[0]
    assert r["episode_start_week"]==4
    assert r["return_week"]==5
    assert r["return_state"]==RETURN_REDUCED


def test_no_later_positive_row_is_no_return_observed():
    health=pd.DataFrame([
        _h(1,True,0.80,False,"NO_REPORT_PARTICIPATED"),
        _h(2,True,0.82,False,"NO_REPORT_PARTICIPATED"),
        _h(3,True,0.78,False,"NO_REPORT_PARTICIPATED"),
        _h(4,False,0.00,True,"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"),
        _h(5,False,0.00,True,"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"),
    ])
    anomaly=pd.DataFrame(columns=[
        "season","week","team","stable_id","participation_anomaly_state",
        "player_prior_snap_pct"
    ])
    ep,_=build_episodes(health,anomaly)
    assert len(ep)==1
    assert ep.iloc[0]["return_state"]==NO_RETURN


def test_injury_nosnap_without_three_prior_positive_rows_is_not_episode():
    health=pd.DataFrame([
        _h(1,True,0.80,False,"NO_REPORT_PARTICIPATED"),
        _h(2,True,0.82,False,"NO_REPORT_PARTICIPATED"),
        _h(3,False,0.00,True,"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"),
        _h(4,True,0.79,False,"NO_REPORT_PARTICIPATED"),
    ])
    anomaly=pd.DataFrame(columns=[
        "season","week","team","stable_id","participation_anomaly_state",
        "player_prior_snap_pct"
    ])
    with pytest.raises(RuntimeError,match="zero supported"):
        build_episodes(health,anomaly)
