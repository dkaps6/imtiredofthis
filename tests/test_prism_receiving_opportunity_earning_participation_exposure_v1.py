import pandas as pd
import pytest

from scripts.research.audit_prism_receiving_opportunity_earning_participation_exposure_v1 import (
    build_panel,
)


def test_partial_participation_preserves_targets_and_exposure_separately():
    health=pd.DataFrame([{
        "season":2024,"week":3,"team":"CAR","stable_id":"P1","family":"WR",
        "positive_participation":True,"offense_snaps":20.0,"offense_pct":0.30,
        "injury_reported":True,"observation_state":"REPORTED_INJURY_PARTICIPATED",
    }])
    anomaly=pd.DataFrame([{
        "season":2024,"week":3,"team":"CAR","stable_id":"P1",
        "participation_anomaly_state":"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
        "player_prior_snap_pct":0.80,"snap_retention_ratio":0.375,
        "corroboration_source":"CURRENT_WEEK_INJURY","corroboration_week":3,
    }])
    player=pd.DataFrame([{
        "season":2024,"week":3,"team":"CAR","stable_id":"P1","family":"WR",
        "targets":4.0,"team_targets":30.0,
    }])
    team=pd.DataFrame([{
        "season":2024,"week":3,"team":"CAR","team_targets":30.0,
    }])

    out=build_panel(health,anomaly,player,team)
    r=out.iloc[0]
    assert r["targets"] == pytest.approx(4.0)
    assert r["offense_pct"] == pytest.approx(0.30)
    assert r["target_share_game"] == pytest.approx(4/30)
    assert r["targets_per_offensive_snap"] == pytest.approx(4/20)
    assert r["targets_per_snap_scaled_team_target"] == pytest.approx(4/(30*0.30))
    assert r["participation_anomaly_state"] == "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED"


def test_positive_snap_zero_usage_player_is_zero_filled_not_dropped():
    health=pd.DataFrame([{
        "season":2024,"week":2,"team":"CAR","stable_id":"P2","family":"TE",
        "positive_participation":True,"offense_snaps":15.0,"offense_pct":0.25,
        "injury_reported":False,"observation_state":"NO_REPORT_PARTICIPATED",
    }])
    anomaly=pd.DataFrame([{
        "season":2024,"week":2,"team":"CAR","stable_id":"P2",
        "participation_anomaly_state":"PARTICIPATION_NOT_COLLAPSED",
        "player_prior_snap_pct":0.25,"snap_retention_ratio":1.0,
        "corroboration_source":"NONE","corroboration_week":None,
    }])
    player=pd.DataFrame(columns=[
        "season","week","team","stable_id","family","targets","team_targets"
    ])
    team=pd.DataFrame([{
        "season":2024,"week":2,"team":"CAR","team_targets":28.0,
    }])

    out=build_panel(health,anomaly,player,team)
    r=out.iloc[0]
    assert bool(r["playerform_weekly_row_present"]) is False
    assert r["targets"] == pytest.approx(0.0)
    assert r["target_share_game"] == pytest.approx(0.0)
    assert r["targets_per_offensive_snap"] == pytest.approx(0.0)


def test_join_is_exact_stable_id_week_team():
    health=pd.DataFrame([{
        "season":2025,"week":4,"team":"CAR","stable_id":"P3","family":"RB",
        "positive_participation":True,"offense_snaps":30.0,"offense_pct":0.50,
        "injury_reported":False,"observation_state":"NO_REPORT_PARTICIPATED",
    }])
    anomaly=pd.DataFrame(columns=[
        "season","week","team","stable_id","participation_anomaly_state",
        "player_prior_snap_pct","snap_retention_ratio","corroboration_source",
        "corroboration_week"
    ])
    player=pd.DataFrame([{
        "season":2025,"week":5,"team":"CAR","stable_id":"P3","family":"RB",
        "targets":5.0,"team_targets":25.0,
    }])
    team=pd.DataFrame([{
        "season":2025,"week":4,"team":"CAR","team_targets":25.0,
    }])

    out=build_panel(health,anomaly,player,team)
    assert out.iloc[0]["targets"] == pytest.approx(0.0)
