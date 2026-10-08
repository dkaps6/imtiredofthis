from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.research.run_player_output_component_decomposition_v1 as d


def _point_rows():
    return pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"NO","player":"WR A",
            "player_clean_key":"wra","position_family":"WR","market":"rec_yards",
            "projection_mean":40.0,"actual":80.0,"actual_opportunities":8.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"NO","player":"WR A",
            "player_clean_key":"wra","position_family":"WR","market":"receptions",
            "projection_mean":2.0,"actual":4.0,"actual_opportunities":8.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G2","team":"LV","player":"WR B",
            "player_clean_key":"wrb","position_family":"WR","market":"rec_yards",
            "projection_mean":20.0,"actual":40.0,"actual_opportunities":4.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G2","team":"LV","player":"WR B",
            "player_clean_key":"wrb","position_family":"WR","market":"receptions",
            "projection_mean":2.0,"actual":2.0,"actual_opportunities":4.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])


def _opp_rows():
    return pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"NO","player":"WR A",
            "player_clean_key":"wra","position_family":"WR","opportunity_type":"targets",
            "predicted_opportunities":4.0,"actual_opportunities":8.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G2","team":"LV","player":"WR B",
            "player_clean_key":"wrb","position_family":"WR","opportunity_type":"targets",
            "predicted_opportunities":4.0,"actual_opportunities":4.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])


def test_build_rows_separates_opportunity_and_efficiency_oracles():
    rows=d.build_rows(_point_rows(),_opp_rows(),require_full_weeks=False)

    a=rows.loc[(rows.player.eq("WR A")) & (rows.market.eq("rec_yards"))].iloc[0]
    assert a.model_effective_efficiency==pytest.approx(10.0)
    assert a.opportunity_oracle==pytest.approx(80.0)
    assert a.efficiency_oracle==pytest.approx(40.0)
    assert a.baseline_reconstructed==pytest.approx(40.0)

    b=rows.loc[(rows.player.eq("WR B")) & (rows.market.eq("rec_yards"))].iloc[0]
    assert b.model_effective_efficiency==pytest.approx(5.0)
    assert b.actual_efficiency==pytest.approx(10.0)
    assert b.opportunity_oracle==pytest.approx(20.0)
    assert b.efficiency_oracle==pytest.approx(40.0)


def test_summary_identifies_expected_mae_reduction():
    rows=d.build_rows(_point_rows(),_opp_rows(),require_full_weeks=False)
    rec=rows.loc[rows.market.eq("rec_yards")]
    s=d._summary(rec)
    assert s["baseline_mae"]==pytest.approx(30.0)
    assert s["opportunity_oracle_mae"]==pytest.approx(10.0)
    assert s["opportunity_oracle_mae_improvement"]==pytest.approx(20.0)
    # On both positive-opportunity rows the efficiency oracle is perfect for WR B
    # but not WR A, yielding 20 yards MAE.
    assert s["efficiency_oracle_mae"]==pytest.approx(20.0)


def test_zero_actual_opportunity_requires_zero_output():
    points=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"IND","player":"RB A",
        "player_clean_key":"rba","position_family":"RB","market":"rush_yards",
        "projection_mean":12.0,"actual":0.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    opp=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"IND","player":"RB A",
        "player_clean_key":"rba","position_family":"RB","opportunity_type":"carries",
        "predicted_opportunities":3.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    rows=d.build_rows(points,opp,require_full_weeks=False)
    r=rows.iloc[0]
    assert r.actual_efficiency==pytest.approx(0.0)
    assert not bool(r.efficiency_oracle_eligible)
    assert r.opportunity_oracle==pytest.approx(0.0)
    assert r.actual_workload_bin=="ZERO"


def test_nonzero_output_with_zero_actual_opportunity_fails_closed():
    points=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"IND","player":"RB A",
        "player_clean_key":"rba","position_family":"RB","market":"rush_yards",
        "projection_mean":12.0,"actual":5.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    opp=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"IND","player":"RB A",
        "player_clean_key":"rba","position_family":"RB","opportunity_type":"carries",
        "predicted_opportunities":3.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    with pytest.raises(RuntimeError,match="nonzero output with zero actual opportunity"):
        d.build_rows(points,opp,require_full_weeks=False)


def test_string_false_is_not_truthy():
    assert not d._truthy(pd.Series(["False","false","0","no"])).any()
    assert d._truthy(pd.Series(["True"])).all()


def test_pbp_target_authority_repairs_impossible_weekly_zero_target_case():
    points=pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"SF","player":"Receiver A",
            "player_clean_key":"receivera","position_family":"WR","market":"rec_yards",
            "projection_mean":30.0,"actual":80.0,"actual_opportunities":0.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"SF","player":"Receiver A",
            "player_clean_key":"receivera","position_family":"WR","market":"receptions",
            "projection_mean":3.0,"actual":5.0,"actual_opportunities":0.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])
    opp=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"SF","player":"Receiver A",
        "player_clean_key":"receivera","position_family":"WR","opportunity_type":"targets",
        "predicted_opportunities":4.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    pbp=pd.DataFrame([{
        "season":2026,"week":1,"team":"SF","player_clean_key":"receivera",
        "receiver_player_id":"00-TEST","pbp_actual_team":"SF",
        "pbp_actual_targets":7.0,"pbp_target_identity_resolved":True,
    }])
    rows=d.build_rows(
        points,opp,pbp_targets=pbp,require_full_weeks=False
    )
    r=rows.iloc[0]
    assert r.predicted_opportunities==pytest.approx(4.0)
    assert r.artifact_actual_opportunities==pytest.approx(0.0)
    assert r.actual_opportunities==pytest.approx(7.0)
    assert r.actual_opportunity_source=="COMPLETED_GAME_PBP_TARGETS"
    assert bool(r.actual_opportunity_discrepancy)
    assert r.actual_efficiency==pytest.approx(80.0/7.0)


def test_pbp_team_mismatch_excludes_entire_player_week():
    points=pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"HOU","player":"RB A",
            "player_clean_key":"rba","position_family":"RB","market":"rush_yards",
            "projection_mean":40.0,"actual":55.0,"actual_opportunities":10.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"HOU","player":"RB A",
            "player_clean_key":"rba","position_family":"RB","market":"rec_yards",
            "projection_mean":20.0,"actual":19.0,"actual_opportunities":3.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"HOU","player":"RB A",
            "player_clean_key":"rba","position_family":"RB","market":"receptions",
            "projection_mean":2.0,"actual":2.0,"actual_opportunities":3.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])
    opp=pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"HOU","player":"RB A",
            "player_clean_key":"rba","position_family":"RB","opportunity_type":"carries",
            "predicted_opportunities":9.0,"actual_opportunities":10.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"HOU","player":"RB A",
            "player_clean_key":"rba","position_family":"RB","opportunity_type":"targets",
            "predicted_opportunities":2.5,"actual_opportunities":3.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])
    pbp=pd.DataFrame([{
        "season":2026,"week":1,"player_clean_key":"rba",
        "receiver_player_id":"00-RB","pbp_actual_team":"DET",
        "pbp_actual_targets":3.0,"pbp_target_identity_resolved":True,
    }])
    rows=d.build_rows(
        points,opp,pbp_targets=pbp,require_full_weeks=False
    )
    assert len(rows)==3
    assert not rows.grading_identity_valid.any()
    assert set(rows.grading_exclusion_reason)=={"HISTORICAL_TEAM_IDENTITY_MISMATCH"}
    assert not rows.component_decomposition_eligible.any()


def test_gsis_roster_resolution_handles_two_way_receiver_and_same_name_collision():
    target_events=pd.DataFrame([
        {
            "week":1,"team":"JAX","receiver_player_id":"00-TH",
            "receiver_player_name":"T.Hunter",
        },
    ])
    roster=pd.DataFrame([
        {"week":1,"receiver_player_id":"00-TH","player_clean_key":"travishunter"},
        # Unrelated same-name person cannot contaminate the targeted GSIS ID.
        {"week":1,"receiver_player_id":"00-OTHER","player_clean_key":"travishunter"},
    ])
    out=d._resolve_pbp_target_identity_rows(target_events,roster)
    assert len(out)==1
    r=out.iloc[0]
    assert r.player_clean_key=="travishunter"
    assert r.receiver_player_id=="00-TH"
    assert r.pbp_actual_team=="JAX"
    assert r.pbp_actual_targets==pytest.approx(1.0)
    assert r.pbp_identity_route=="ROSTER_GSIS_ALIAS"


def test_multiweek_gsis_aliases_inherit_same_pbp_target_count():
    target_events=pd.DataFrame([
        {
            "week":1,"team":"IND","receiver_player_id":"00-X",
            "receiver_player_name":"P.X",
        },
        {
            "week":2,"team":"IND","receiver_player_id":"00-X",
            "receiver_player_name":"P.X",
        },
        {
            "week":2,"team":"IND","receiver_player_id":"00-X",
            "receiver_player_name":"P.X",
        },
    ])
    roster=pd.DataFrame([
        {"week":1,"receiver_player_id":"00-X","player_clean_key":"playerx"},
        {"week":2,"receiver_player_id":"00-X","player_clean_key":"playerxjr"},
    ])
    out=d._resolve_pbp_target_identity_rows(target_events,roster)
    w1=out.loc[out.week.eq(1)]
    assert len(w1)==1
    assert w1.iloc[0].player_clean_key=="playerx"
    assert w1.iloc[0].pbp_actual_targets==pytest.approx(1.0)

    w2=out.loc[out.week.eq(2)].sort_values("player_clean_key")
    assert set(w2.player_clean_key)=={"playerx","playerxjr"}
    assert set(w2.pbp_actual_targets)=={2.0}


def test_unique_pbp_name_fallback_only_when_roster_alias_missing():
    target_events=pd.DataFrame([
        {
            "week":1,"team":"LV","receiver_player_id":"00-MISSING",
            "receiver_player_name":"Receiver C",
        },
    ])
    roster=pd.DataFrame(columns=["week","receiver_player_id","player_clean_key"])
    out=d._resolve_pbp_target_identity_rows(target_events,roster)
    assert len(out)==1
    r=out.iloc[0]
    assert r.player_clean_key=="receiverc"
    assert r.pbp_identity_route=="PBP_UNIQUE_NAME_FALLBACK"


def test_unresolved_pbp_receiving_conflict_excludes_entire_player_week():
    points=pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G1","team":"SF","player":"WR X",
            "player_clean_key":"wrx","position_family":"WR","market":"rec_yards",
            "projection_mean":35.0,"actual":80.0,"actual_opportunities":0.0,
            "sportsbook_inputs_used_upstream":False,
        },
        {
            "season":2026,"week":1,"event_id":"G1","team":"SF","player":"WR X",
            "player_clean_key":"wrx","position_family":"WR","market":"receptions",
            "projection_mean":3.0,"actual":5.0,"actual_opportunities":0.0,
            "sportsbook_inputs_used_upstream":False,
        },
    ])
    opp=pd.DataFrame([{
        "season":2026,"week":1,"event_id":"G1","team":"SF","player":"WR X",
        "player_clean_key":"wrx","position_family":"WR","opportunity_type":"targets",
        "predicted_opportunities":5.0,"actual_opportunities":0.0,
        "sportsbook_inputs_used_upstream":False,
    }])
    pbp=pd.DataFrame(columns=[
        "season","week","player_clean_key","pbp_actual_targets","pbp_actual_team",
        "pbp_identity_route","pbp_target_identity_resolved",
    ])
    rows=d.build_rows(points,opp,pbp_targets=pbp,require_full_weeks=False)
    assert len(rows)==2
    assert not rows.grading_identity_valid.any()
    assert set(rows.grading_exclusion_reason)=={
        "GRADING_SOURCE_CONFLICT_UNRESOLVED_PBP_TARGET"
    }
    assert not rows.component_decomposition_eligible.any()
    assert rows.opportunity_oracle.isna().all()
    assert rows.efficiency_oracle.isna().all()
    assert rows.full_oracle.isna().all()

