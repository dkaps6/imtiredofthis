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


def test_receiving_identity_scope_drops_defensive_same_name_collision():
    frame=pd.DataFrame([
        {"player_name":"Byron Young","position":"WR","player_id":"00-WR"},
        {"player_name":"Byron Young","position":"DE","player_id":"00-DE"},
        {"player_name":"Marcus Harris","position":"DT","player_id":"00-DT"},
        {"player_name":"Marcus Harris","position":"TE","player_id":"00-TE"},
    ])
    out=d._restrict_receiving_identity_population(frame)
    assert set(out.position)=={"WR","TE"}
    assert set(out.player_id)=={"00-WR","00-TE"}


def test_pbp_observed_two_way_player_survives_receiving_identity_scope():
    weekly=pd.DataFrame([
        {"week":1,"player_name":"Travis Hunter","position":"CB","receiver_player_id":"00-TH"},
        {"week":1,"player_name":"Byron Young","position":"DE","receiver_player_id":"00-BY"},
        {"week":1,"player_name":"Receiver C","position":"WR","receiver_player_id":"00-WR"},
    ])
    counts=pd.DataFrame([
        {"week":1,"receiver_player_id":"00-TH","pbp_actual_targets":1.0,"pbp_actual_team":"JAX"},
        {"week":1,"receiver_player_id":"00-WR","pbp_actual_targets":4.0,"pbp_actual_team":"IND"},
    ])
    out=d._receiver_identity_candidates(weekly,counts)
    assert set(out.receiver_player_id)=={"00-TH","00-WR"}


def test_multiweek_gsis_alias_expansion_is_intentional_not_ambiguous():
    counts=pd.DataFrame([
        {
            "week":1,"receiver_player_id":"00-X","pbp_actual_targets":4.0,
            "pbp_actual_team":"IND","pbp_receiver_name_key":"playerx",
        },
        {
            "week":2,"receiver_player_id":"00-X","pbp_actual_targets":7.0,
            "pbp_actual_team":"IND","pbp_receiver_name_key":"playerx",
        },
    ])
    aliases=pd.DataFrame([
        {"receiver_player_id":"00-X","roster_player_clean_key":"playerx"},
        {"receiver_player_id":"00-X","roster_player_clean_key":"playerxjr"},
    ])
    out=d._expand_counts_across_roster_aliases(counts,aliases)
    assert len(out)==4
    assert not out.duplicated(
        ["week","receiver_player_id","roster_player_clean_key"]
    ).any()
    for alias in ("playerx","playerxjr"):
        q=out.loc[out.roster_player_clean_key.eq(alias)].sort_values("week")
        assert q.pbp_actual_targets.tolist()==[4.0,7.0]

