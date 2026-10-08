from __future__ import annotations

import pandas as pd

import scripts.research.run_player_landscape_transmission_audit_v1 as audit


def test_matrix_uses_only_frozen_status_vocabulary():
    inv=pd.DataFrame(audit.feature_specs())
    assert len(inv)>30
    assert set(inv.production_consumption_status).issubset(audit.VALID_STATUS)


def test_matrix_covers_all_required_positions_and_markets():
    matrix=audit._expand_matrix(audit.feature_specs())
    expected={(p,m) for p,markets in audit.MARKETS.items() for m in markets}
    got=set(zip(matrix.position_family,matrix.market))
    assert expected==got
    assert set(matrix.landscape_layer).issubset(set(audit.LAYERS))


def test_known_matchup_transmission_gaps_remain_visible():
    matrix=audit._expand_matrix(audit.feature_specs())
    q=matrix.set_index(["position_family","feature"])
    assert q.loc[("RB","def_rush_epa")].iloc[0].production_consumption_status=="AVAILABLE_BUT_NOT_CONSUMED"
    assert q.loc[("RB","dl_ybc_stuff_rate")].iloc[0].production_consumption_status=="AVAILABLE_BUT_DROPPED"
    assert q.loc[("WR","position_specific_ypt_allowed")].iloc[0].production_consumption_status=="AVAILABLE_BUT_DROPPED"
    assert q.loc[("TE","position_specific_ypt_allowed")].iloc[0].production_consumption_status=="AVAILABLE_BUT_DROPPED"


def test_failed_fmt_candidates_are_not_reopened():
    matrix=audit._expand_matrix(audit.feature_specs())
    for feature in (
        "fmt_rb_pass_rate_faced_candidate",
        "fmt_wr_true_proe_candidate",
        "fmt_te_pass_success_candidate",
    ):
        q=matrix.loc[matrix.feature.eq(feature)]
        assert not q.empty
        assert q.production_consumption_status.eq("TESTED_AND_CLOSED").all()
        assert q.historical_validation_status.eq("HISTORICAL_INTEGRATION_FAIL_CLOSED").all()


def test_player_level_prospective_shadows_are_preserved_not_promoted():
    matrix=audit._expand_matrix(audit.feature_specs())
    rb=matrix.loc[matrix.feature.eq("rb_prior_receiving_room_share")]
    assert not rb.empty
    assert rb.production_consumption_status.eq("PROSPECTIVE_ONLY_FROZEN").all()
    traj=matrix.loc[matrix.feature.eq("wr_te_target_share_trajectory")]
    assert not traj.empty
    assert traj.production_consumption_status.eq("PROSPECTIVE_ONLY_FROZEN").all()


def test_static_summary_distinguishes_usage_efficiency_and_opponent_consumption():
    matrix=audit._expand_matrix(audit.feature_specs())
    s=audit._position_market_summary(matrix)
    assert not s.empty
    # Every required player market has some individualized usage authority.
    assert s.individualized_usage.all()
    # Required yardage/count markets have at least one player efficiency seam.
    assert s.individualized_efficiency.all()
    # Opponent context is materially consumed somewhere for every required market,
    # while the detailed matrix still preserves missing/dropped matchup features.
    assert s.opponent_materially_consumed.all()
    assert (s.gap_feature_rows>0).all()

def test_route_state_remains_source_parity_blocked():
    matrix=audit._expand_matrix(audit.feature_specs())
    for feature in ("route_rate","yprr"):
        q=matrix.loc[matrix.feature.eq(feature)]
        assert not q.empty
        assert q.production_consumption_status.eq("SOURCE_PARITY_BLOCKED").all()
        assert q.historical_validation_status.eq("HISTORICAL_WEEKLY_ROUTE_PARITY_NOT_CLEARED").all()

