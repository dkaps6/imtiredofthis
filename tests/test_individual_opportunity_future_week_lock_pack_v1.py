from __future__ import annotations

import pytest

from scripts.research.certify_individual_opportunity_future_week_lock_pack_v1 import certify


def valid_components():
    player_state={
        "version":"PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1",
        "season":2026,"target_week":6,
        "disposition":"PLAYER_STATE_FUTURE_WEEK_READY",
        "target_week_roster_source_ready":True,
        "prior_week_source_completion":{"complete":True},
        "chronology_violations":0,
        "target_week_outcomes_read":0,
        "sportsbook_inputs_used":0,
        "candidate_models_fit":0,
        "production_changed":False,
        "skill_players":200,"scheduled_teams":28,"roster_source_week":6,
    }
    rb_carry={
        "version":"RB_PLAYER_STATE_ALLOCATION_SHADOW_FUTURE_WEEK_V1",
        "season":2026,"week":6,"status":"WEEK6_PREGAME_LOCK_FROZEN",
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,"parameters_fit":0,
        "production_changed":False,"scientific_formula_changed_from_week5":False,
        "max_team_control_sum_gap":0.0,"max_team_shadow_sum_gap":0.0,
        "locked_teams":28,"locked_players":90,"row_csv_digest":"sha256:a",
    }
    rb_receiving={
        "version":"RB_RECEIVING_ROOM_SHARE_SHADOW_FUTURE_WEEK_V1",
        "season":2026,"week":6,"status":"WEEK6_PREGAME_LOCK_FROZEN",
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,"parameters_fit":0,
        "production_changed":False,"scientific_formula_changed_from_week5":False,
        "max_candidate_conservation_gap":0.0,
        "teams":28,"players":90,"row_csv_sha256":"sha256:b",
    }
    wrte={
        "version":"PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_FUTURE_WEEK_V1",
        "season":2026,"week":6,"status":"WEEK6_PREGAME_LOCK_FROZEN",
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,"parameters_fit":0,
        "production_changed":False,"scientific_formula_changed_from_week5":False,
        "same_or_future_feature_violations":0,
        "max_room_pool_gap":0.0,"max_team_modeled_target_mass_gap":0.0,
        "locked_players":220,"locked_wr":150,"locked_te":70,
        "trajectory_available_players":180,"row_csv_digest":"sha256:c",
    }
    share_volatility={
        "version":"PLAYER_OPPORTUNITY_SHARE_VOLATILITY_FUTURE_WEEK_LOCK_V1",
        "season":2026,"target_week":6,
        "status":"WEEK6_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN",
        "players":210,
        "rb_carry_supported":45,"rb_target_supported":45,
        "wr_target_supported":95,"te_target_supported":42,
        "chronology_violations":0,
        "target_week_outcomes_read":0,"sportsbook_inputs_used":0,
        "oddsapi_calls":0,"fitted_coefficients":0,"thresholds_applied":0,
        "projection_changes":0,"unsupported_rows_imputed":0,
        "production_changed":False,
    }
    return player_state,rb_carry,rb_receiving,wrte,share_volatility


def test_certify_valid_future_week_pack():
    ps,rc,rr,wt,sv=valid_components()
    got=certify(
        season=2026,week=6,
        player_state=ps,rb_carry=rc,rb_receiving=rr,wrte_trajectory=wt,share_volatility=sv,
    )
    assert got["status"]=="WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_PACK_FROZEN"
    assert got["components"]["rb_carry_snap"]["locked_players"]==90
    assert got["components"]["wrte_trajectory"]["locked_players"]==220
    assert got["components"]["share_volatility"]["players"]==210


def test_certify_rejects_prior_week_roster_fallback():
    ps,rc,rr,wt,sv=valid_components()
    ps["target_week_roster_source_ready"]=False
    with pytest.raises(RuntimeError,match="exact target-week roster"):
        certify(
            season=2026,week=6,
            player_state=ps,rb_carry=rc,rb_receiving=rr,wrte_trajectory=wt,share_volatility=sv,
    )


def test_certify_rejects_formula_drift():
    ps,rc,rr,wt,sv=valid_components()
    rr["scientific_formula_changed_from_week5"]=True
    with pytest.raises(RuntimeError,match="formula changed"):
        certify(
            season=2026,week=6,
            player_state=ps,rb_carry=rc,rb_receiving=rr,wrte_trajectory=wt,share_volatility=sv,
    )


def test_certify_rejects_target_week_outcome_contamination():
    ps,rc,rr,wt,sv=valid_components()
    wt["target_week_outcomes_read"]=1
    with pytest.raises(RuntimeError,match="contamination"):
        certify(
            season=2026,week=6,
            player_state=ps,rb_carry=rc,rb_receiving=rr,wrte_trajectory=wt,share_volatility=sv,
    )


def test_certify_rejects_share_volatility_projection_change():
    ps,rc,rr,wt,sv=valid_components()
    sv["projection_changes"]=1
    with pytest.raises(RuntimeError,match="share_volatility contamination"):
        certify(
            season=2026,week=6,
            player_state=ps,rb_carry=rc,rb_receiving=rr,
            wrte_trajectory=wt,share_volatility=sv,
        )
