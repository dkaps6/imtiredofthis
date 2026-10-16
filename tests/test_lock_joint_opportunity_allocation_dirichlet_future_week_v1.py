from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.lock_joint_opportunity_allocation_dirichlet_future_week_v1 as M


def _state(ready=True):
    return {
        "version":"PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1",
        "season":2026,"target_week":6,
        "disposition":"PLAYER_STATE_FUTURE_WEEK_READY" if ready else "PLAYER_STATE_FUTURE_WEEK_NOT_READY",
        "target_week_roster_source_ready":bool(ready),
        "roster_source_week":6 if ready else 5,
        "prior_week_source_completion":{"complete":True},
        "chronology_violations":0,
    }


def _vol_result():
    return {
        "version":"PLAYER_OPPORTUNITY_SHARE_VOLATILITY_FUTURE_WEEK_LOCK_V1",
        "season":2026,"target_week":6,
        "status":"WEEK6_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN",
        "chronology_violations":0,
        "target_week_outcomes_read":0,
        "allocation_uncertainty_candidate_cells":["RB_CARRY_SHARE","WR_TARGET_SHARE"],
        "rb_target_betabinomial_authorized":False,
        "te_target_betabinomial_authorized":False,
    }


def _vol_rows():
    return pd.DataFrame([
        {
            "season":2026,"target_week":6,"team":"IND","player":"Back A","position":"RB",
            "player_identity_key":"gsis:rb-a",
            "prior_target_share_mean":0.10,"prior_target_excess_share_var":0.001,
            "target_share_source_max_week":5,"wr_target_betabinomial_eligible":False,
            "prior_rush_share_mean":0.40,"prior_rush_excess_share_var":0.030,
            "rush_share_source_max_week":5,"rb_carry_betabinomial_eligible":True,
        },
        {
            "season":2026,"target_week":6,"team":"IND","player":"Wide A","position":"WR",
            "player_identity_key":"gsis:wr-a",
            "prior_target_share_mean":0.30,"prior_target_excess_share_var":0.020,
            "target_share_source_max_week":5,"wr_target_betabinomial_eligible":True,
            "prior_rush_share_mean":np.nan,"prior_rush_excess_share_var":np.nan,
            "rush_share_source_max_week":np.nan,"rb_carry_betabinomial_eligible":False,
        },
    ])


def test_validate_authorities_rejects_stale_roster():
    with pytest.raises(RuntimeError,match="not READY|roster"):
        M.validate_authorities(
            season=2026,week=6,player_state=_state(False),volatility_result=_vol_result()
        )


def test_shadow_lock_preserves_canonical_means(monkeypatch):
    dummy_metrics=pd.DataFrame([{"team":"IND","player":"x"}])

    monkeypatch.setattr(M,"build_week_metrics",lambda **kwargs: dummy_metrics)

    def fake_probs(metrics,cell):
        if cell=="RB_CARRY_SHARE":
            return pd.DataFrame([
                {"team":"IND","player_clean_key":"backa","position":"RB","q":0.40},
                {"team":"IND","player_clean_key":"other","position":"RB","q":0.30},
                {"team":"IND","player_clean_key":"__RESIDUAL__","position":"RESIDUAL","q":0.30},
            ])
        return pd.DataFrame([
            {"team":"IND","player_clean_key":"widea","position":"WR","q":0.30},
            {"team":"IND","player_clean_key":"tighta","position":"TE","q":0.20},
            {"team":"IND","player_clean_key":"__RESIDUAL__","position":"RESIDUAL","q":0.50},
        ])

    monkeypatch.setattr(M,"canonical_player_probabilities",fake_probs)

    rooms,players,result=M.lock_shadow(
        season=2026,week=6,
        volatility_rows=_vol_rows(),
        volatility_result=_vol_result(),
        player_state=_state(True),
        team_weekly=pd.DataFrame(),
        player_logs=pd.DataFrame(),
        schedule=pd.DataFrame(),
        universe=pd.DataFrame(),
    )
    assert result["status"]=="WEEK6_JOINT_ALLOCATION_DIRICHLET_SHADOW_FROZEN"
    assert result["rooms"]==2
    assert result["rb_carry_rooms"]==1
    assert result["wr_target_rooms"]==1
    assert result["max_expected_share_gap"]<=1e-12
    assert result["chronology_violations"]==0
    assert set(players["cell"])=={"RB_CARRY_SHARE","WR_TARGET_SHARE"}
    assert (rooms["probability_mass_gap"]<=1e-12).all()


def test_shadow_excludes_unauthorized_or_zero_variance_player(monkeypatch):
    rows=_vol_rows()
    rows.loc[rows["position"].eq("WR"),"prior_target_excess_share_var"]=0.0

    monkeypatch.setattr(
        M,"build_week_metrics",
        lambda **kwargs: pd.DataFrame([{"team":"IND","player":"x"}]),
    )
    monkeypatch.setattr(
        M,"canonical_player_probabilities",
        lambda metrics,cell: pd.DataFrame([
            {"team":"IND","player_clean_key":"backa" if cell=="RB_CARRY_SHARE" else "widea",
             "position":"RB" if cell=="RB_CARRY_SHARE" else "WR","q":0.4},
            {"team":"IND","player_clean_key":"__RESIDUAL__","position":"RESIDUAL","q":0.6},
        ]),
    )

    rooms,players,result=M.lock_shadow(
        season=2026,week=6,
        volatility_rows=rows,
        volatility_result=_vol_result(),
        player_state=_state(True),
        team_weekly=pd.DataFrame(),player_logs=pd.DataFrame(),
        schedule=pd.DataFrame(),universe=pd.DataFrame(),
    )
    assert result["rb_carry_rooms"]==1
    assert result["wr_target_rooms"]==0
    assert set(players["cell"])=={"RB_CARRY_SHARE"}
