from __future__ import annotations

import pandas as pd

from scripts.research.audit_player_opportunity_share_stability_v1 import (
    _cell_rows,
    _rho,
    build_panel,
)


def _universe():
    rows=[]
    # One RB has a stable early role and then drops to zero in target week.
    shares=[0.40,0.42,0.38,0.40,0.00]
    for week,share in enumerate(shares,start=1):
        rows.append({
            "season":2024,"week":week,"game_id":f"g{week}",
            "team":"IND","opponent":"HOU",
            "player_identity_key":"00-rb","player":"Test Back","position":"RB",
            "tgt_share_game":0.10,
            "rush_share_game":share,
        })
    # WR target-share sequence for independent cell semantics.
    wr=[0.20,0.30,0.25,0.35,0.10]
    for week,share in enumerate(wr,start=1):
        rows.append({
            "season":2024,"week":week,"game_id":f"g{week}",
            "team":"IND","opponent":"HOU",
            "player_identity_key":"00-wr","player":"Test Wideout","position":"WR",
            "tgt_share_game":share,
            "rush_share_game":0.0,
        })
    return pd.DataFrame(rows)


def test_cell_semantics_use_frozen_playerform_shares():
    x=_universe()
    rb=_cell_rows(x,"RB_CARRY_SHARE")
    wr=_cell_rows(x,"WR_TARGET_SHARE")
    assert rb["player_identity_key"].eq("00-rb").all()
    assert wr["player_identity_key"].eq("00-wr").all()
    assert rb.loc[rb["week"].eq(5),"share"].iloc[0] == 0.0
    assert wr.loc[wr["week"].eq(5),"share"].iloc[0] == 0.10


def test_build_panel_is_strictly_prior_and_keeps_zero_target_share():
    panel,integrity=build_panel(_universe())
    rb=panel.loc[
        panel["cell"].eq("RB_CARRY_SHARE")
        & panel["player_identity_key"].eq("00-rb")
        & panel["week"].eq(5)
    ].iloc[0]
    assert rb["prior_games"] == 4
    assert rb["prior_feature_max_week"] == 4
    assert abs(rb["prior_share_mean"] - 0.40) < 1e-12
    assert rb["actual_share"] == 0.0
    assert abs(rb["share_surprise"] - 0.40) < 1e-12
    assert integrity["same_future_feature_violations"] == 0
    assert integrity["same_team_feature_violations"] == 0


def test_share_volatility_expected_direction_helper():
    # Higher prior SD paired with larger surprise should be positive.
    x=pd.Series([0.01,0.02,0.05,0.10]).to_numpy(float)
    y=pd.Series([0.02,0.03,0.08,0.20]).to_numpy(float)
    assert _rho(x,y) > 0
