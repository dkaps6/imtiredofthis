from __future__ import annotations

import pandas as pd

from scripts.research.score_rb_individual_run_location_matchup_predictive_v1 import (
    MULT_CEIL,
    MULT_FLOOR,
    attach_multiplier,
    build_league_location_ypc,
    prepare_source,
)


def _rows():
    rows=[]
    for season in (2024,2025):
        for week in range(2,19):
            # Two unique opponent defense states repeated across players.
            for team,opp,player,shares,defypc in [
                ("IND","HOU","Back A",(0.50,0.30,0.20),(5.0,4.0,3.0)),
                ("TEN","JAX","Back B",(0.20,0.50,0.30),(3.0,4.0,5.0)),
            ]:
                rows.append({
                    "season":season,"week":week,"team":team,"opponent":opp,
                    "player":player,"gsis_id":"00-test-"+player[-1],
                    "matchup_state_ready":True,"defense_state_ready":True,
                    "player_share_left":shares[0],
                    "player_share_middle":shares[1],
                    "player_share_right":shares[2],
                    "def_rushes_left":20,"def_rushes_middle":20,"def_rushes_right":20,
                    "def_ypc_left":defypc[0],"def_ypc_middle":defypc[1],"def_ypc_right":defypc[2],
                })
    return pd.DataFrame(rows)


def test_prepare_source_and_league_location_baseline():
    src=prepare_source(_rows())
    league=build_league_location_ypc(src)
    assert len(league)==34
    row=league.loc[(league["season"].eq(2024))&(league["week"].eq(2))].iloc[0]
    assert abs(row["league_ypc_left"]-4.0)<1e-12
    assert abs(row["league_ypc_middle"]-4.0)<1e-12
    assert abs(row["league_ypc_right"]-4.0)<1e-12


def test_attach_multiplier_is_player_specific_and_bounded():
    src=attach_multiplier(prepare_source(_rows()))
    a=src.loc[(src["season"].eq(2024))&(src["week"].eq(2))&(src["player"].eq("Back A"))].iloc[0]
    # Player A mix: .5*5 + .3*4 + .2*3 = 4.3 vs league mix 4.0.
    assert abs(a["opp_mix_ypc"]-4.3)<1e-12
    assert abs(a["league_mix_ypc"]-4.0)<1e-12
    assert abs(a["run_location_matchup_mult"]-1.075)<1e-12
    assert MULT_FLOOR <= a["run_location_matchup_mult"] <= MULT_CEIL
    assert bool(a["source_ready_for_candidate"]) is True


def test_unsupported_player_gets_neutral_multiplier():
    raw=_rows()
    raw.loc[0,"matchup_state_ready"]=False
    src=attach_multiplier(prepare_source(raw))
    a=src.iloc[0]
    assert a["run_location_matchup_mult"]==1.0
    assert bool(a["source_ready_for_candidate"]) is False
