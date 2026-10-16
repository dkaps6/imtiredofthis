from __future__ import annotations

import pandas as pd

from scripts.research.score_player_role_matchup_depth_predictive_v1 import (
    attach_multiplier,
    build_league_band_ypt,
    candidate_invariants,
    prepare_source,
)


def _source_rows():
    rows=[]
    for defense,targets,ypt in [
        ("BAL",10.0,5.0),
        ("PIT",30.0,9.0),
    ]:
        row={
            "season":2024,"week":2,"team":"IND" if defense=="BAL" else "CLE",
            "opponent":defense,"position":"WR","player":"Test Player "+defense,
            "gsis_id":"00-"+defense,"matchup_state_ready":True,
            "player_share_behind":0.10,"player_share_short":0.40,
            "player_share_intermediate":0.30,"player_share_deep":0.20,
        }
        for band in ("behind","short","intermediate","deep"):
            row[f"def_targets_{band}"]=targets
            row[f"def_ypt_{band}"]=ypt
        rows.append(row)
    # Fill remaining week-cells so the frozen helper sees the exact 34 cells.
    for season in (2024,2025):
        for week in range(2,19):
            if season==2024 and week==2:
                continue
            row={
                "season":season,"week":week,"team":"IND","opponent":"HOU",
                "position":"TE","player":f"TE {season} {week}","gsis_id":f"{season}-{week}",
                "matchup_state_ready":True,
                "player_share_behind":0.10,"player_share_short":0.40,
                "player_share_intermediate":0.30,"player_share_deep":0.20,
            }
            for band in ("behind","short","intermediate","deep"):
                row[f"def_targets_{band}"]=20.0
                row[f"def_ypt_{band}"]=7.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_league_depth_ypt_is_target_weighted_by_defense_band():
    source=prepare_source(_source_rows())
    league=build_league_band_ypt(source)
    row=league.loc[(league.season.eq(2024))&(league.week.eq(2))].iloc[0]
    # (10*5 + 30*9) / 40 = 8
    for band in ("behind","short","intermediate","deep"):
        assert abs(float(row[f"league_ypt_{band}"])-8.0)<1e-12


def test_player_specific_multiplier_uses_player_mix_and_frozen_bounds():
    source=prepare_source(_source_rows())
    got=attach_multiplier(source)
    bal=got.loc[(got.season.eq(2024))&(got.week.eq(2))&(got.opponent.eq("BAL"))].iloc[0]
    pit=got.loc[(got.season.eq(2024))&(got.week.eq(2))&(got.opponent.eq("PIT"))].iloc[0]
    assert abs(float(bal.depth_matchup_mult)-(5.0/8.0))<1e-12
    assert abs(float(pit.depth_matchup_mult)-(9.0/8.0))<1e-12
    assert bool(bal.source_ready_for_candidate) is True


def test_multiplier_clips_to_predeclared_fmt_safety_bounds():
    x=_source_rows()
    mask=(x["season"].eq(2024))&(x["week"].eq(2))&(x["opponent"].eq("BAL"))
    for band in ("behind","short","intermediate","deep"):
        x.loc[mask,f"def_ypt_{band}"]=0.1
    got=attach_multiplier(prepare_source(x))
    bal=got.loc[(got.season.eq(2024))&(got.week.eq(2))&(got.opponent.eq("BAL"))].iloc[0]
    assert float(bal.depth_matchup_mult)>=0.50


def test_candidate_invariants_detect_only_treated_rec_yards_change():
    base=pd.DataFrame([
        {"season":2024,"week":2,"team":"IND","opponent":"HOU","player_clean_key":"a","position":"WR","market":"rec_yards","ensemble_proj":50.0},
        {"season":2024,"week":2,"team":"IND","opponent":"HOU","player_clean_key":"a","position":"WR","market":"receptions","ensemble_proj":5.0},
        {"season":2024,"week":2,"team":"IND","opponent":"HOU","player_clean_key":"b","position":"TE","market":"rec_yards","ensemble_proj":30.0},
    ])
    cand=base.copy()
    cand.loc[(cand.position.eq("WR"))&(cand.market.eq("rec_yards")),"ensemble_proj"]=52.0
    got=candidate_invariants(base,cand,"WR")
    assert got["max_untreated_projection_delta"]==0.0
    assert got["max_receptions_projection_delta"]==0.0
