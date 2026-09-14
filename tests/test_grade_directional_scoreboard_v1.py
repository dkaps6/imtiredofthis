from __future__ import annotations

import pandas as pd

from scripts.backtest.grade_directional_scoreboard_v1 import score, summarize


def _proj() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "team": "KC", "opponent": "DEN",
         "player": "Patrick Mahomes", "player_clean_key": "mahomes", "market": "pass_yards",
         "ensemble_proj": 270.0, "actual": 280.0, "qb_m89_synthesis_applied": 1},
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "team": "KC", "opponent": "DEN",
         "player": "Travis Kelce", "player_clean_key": "kelce", "market": "rec_yards",
         "ensemble_proj": 55.0, "actual": 40.0, "wrte_authorized_treatment": True, "wrte_route": "TE_R5P_UPSTREAM_MC"},
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "team": "KC", "opponent": "DEN",
         "player": "Nobody", "player_clean_key": "nobody", "market": "rush_yards",
         "ensemble_proj": 50.0, "actual": 50.0},
    ])


def _props() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "player_clean_key": "mahomes", "market": "pass_yards",
         "book": "draftkings", "line": 260.5, "over_odds": -110, "under_odds": -110, "player": "Patrick Mahomes"},
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "player_clean_key": "kelce", "market": "rec_yards",
         "book": "draftkings", "line": 60.5, "over_odds": -110, "under_odds": -110, "player": "Travis Kelce"},
        {"season": 2024, "week": 1, "game_id": "2024_01_KC_DEN", "player_clean_key": "nobody", "market": "rush_yards",
         "book": "draftkings", "line": 50.0, "over_odds": -110, "under_odds": -110, "player": "Nobody"},
    ])


def test_over_pick_wins_when_actual_exceeds_line():
    detail = score(_proj(), _props())
    row = detail.loc[detail.player.eq("Patrick Mahomes")].iloc[0]
    assert row["directional_pick"] == "OVER"
    assert row["win_loss_push"] == "WIN"
    assert row["promoted_model_version"] == "QB_M89_SYNTHESIS_V1"


def test_under_pick_wins_when_actual_below_line():
    detail = score(_proj(), _props())
    row = detail.loc[detail.player.eq("Travis Kelce")].iloc[0]
    assert row["directional_pick"] == "UNDER"
    assert row["win_loss_push"] == "WIN"
    assert row["promoted_model_version"] == "TE_R5P_UPSTREAM_MC"


def test_equal_projection_and_line_is_no_bet():
    detail = score(_proj(), _props())
    row = detail.loc[detail.player.eq("Nobody")].iloc[0]
    assert row["directional_pick"] == "NO_BET"
    assert row["win_loss_push"] == "NO_BET"
    assert row["promoted_model_version"] == "BASE_ENSEMBLE"


def test_summary_excludes_no_bet_from_win_rate():
    detail = score(_proj(), _props())
    summary = summarize(detail)
    all_markets = summary.loc[summary.market.eq("ALL_MARKETS")].iloc[0]
    assert all_markets["decided_bets"] == 2
    assert all_markets["no_bets"] == 1
    assert all_markets["win_rate"] == 1.0
