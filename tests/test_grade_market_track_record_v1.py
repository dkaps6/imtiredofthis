"""Tests for the forward market track-record grader."""
from __future__ import annotations

import pandas as pd

from scripts.operations.grade_market_track_record_v1 import (
    american_profit,
    edge_bucket,
    grade_matched_rows,
    match_bets_to_actuals,
    model_side,
    outcome_side,
    select_model_bet,
)


def test_model_side_and_outcome_side():
    assert model_side(270.0, 265.5) == "OVER"
    assert model_side(260.0, 265.5) == "UNDER"
    assert outcome_side(280.0, 265.5) == "OVER"
    assert outcome_side(250.0, 265.5) == "UNDER"
    assert outcome_side(265.5, 265.5) == "PUSH"


def test_american_profit_positive_and_negative_odds():
    assert abs(american_profit(150) - 1.5) < 1e-9
    assert abs(american_profit(-110) - (100.0 / 110.0)) < 1e-9


def test_edge_bucket_uses_fractional_probability_units():
    assert edge_bucket(-0.01) == "<=0"
    assert edge_bucket(0.01) == "0-2"
    assert edge_bucket(0.03) == "2-5"
    assert edge_bucket(0.075) == "5-10"
    assert edge_bucket(0.15) == "10-20"
    assert edge_bucket(0.25) == "20+"


def _quote_rows(
    *,
    book="bookA",
    line=265.5,
    proj=270.0,
    p_over=0.60,
    p_under=0.40,
    over_odds=-110.0,
    under_odds=-110.0,
    event_id="e1",
    player="Patrick Mahomes",
    key="patrickmahomes",
    market="pass_yards",
    source_market="player_pass_yds",
):
    common = {
        "season": 2026,
        "week": 1,
        "event_id": event_id,
        "player": player,
        "player_clean_key": key,
        "team": "KC",
        "opponent": "BUF",
        "market": market,
        "source_market": source_market,
        "book": book,
        "book_title": book,
        "vegas_line": line,
        "model_proj": proj,
        "vegas_over_odds": over_odds,
        "vegas_under_odds": under_odds,
    }
    return [
        {
            **common,
            "side": "OVER",
            "fair_prob": p_over,
            "vegas_odds": over_odds,
            "market_prob": 0.5,
            "edge_pct": p_over - 0.5,
        },
        {
            **common,
            "side": "UNDER",
            "fair_prob": p_under,
            "vegas_odds": under_odds,
            "market_prob": 0.5,
            "edge_pct": p_under - 0.5,
        },
    ]


def _board():
    return pd.DataFrame(_quote_rows())


def test_select_model_bet_uses_deployed_ev_side_not_mean_relative_side():
    # Projection is above the line, but the simulated distribution makes UNDER
    # the higher-EV side. Production chooses UNDER; the grader must do the same.
    board = pd.DataFrame(
        _quote_rows(proj=270.0, line=265.5, p_over=0.45, p_under=0.60)
    )
    picked = select_model_bet(board)
    assert len(picked) == 1
    assert picked.iloc[0]["side"] == "UNDER"
    assert picked.iloc[0]["model_pick_side"] == "UNDER"
    assert float(picked.iloc[0]["production_best_ev"]) > 0


def test_select_model_bet_passes_nonpositive_best_ev():
    board = pd.DataFrame(
        _quote_rows(p_over=0.50, p_under=0.50, over_odds=-110, under_odds=-110)
    )
    assert select_model_bet(board).empty


def test_select_model_bet_chooses_best_snapshot_offer_across_books():
    board = pd.DataFrame(
        _quote_rows(book="bookA", p_over=0.56, p_under=0.44)
        + _quote_rows(book="bookB", p_over=0.62, p_under=0.38)
    )
    got = select_model_bet(board)
    assert len(got) == 1
    assert got.iloc[0]["book"] == "bookB"
    assert got.iloc[0]["side"] == "OVER"


def test_select_model_bet_is_row_order_invariant_when_best_offer_is_unique():
    board = pd.DataFrame(
        _quote_rows(book="bookA", p_over=0.56, p_under=0.44)
        + _quote_rows(book="bookB", p_over=0.62, p_under=0.38)
    )
    picks = []
    for order in (
        list(range(len(board))),
        list(reversed(range(len(board)))),
        [2, 0, 3, 1],
    ):
        got = select_model_bet(board.iloc[order].reset_index(drop=True))
        picks.append(
            (
                got.iloc[0]["book"],
                got.iloc[0]["side"],
                float(got.iloc[0]["vegas_line"]),
                float(got.iloc[0]["vegas_odds"]),
            )
        )
    assert len(set(picks)) == 1


def test_material_exact_ev_cross_book_tie_fails_closed():
    # Both offers have the same best EV but imply different real wagers.
    a = _quote_rows(
        book="bookA", line=49.5, proj=50.0,
        p_over=0.60, p_under=0.40, over_odds=-110, under_odds=-110,
        market="rec_yards", source_market="player_reception_yds",
    )
    b = _quote_rows(
        book="bookB", line=50.5, proj=50.0,
        p_over=0.40, p_under=0.60, over_odds=-110, under_odds=-110,
        market="rec_yards", source_market="player_reception_yds",
    )
    assert select_model_bet(pd.DataFrame(a + b)).empty


def test_identical_wager_ev_tie_uses_row_level_book_title_fallback():
    a = _quote_rows(book="", line=49.5, p_over=0.60, p_under=0.40)
    for row in a:
        row["book_title"] = "Alpha Sports"
    b = _quote_rows(book="bookB", line=49.5, p_over=0.60, p_under=0.40)
    for row in b:
        row["book_title"] = "Beta Sports"
    got = select_model_bet(pd.DataFrame(a + b))
    assert len(got) == 1
    assert got.iloc[0]["book_title"] == "Alpha Sports"


def test_match_bets_to_actuals_and_grade_matched_rows_win_case():
    bets = select_model_bet(_board())
    actual = pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "team": "KC",
            "player_clean_key": "patrickmahomes",
            "passing_yards": 300,
        }]
    )
    detail = match_bets_to_actuals(bets, actual)
    graded, summary = grade_matched_rows(detail, season=2026, present_weeks=[1])

    assert summary["status"] == "graded"
    assert summary["decided_bets"] == 1
    assert summary["wins"] == 1
    row = graded.iloc[0]
    assert row["bet_result"] == "WIN"
    assert abs(row["unit_result"] - (100.0 / 110.0)) < 1e-9


def test_match_bets_to_actuals_and_grade_matched_rows_loss_case():
    bets = select_model_bet(_board())
    actual = pd.DataFrame(
        [{
            "season": 2026,
            "week": 1,
            "team": "KC",
            "player_clean_key": "patrickmahomes",
            "passing_yards": 200,
        }]
    )
    detail = match_bets_to_actuals(bets, actual)
    graded, summary = grade_matched_rows(detail, season=2026, present_weeks=[1])
    assert summary["wins"] == 0
    assert summary["losses"] == 1
    assert graded.iloc[0]["unit_result"] == -1.0
