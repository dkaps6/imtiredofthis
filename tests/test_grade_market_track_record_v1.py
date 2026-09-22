"""Tests for the forward market track record grader.

Sportsbook lines are graded exactly like the model; they are a benchmark
only. These tests exercise the pure matching/arithmetic core
(no network, no file I/O) so the CLV/ROI machinery is verified without
depending on nflreadpy or the real archived ledger.
"""
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


def test_edge_bucket_boundaries():
    assert edge_bucket(1.0) == "0-2"
    assert edge_bucket(3.0) == "2-5"
    assert edge_bucket(15.0) == "10-20"
    assert edge_bucket(25.0) == "20+"


def _board():
    return pd.DataFrame(
        [
            {
                "season": 2026, "week": 1, "event_id": "e1", "player": "p.mahomes",
                "market": "pass_yards", "team": "KC", "player_clean_key": "patrickmahomes",
                "vegas_line": 265.5, "model_proj": 270.0, "side": "OVER", "vegas_odds": -110,
            },
            {
                "season": 2026, "week": 1, "event_id": "e1", "player": "p.mahomes",
                "market": "pass_yards", "team": "KC", "player_clean_key": "patrickmahomes",
                "vegas_line": 265.5, "model_proj": 270.0, "side": "UNDER", "vegas_odds": -110,
            },
        ]
    )


def test_select_model_bet_keeps_only_models_own_side():
    picked = select_model_bet(_board())
    assert len(picked) == 1
    assert picked.iloc[0]["side"] == "OVER"


def test_match_bets_to_actuals_and_grade_matched_rows_win_case():
    bets = select_model_bet(_board())
    actual = pd.DataFrame(
        [{"season": 2026, "week": 1, "team": "KC", "player_clean_key": "patrickmahomes", "passing_yards": 300}]
    )

    detail = match_bets_to_actuals(bets, actual)
    graded, summary = grade_matched_rows(detail, season=2026, present_weeks=[1])

    assert summary["status"] == "graded"
    assert summary["decided_bets"] == 1
    assert summary["wins"] == 1
    row = graded.iloc[0]
    assert row["bet_result"] == "WIN"
    assert abs(row["unit_result"] - (100.0 / 110.0)) < 1e-9
    assert abs(row["model_error"] - (270.0 - 300.0)) < 1e-9


def test_match_bets_to_actuals_and_grade_matched_rows_loss_case():
    bets = select_model_bet(_board())
    actual = pd.DataFrame(
        [{"season": 2026, "week": 1, "team": "KC", "player_clean_key": "patrickmahomes", "passing_yards": 200}]
    )

    detail = match_bets_to_actuals(bets, actual)
    graded, summary = grade_matched_rows(detail, season=2026, present_weeks=[1])

    assert summary["wins"] == 0
    assert summary["losses"] == 1
    assert graded.iloc[0]["unit_result"] == -1.0


def test_grade_matched_rows_reports_status_when_nothing_matches():
    bets = select_model_bet(_board())
    actual = pd.DataFrame(
        [{"season": 2026, "week": 1, "team": "BUF", "player_clean_key": "someoneelse", "passing_yards": 200}]
    )
    detail = match_bets_to_actuals(bets, actual)
    graded, summary = grade_matched_rows(detail, season=2026, present_weeks=[1])
    assert summary["status"] == "matched_zero_rows_to_actual_results"
    assert graded.empty


def _multi_book_board():
    """Two books quoting different lines for one player-market, with the
    model's projection sitting between them -- the real 2026 Week 2 Jacoby
    Brissett shape, where the surviving row decides the side."""
    rows = []
    for book, line in (("bookA", 216.5), ("bookB", 221.5)):
        for side in ("OVER", "UNDER"):
            rows.append({
                "season": 2026, "week": 2, "event_id": "evt1", "book": book,
                "player": "Jacoby Brissett", "player_clean_key": "jacobybrissett",
                "team": "ARI", "market": "pass_yards", "side": side,
                "vegas_line": line, "vegas_odds": -114.0, "model_proj": 219.805606,
            })
    return pd.DataFrame(rows)


def test_select_model_bet_is_independent_of_row_order():
    board = _multi_book_board()
    picks = []
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        got = select_model_bet(board.iloc[order].reset_index(drop=True))
        assert len(got) == 1
        picks.append((got.iloc[0]["side"], float(got.iloc[0]["vegas_line"])))
    assert len(set(picks)) == 1, f"row order changed the graded bet: {picks}"


def test_select_model_bet_uses_the_consensus_line_not_an_arbitrary_book():
    # Median of {216.5, 221.5} is 219.0; the projection 219.81 is above it,
    # so the model is on OVER regardless of which book sorted last.
    got = select_model_bet(_multi_book_board())
    assert got.iloc[0]["side"] == "OVER"



def _straddle_board():
    """Real Week-1 straddle shape: the projection is above the lower quote
    but below the consensus, so an UNDER must grade at the higher compatible
    captured line rather than the contradictory lower one."""
    rows = []
    for book, line in (("bookA", 49.5), ("bookB", 52.5)):
        for side in ("OVER", "UNDER"):
            rows.append({
                "season": 2026, "week": 1, "event_id": "evt2", "book": book,
                "player": "Terry McLaurin", "player_clean_key": "terrymclaurin",
                "team": "WAS", "market": "rec_yards", "side": side,
                "vegas_line": line, "vegas_odds": -110.0,
                "model_proj": 50.32831853448081,
            })
    return pd.DataFrame(rows)


def test_select_model_bet_straddle_uses_side_compatible_real_quote():
    got = select_model_bet(_straddle_board())
    assert len(got) == 1
    row = got.iloc[0]
    assert float(row["consensus_line"]) == 51.0
    assert row["model_pick_side"] == "UNDER"
    assert row["side"] == "UNDER"
    assert float(row["vegas_line"]) == 52.5
    assert model_side(float(row["model_proj"]), float(row["vegas_line"])) == row["side"]


def test_select_model_bet_straddle_is_row_order_invariant():
    board = _straddle_board()
    picks = []
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        got = select_model_bet(board.iloc[order].reset_index(drop=True))
        assert len(got) == 1
        picks.append((got.iloc[0]["side"], float(got.iloc[0]["vegas_line"]), got.iloc[0]["book"]))
    assert len(set(picks)) == 1, f"row order changed the straddle wager: {picks}"


def test_select_model_bet_abstains_when_no_real_quote_matches_consensus_side():
    board = _straddle_board()
    # Keep only the contradictory UNDER row at 49.5 plus OVER rows. Consensus
    # still selects UNDER, but no captured UNDER quote has projection < line.
    board = board.loc[~((board["book"] == "bookB") & (board["side"] == "UNDER"))].copy()
    got = select_model_bet(board)
    assert got.empty


def test_select_model_bet_still_collapses_a_single_book_pair():
    board = _multi_book_board()
    board = board.loc[board["book"].eq("bookA")].reset_index(drop=True)
    got = select_model_bet(board)
    assert len(got) == 1
    assert got.iloc[0]["side"] == "OVER"
    assert float(got.iloc[0]["vegas_line"]) == 216.5
