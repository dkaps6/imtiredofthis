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
