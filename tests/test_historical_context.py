import pandas as pd
import pytest

from scripts.backtest.historical_context import (
    assert_no_future_rows,
    before_cutoff,
    build_historical_player_inputs,
    build_historical_team_form,
)


def test_before_cutoff_excludes_target_and_future_weeks():
    df = pd.DataFrame([
        {"season": 2024, "week": 18, "x": 1},
        {"season": 2025, "week": 1, "x": 2},
        {"season": 2025, "week": 7, "x": 3},
        {"season": 2025, "week": 8, "x": 999},
        {"season": 2025, "week": 9, "x": 9999},
    ])
    out = before_cutoff(df, 2025, 8)
    assert set(out["x"]) == {1, 2, 3}
    assert_no_future_rows(out, 2025, 8, "test")


def test_leakage_guard_hard_fails_on_target_week():
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        assert_no_future_rows(pd.DataFrame([{"season": 2025, "week": 8}]), 2025, 8, "bad")


def test_player_inputs_ignore_target_week_outcome():
    logs = pd.DataFrame([
        {"season": 2024, "week": 18, "player": "Alpha WR", "team": "BUF", "position": "WR", "tgt_share_game": .20, "rush_share_game": 0, "ypt_game": 8.0, "catch_rate_game": .65},
        {"season": 2025, "week": 7, "player": "Alpha WR", "team": "BUF", "position": "WR", "tgt_share_game": .30, "rush_share_game": 0, "ypt_game": 9.0, "catch_rate_game": .70},
        {"season": 2025, "week": 8, "player": "Alpha WR", "team": "BUF", "position": "WR", "tgt_share_game": .99, "rush_share_game": 0, "ypt_game": 99.0, "catch_rate_game": 1.0},
    ])
    universe = pd.DataFrame([{"player": "Alpha WR", "team": "BUF", "opponent": "MIA", "position": "WR", "role": "LWR"}])
    hist, pf, con = build_historical_player_inputs(logs, universe, 2025, 8, 2024)
    assert hist["week"].max() == 7
    assert pf.iloc[0]["tgt_share"] == pytest.approx(.30)
    assert con.iloc[0]["tgt_share_prior"] == pytest.approx(.20)
    assert con.iloc[0]["tgt_share_current"] == pytest.approx(.30)
    assert con.iloc[0]["ypt_current"] == pytest.approx(9.0)


def test_player_universe_is_explicit_not_derived_from_results():
    logs = pd.DataFrame([
        {"season": 2025, "week": 7, "player": "Known WR", "team": "BUF", "position": "WR", "tgt_share_game": .25},
        {"season": 2025, "week": 8, "player": "Surprise WR", "team": "BUF", "position": "WR", "tgt_share_game": .80},
    ])
    universe = pd.DataFrame([{"player": "Known WR", "team": "BUF", "opponent": "MIA", "position": "WR"}])
    _, pf, _ = build_historical_player_inputs(logs, universe, 2025, 8, 2024)
    assert pf["player"].tolist() == ["Known WR"]


def test_team_form_uses_current_weeks_before_cutoff_and_prior_for_week1():
    teams = pd.DataFrame([
        {"season": 2024, "week": 17, "team": "BUF", "success_rate_off": .50, "plays_est": 64},
        {"season": 2024, "week": 18, "team": "BUF", "success_rate_off": .54, "plays_est": 66},
        {"season": 2025, "week": 1, "team": "BUF", "success_rate_off": .60, "plays_est": 70},
        {"season": 2025, "week": 2, "team": "BUF", "success_rate_off": .99, "plays_est": 80},
    ])
    _, week2 = build_historical_team_form(teams, 2025, 2, 2024)
    assert week2.iloc[0]["success_rate_off"] == pytest.approx(.60)
    _, week1 = build_historical_team_form(teams, 2025, 1, 2024)
    assert week1.iloc[0]["success_rate_off"] == pytest.approx(.52)
    assert week1.iloc[0]["team_history_source"] == "prior"
