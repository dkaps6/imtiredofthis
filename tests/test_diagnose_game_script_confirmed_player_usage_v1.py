import numpy as np
import pandas as pd
import pytest

from scripts.research.diagnose_game_script_confirmed_player_usage_v1 import (
    build_analysis_frame,
    compare_by_side,
    games_to_schedule_history,
    run_margin_hypothesis,
    run_total_hypothesis,
    to_team_game_market_frame,
)


def _fake_games():
    """Four games, home/away, with predicted vs actual margin/total set up so
    some are 'confirmed' (close) and some are 'missed' (far off).
    """
    return pd.DataFrame([
        {"season": 2024, "week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB",
         "predicted_margin_home": 10.0, "predicted_total": 44.0,
         "actual_margin_home": 11.0, "actual_total": 45.0},  # confirmed both
        {"season": 2024, "week": 2, "game_id": "g2", "home_team": "CCC", "away_team": "DDD",
         "predicted_margin_home": -6.0, "predicted_total": 50.0,
         "actual_margin_home": -5.0, "actual_total": 48.0},  # confirmed both
        {"season": 2024, "week": 3, "game_id": "g3", "home_team": "EEE", "away_team": "FFF",
         "predicted_margin_home": 3.0, "predicted_total": 38.0,
         "actual_margin_home": -14.0, "actual_total": 60.0},  # missed both, badly
        {"season": 2024, "week": 4, "game_id": "g4", "home_team": "GGG", "away_team": "HHH",
         "predicted_margin_home": -2.0, "predicted_total": 46.0,
         "actual_margin_home": 20.0, "actual_total": 20.0},  # missed both, badly
    ])


def test_to_team_game_market_frame_signs_away_perspective_correctly():
    games = _fake_games()
    out = to_team_game_market_frame(games)
    assert len(out) == 8
    home_row = out.loc[out.team.eq("AAA")].iloc[0]
    away_row = out.loc[out.team.eq("BBB")].iloc[0]
    assert home_row["predicted_team_margin"] == pytest.approx(10.0)
    assert away_row["predicted_team_margin"] == pytest.approx(-10.0)
    assert home_row["actual_team_margin"] == pytest.approx(11.0)
    assert away_row["actual_team_margin"] == pytest.approx(-11.0)
    assert home_row["predicted_total"] == away_row["predicted_total"] == pytest.approx(44.0)


def test_games_to_schedule_history_has_both_team_perspectives():
    games = _fake_games()
    out = games_to_schedule_history(games)
    assert len(out) == 8
    assert set(out.loc[out.team.eq("AAA")]["opponent"]) == {"BBB"}
    assert set(out.loc[out.team.eq("BBB")]["opponent"]) == {"AAA"}


def _fake_player_logs():
    """Team-game position aggregates baked directly into player rows (one
    row per position group per team-game is enough for team_game_position_
    aggregates to sum correctly).
    """
    rows = []
    teams_games = [
        ("AAA", 1, 30, 20), ("BBB", 1, 15, 35),
        ("CCC", 2, 12, 38), ("DDD", 2, 28, 22),
        ("EEE", 3, 20, 25), ("FFF", 3, 20, 25),
        ("GGG", 4, 20, 25), ("HHH", 4, 20, 25),
    ]
    for team, week, rb_rush, wrte_tgt in teams_games:
        rows.append({
            "season": 2024, "week": week, "team": team, "position": "RB",
            "rushes": rb_rush, "rush_yards": rb_rush * 4.2, "targets": 0, "rec_yards": 0,
            "team_rushes": rb_rush, "team_targets": wrte_tgt, "team_dropbacks": wrte_tgt + 5,
        })
        rows.append({
            "season": 2024, "week": week, "team": team, "position": "WR",
            "rushes": 0, "rush_yards": 0, "targets": wrte_tgt, "rec_yards": wrte_tgt * 8.0,
            "team_rushes": rb_rush, "team_targets": wrte_tgt, "team_dropbacks": wrte_tgt + 5,
        })
    return pd.DataFrame(rows)


def test_build_analysis_frame_joins_market_and_position_aggregates():
    games = _fake_games()
    player_logs = _fake_player_logs()
    frame = build_analysis_frame(games, player_logs)
    assert len(frame) == 8
    aaa = frame.loc[frame.team.eq("AAA")].iloc[0]
    assert aaa["rb_rush_att"] == pytest.approx(30)
    assert aaa["wrte_targets"] == pytest.approx(20)
    assert aaa["margin_abs_error"] == pytest.approx(1.0)
    assert aaa["total_abs_error"] == pytest.approx(1.0)


def test_compare_by_side_fails_closed_on_insufficient_rows():
    frame = pd.DataFrame({"side": [True, True, False, False], "metric": [1.0, 2.0, 3.0, 4.0]})
    result = compare_by_side(frame, side_col="side", metric_col="metric")
    assert result["status"] == "INSUFFICIENT_ROWS"


def test_margin_hypothesis_ground_truth_recovers_designed_effect():
    """Synthetic team-games where actually-leading teams run the ball ~15
    more times than actually-trailing teams -- ground truth arm (A) should
    recover this cleanly; confirmed arm (C), restricted to accurate Vegas
    predictions, should also show a positive effect since by construction
    predicted and actual agree there.
    """
    rng = np.random.default_rng(5)
    rows = []
    for i in range(80):
        actual_margin = rng.choice([-14.0, 14.0])
        predicted_margin = actual_margin + rng.normal(0, 1.0)  # tight -> mostly confirmed at T=3
        rb_rush_att = 28.0 if actual_margin > 0 else 13.0
        rows.append({
            "season": 2024, "actual_team_margin": actual_margin, "predicted_team_margin": predicted_margin,
            "margin_abs_error": abs(actual_margin - predicted_margin),
            "rb_rush_att": rb_rush_att + rng.normal(0, 1.0), "rb_rush_yards": rb_rush_att * 4.0 + rng.normal(0, 2.0),
        })
    frame = pd.DataFrame(rows)
    results = run_margin_hypothesis(frame, season_label="2024", threshold=3.0)
    by_arm = {(r["arm"], r["metric"]): r for r in results}
    gt = by_arm[("A_ground_truth_actual_split", "rb_rush_att")]
    confirmed = by_arm[("C_vegas_confirmed_only", "rb_rush_att")]
    assert gt["status"] == "OK" and gt["diff"] > 10
    assert confirmed["status"] == "OK" and confirmed["diff"] > 10


def test_total_hypothesis_uses_frozen_cutoff_consistently_across_arms():
    rows = []
    rng = np.random.default_rng(9)
    for i in range(80):
        actual_total = rng.choice([36.0, 56.0])
        predicted_total = actual_total + rng.normal(0, 1.0)
        wrte_targets = 22.0 if actual_total > 45.0 else 14.0
        rows.append({
            "season": 2024, "actual_total": actual_total, "predicted_total": predicted_total,
            "total_abs_error": abs(actual_total - predicted_total),
            "wrte_targets": wrte_targets + rng.normal(0, 1.0), "wrte_rec_yards": wrte_targets * 8.0 + rng.normal(0, 2.0),
        })
    frame = pd.DataFrame(rows)
    results = run_total_hypothesis(frame, season_label="2024", threshold=3.0, total_cutoff=45.0)
    by_arm = {(r["arm"], r["metric"]): r for r in results}
    assert by_arm[("A_ground_truth_actual_split", "wrte_targets")]["diff"] > 5
    assert by_arm[("C_vegas_confirmed_only", "wrte_targets")]["diff"] > 5
