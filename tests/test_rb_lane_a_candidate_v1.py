import pandas as pd
import pytest

from scripts.backtest.rb_lane_a_candidate_v1 import (
    check_rush_yard_translation_constructibility,
    compute_incumbent_ypc,
    translate_candidate_rush_yards,
)


def _scored(rows):
    cols = ["season", "week", "team", "player_clean_key"]
    return pd.DataFrame(rows, columns=cols)


def _dual(rows):
    cols = ["season", "week", "team", "player_clean_key", "promotion_rush_att", "promotion_rush_yards"]
    return pd.DataFrame(rows, columns=cols)


def test_constructibility_passes_when_all_rows_have_finite_efficiency():
    scored = _scored([[2024, 3, "TB", "p1"], [2024, 3, "TB", "p2"]])
    dual = _dual(
        [
            [2024, 3, "TB", "p1", 12.0, 60.0],
            [2024, 3, "TB", "p2", 8.0, 30.0],
        ]
    )
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"
    assert result["rows_failing"] == 0


def test_constructibility_fails_closed_on_missing_join():
    scored = _scored([[2024, 3, "TB", "p1"], [2024, 3, "TB", "p2"]])
    dual = _dual([[2024, 3, "TB", "p1", 12.0, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
    assert result["rows_failing"] == 1
    assert result["failing_rows"][0]["player_clean_key"] == "p2"


def test_constructibility_fails_closed_on_rush_att_at_or_below_floor():
    scored = _scored([[2024, 3, "TB", "p1"]])
    dual = _dual([[2024, 3, "TB", "p1", 0.20, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"


def test_constructibility_fails_closed_on_nonfinite_rush_yards():
    scored = _scored([[2024, 3, "TB", "p1"]])
    dual = pd.DataFrame(
        [[2024, 3, "TB", "p1", 12.0, None]],
        columns=["season", "week", "team", "player_clean_key", "promotion_rush_att", "promotion_rush_yards"],
    )
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"


def test_constructibility_fails_closed_on_missing_columns():
    scored = pd.DataFrame({"season": [2024]})
    dual = _dual([[2024, 3, "TB", "p1", 12.0, 60.0]])
    result = check_rush_yard_translation_constructibility(scored, dual)
    assert result["disposition"] == "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
    assert "missing required columns" in result["reason"]


def test_compute_incumbent_ypc():
    dual = _dual([[2024, 3, "TB", "p1", 10.0, 50.0]])
    out = compute_incumbent_ypc(dual)
    assert out.iloc[0]["incumbent_ypc"] == pytest.approx(5.0)


def test_translate_candidate_rush_yards():
    candidate_att = pd.DataFrame(
        [[2024, 3, "TB", "p1", 8.0]],
        columns=["season", "week", "team", "player_clean_key", "candidate_att"],
    )
    incumbent = compute_incumbent_ypc(_dual([[2024, 3, "TB", "p1", 10.0, 50.0]]))
    out = translate_candidate_rush_yards(candidate_att, incumbent)
    assert out.iloc[0]["candidate_rush_yards"] == pytest.approx(40.0)


def test_translate_candidate_rush_yards_fails_closed_on_unmatched_row():
    candidate_att = pd.DataFrame(
        [[2024, 3, "TB", "p_unknown", 8.0]],
        columns=["season", "week", "team", "player_clean_key", "candidate_att"],
    )
    incumbent = compute_incumbent_ypc(_dual([[2024, 3, "TB", "p1", 10.0, 50.0]]))
    with pytest.raises(RuntimeError, match="no incumbent_ypc match"):
        translate_candidate_rush_yards(candidate_att, incumbent)
