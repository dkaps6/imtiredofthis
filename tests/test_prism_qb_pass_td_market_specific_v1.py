import pandas as pd
import pytest

from scripts.research.audit_prism_qb_pass_td_market_specific_v1 import (
    add_prior_counts,
    aggregate,
    resolve_schema,
)


def _synthetic_pbp():
    rows = []
    games = [
        (2023, 1, "AAA", "g1"),
        (2023, 2, "AAA", "g2"),
        (2024, 1, "BBB", "g3"),
        (2025, 1, "BBB", "g4"),
    ]
    for season, week, team, game_id in games:
        rows.extend([
            {
                "season": season,
                "week": week,
                "game_type": "REG",
                "offense_team": team,
                "pass_attempt": 1,
                "passing_touchdown": 0,
                "passer_id": "QB-1",
                "passer_name": "Test QB",
                "yardline_100": 35,
                "game_id": game_id,
            },
            {
                "season": season,
                "week": week,
                "game_type": "REG",
                "offense_team": team,
                "pass_attempt": 1,
                "passing_touchdown": 1,
                "passer_id": "QB-1",
                "passer_name": "Test QB",
                "yardline_100": 5,
                "game_id": game_id,
            },
        ])
    return pd.DataFrame(rows)


def test_schema_aliases_and_scoring_area_aggregation():
    raw = _synthetic_pbp()
    schema = resolve_schema(raw)

    assert schema["season_type"] == "game_type"
    assert schema["team"] == "offense_team"
    assert schema["pass_td"] == "passing_touchdown"
    assert schema["passer_id"] == "passer_id"

    qb_rows, team_rows, coverage = aggregate(raw, schema)

    assert len(qb_rows) == 4
    row = qb_rows.iloc[0]
    assert row["pass_attempts"] == 2
    assert row["pass_tds"] == 1
    assert row["inside20_pass_attempts"] == 1
    assert row["inside10_pass_attempts"] == 1
    assert row["inside5_pass_attempts"] == 1
    assert row["inside20_pass_tds"] == 1
    assert row["inside10_pass_tds"] == 1
    assert row["inside5_pass_tds"] == 1
    assert row["pass_td_rate"] == pytest.approx(0.5)

    assert len(team_rows) == 4
    assert coverage["passer_id_coverage"] == pytest.approx(1.0)
    assert coverage["pass_td_coverage"] == pytest.approx(1.0)
    assert coverage["yardline_coverage"] == pytest.approx(1.0)


def test_prior_counts_are_strictly_prior_across_seasons():
    raw = _synthetic_pbp()
    schema = resolve_schema(raw)
    qb_rows, _, _ = aggregate(raw, schema)
    out = add_prior_counts(qb_rows)

    assert out["strict_prior_qb_games"].tolist() == [0, 1, 2, 3]


def test_missing_regular_season_semantic_fails_closed():
    raw = _synthetic_pbp().drop(columns=["game_type"])
    with pytest.raises(RuntimeError, match="season_type"):
        resolve_schema(raw)
