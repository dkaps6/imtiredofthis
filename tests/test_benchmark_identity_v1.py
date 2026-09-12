from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest.benchmark_identity_v1 import (
    assert_benchmark_identity,
    home_away_from_game_id,
    parse_game_id,
)


def test_parse_game_id_canonicalizes_la_to_lar():
    parsed = parse_game_id("2024_01_SF_LA")
    assert parsed == {"season": 2024, "week": 1, "away": "SF", "home": "LAR"}


def test_home_away_handles_lar_alias():
    assert home_away_from_game_id("LAR", "2024_01_SF_LA") == "HOME"
    assert home_away_from_game_id("SF", "2024_01_SF_LA") == "AWAY"


def test_identity_accepts_valid_projection_row():
    frame = pd.DataFrame([{
        "season": 2024,
        "week": 1,
        "team": "LAR",
        "opponent": "SF",
        "game_id": "2024_01_SF_LA",
        "player_clean_key": "player",
        "market": "rec_yards",
    }])
    stats = assert_benchmark_identity(
        frame,
        label="valid",
        require_team=True,
        require_opponent=True,
    )
    assert stats["bad_rows"] == 0


def test_identity_rejects_wrong_season_game_id():
    frame = pd.DataFrame([{
        "season": 2024,
        "week": 1,
        "team": "NYJ",
        "opponent": "SF",
        "game_id": "2023_01_BUF_NYJ",
    }])
    with pytest.raises(RuntimeError, match="season_mismatch"):
        assert_benchmark_identity(frame, label="bad", require_team=True, require_opponent=True)


def test_identity_rejects_wrong_matchup_even_when_season_week_match():
    frame = pd.DataFrame([{
        "season": 2024,
        "week": 1,
        "team": "NYJ",
        "opponent": "SF",
        "game_id": "2024_01_BUF_NYJ",
    }])
    with pytest.raises(RuntimeError, match="opponent_mismatch"):
        assert_benchmark_identity(frame, label="bad", require_team=True, require_opponent=True)
