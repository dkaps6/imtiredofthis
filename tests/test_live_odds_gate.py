import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_live_odds_gate import (
    _active_game_pairs,
    _active_game_windows,
    _actual_prop_rows,
    _allowed_event_ids,
    _clear_stale_odds_artifacts,
    _filter_event_csv,
)


def test_active_slate_event_gate_excludes_other_week_and_same_pair_future_rematch():
    schedule = pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "team": "IND",
            "opponent": "BAL",
            "kickoff_utc": "2026-09-13T00:00:00Z",
        },
        {
            "season": 2026,
            "week": 1,
            "team": "BAL",
            "opponent": "IND",
            "kickoff_utc": "2026-09-13T00:00:00Z",
        },
        {
            "season": 2026,
            "week": 1,
            "team": "KC",
            "opponent": "DEN",
            "kickoff_utc": "2026-09-14T00:00:00Z",
        },
        {
            "season": 2026,
            "week": 1,
            "team": "DEN",
            "opponent": "KC",
            "kickoff_utc": "2026-09-14T00:00:00Z",
        },
    ])
    pairs = _active_game_pairs(schedule, 2026, 1)
    windows = _active_game_windows(schedule, 2026, 1)
    assert pairs == set(windows)
    odds = pd.DataFrame([
        {
            "event_id": "week1-a",
            "home_team": "IND",
            "away_team": "BAL",
            "commence_time": "2026-09-13T17:00:00Z",
        },
        {
            "event_id": "week1-b",
            "home_team": "DEN",
            "away_team": "KC",
            "commence_time": "2026-09-15T00:15:00Z",
        },
        {
            "event_id": "future-rematch",
            "home_team": "KC",
            "away_team": "DEN",
            "commence_time": "2026-11-01T21:25:00Z",
        },
        {
            "event_id": "other-matchup",
            "home_team": "IND",
            "away_team": "DET",
            "commence_time": "2026-09-13T17:00:00Z",
        },
    ])
    assert _allowed_event_ids(odds, windows) == {"week1-a", "week1-b"}


def test_active_pair_with_invalid_commence_time_fails_closed():
    schedule = pd.DataFrame([
        {
            "season": 2026,
            "week": 1,
            "team": "KC",
            "opponent": "DEN",
            "kickoff_utc": "2026-09-14T00:00:00Z",
        },
        {
            "season": 2026,
            "week": 1,
            "team": "DEN",
            "opponent": "KC",
            "kickoff_utc": "2026-09-14T00:00:00Z",
        },
    ])
    windows = _active_game_windows(schedule, 2026, 1)
    odds = pd.DataFrame([
        {
            "event_id": "bad-time",
            "home_team": "KC",
            "away_team": "DEN",
            "commence_time": "not-a-time",
        },
    ])
    with pytest.raises(RuntimeError, match="invalid commence_time"):
        _allowed_event_ids(odds, windows)


def test_event_csv_filter_preserves_only_allowed_event_ids(tmp_path):
    path = tmp_path / "props_raw.csv"
    pd.DataFrame([
        {"event_id": "keep", "canonical_player_name": "Example QB"},
        {"event_id": "drop", "canonical_player_name": "Other QB"},
    ]).to_csv(path, index=False)
    assert _filter_event_csv(path, {"keep"}) == 1
    scoped = pd.read_csv(path)
    assert scoped["event_id"].tolist() == ["keep"]


def test_actual_prop_rows_ignores_missing_market_placeholders():
    props = pd.DataFrame([
        {"canonical_player_name": "", "bookmaker_missing": 1},
        {"canonical_player_name": "Example QB", "bookmaker_missing": 0},
        {"canonical_player_name": "Example WR", "bookmaker_missing": 1},
    ])
    assert _actual_prop_rows(props) == 1


def test_clear_stale_odds_artifacts_removes_previous_outputs(tmp_path, monkeypatch):
    import scripts.run_live_odds_gate as gate

    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    data.mkdir()
    outputs.mkdir()
    (outputs / "props_raw").mkdir()

    stale = [
        outputs / "props_raw.csv",
        outputs / "props_player_pass_yds.csv",
        outputs / "props_raw" / "player_pass_yds.csv",
        data / "opponent_map_from_props.csv",
        data / "live_odds_status.json",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(gate, "DATA", data)
    monkeypatch.setattr(gate, "OUTPUTS", outputs)
    monkeypatch.setattr(gate, "STATUS", data / "live_odds_status.json")
    monkeypatch.setattr(
        gate,
        "CRITICAL_EVENT_ARTIFACTS",
        [outputs / "props_raw.csv", data / "opponent_map_from_props.csv"],
    )
    # LIVE_DERIVED_ARTIFACTS is intentionally materialized at module import from
    # the canonical production paths. Unit tests that relocate DATA/OUTPUTS must
    # relocate that frozen path set as well; production behavior is unchanged.
    monkeypatch.setattr(gate, "LIVE_DERIVED_ARTIFACTS", {data / "live_odds_status.json"})

    _clear_stale_odds_artifacts()
    assert all(not path.exists() for path in stale)
