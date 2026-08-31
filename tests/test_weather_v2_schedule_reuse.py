from pathlib import Path

import pandas as pd

from scripts.build import build_weather_week_v2 as weather


def test_team_week_map_collapses_team_perspectives_to_one_game(tmp_path: Path):
    path = tmp_path / "team_week_map.csv"
    pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "ARI",
                "opponent": "LAC",
                "home_team_abbr": "LAC",
                "away_team_abbr": "ARI",
                "kickoff_utc": "2026-09-13T00:00:00+00:00",
                "game_id": "2026_01_ARI_LAC",
            },
            {
                "season": 2026,
                "week": 1,
                "team": "LAC",
                "opponent": "ARI",
                "home_team_abbr": "LAC",
                "away_team_abbr": "ARI",
                "kickoff_utc": "2026-09-13T00:00:00+00:00",
                "game_id": "2026_01_ARI_LAC",
            },
        ]
    ).to_csv(path, index=False)

    schedule = weather.load_schedule_from_team_week_map(2026, 1, path)

    assert len(schedule) == 1
    assert schedule.loc[0, "home"] == "LAC"
    assert schedule.loc[0, "away"] == "ARI"
    assert schedule.loc[0, "game_id"] == "2026_01_ARI_LAC"


def test_weather_slate_uses_supplied_authoritative_schedule_without_provider(monkeypatch):
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "home": "LAC",
                "away": "ARI",
                "kickoff_utc": "2026-09-13T00:00:00+00:00",
                "game_id": "2026_01_ARI_LAC",
            }
        ]
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("weather attempted a redundant schedule provider call")

    monkeypatch.setattr(weather, "get_nfl_schedule", fail_if_called)
    slate = weather.build_weather_slate(2026, 1, schedule=schedule)

    assert len(slate) == 1
    assert slate.loc[0, "home"] == "LAC"
    assert slate.loc[0, "away"] == "ARI"
