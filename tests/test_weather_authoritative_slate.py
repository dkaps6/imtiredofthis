import pandas as pd
import pytest

from scripts.build.build_weather_week_v2 import build_weather_slate


def test_weather_slate_uses_only_authoritative_week():
    schedule = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 2],
            "game_id": ["g1", "g2", "g3"],
            "home": ["PHI", "KC", "DAL"],
            "away": ["DAL", "LAC", "NYG"],
            "kickoff_utc": [
                "2026-09-04T00:20:00Z",
                "2026-09-05T00:00:00Z",
                "2026-09-13T00:20:00Z",
            ],
        }
    )

    out = build_weather_slate(2026, 1, schedule)

    assert len(out) == 2
    assert set(out["game_id"]) == {"g1", "g2"}
    assert set(out["week"]) == {1}
    assert out["kickoff_utc"].notna().all()
    assert out["kickoff_local"].notna().all()


def test_weather_slate_rejects_duplicate_games():
    schedule = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "game_id": ["g1", "g1"],
            "home": ["PHI", "PHI"],
            "away": ["DAL", "DAL"],
            "kickoff_utc": ["2026-09-04T00:20:00Z", "2026-09-04T00:20:00Z"],
        }
    )

    with pytest.raises(RuntimeError, match="duplicate games"):
        build_weather_slate(2026, 1, schedule)


def test_weather_slate_rejects_missing_kickoff():
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": ["g1"],
            "home": ["PHI"],
            "away": ["DAL"],
            "kickoff_utc": [None],
        }
    )

    with pytest.raises(RuntimeError, match="invalid kickoff_utc"):
        build_weather_slate(2026, 1, schedule)
