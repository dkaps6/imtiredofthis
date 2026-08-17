import pandas as pd

from scripts.runtime_context import resolve_week


def _write_schedule(tmp_path):
    path = tmp_path / "team_week_map.csv"
    pd.DataFrame([
        {"season": 2026, "week": 1, "team": "BUF", "opponent": "NYJ", "kickoff_utc": "2026-09-10T00:00:00Z"},
        {"season": 2026, "week": 1, "team": "NYJ", "opponent": "BUF", "kickoff_utc": "2026-09-10T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "BUF", "opponent": "MIA", "kickoff_utc": "2026-09-17T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "MIA", "opponent": "BUF", "kickoff_utc": "2026-09-17T00:00:00Z"},
    ]).to_csv(path, index=False)
    return path


def test_blank_date_uses_nearest_upcoming_week(tmp_path):
    path = _write_schedule(tmp_path)
    week = resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=path,
        now=pd.Timestamp("2026-09-12T12:00:00Z"),
    )
    assert week == 2


def test_explicit_date_uses_nfl_schedule_not_iso_week(tmp_path):
    path = _write_schedule(tmp_path)
    assert resolve_week(season=2026, slate_date="2026-09-10", team_week_map_path=path) == 1
