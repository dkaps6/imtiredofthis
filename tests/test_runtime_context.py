from pathlib import Path

import pandas as pd
import pytest

from scripts.runtime_context import RUNTIME_WEEK_ENV, resolve_week


def _write_schedule(tmp_path):
    path = tmp_path / "team_week_map.csv"
    pd.DataFrame([
        {"season": 2026, "week": 1, "team": "BUF", "opponent": "NYJ", "kickoff_utc": "2026-09-10T00:00:00Z"},
        {"season": 2026, "week": 1, "team": "NYJ", "opponent": "BUF", "kickoff_utc": "2026-09-10T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "BUF", "opponent": "MIA", "kickoff_utc": "2026-09-17T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "MIA", "opponent": "BUF", "kickoff_utc": "2026-09-17T00:00:00Z"},
    ]).to_csv(path, index=False)
    return path


def _write_rollover_schedule(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"season": 2026, "week": 1, "team": "DAL", "opponent": "NYG", "gameday": "2026-09-13", "kickoff_utc": "2026-09-13T00:00:00Z"},
        {"season": 2026, "week": 1, "team": "NYG", "opponent": "DAL", "gameday": "2026-09-13", "kickoff_utc": "2026-09-13T00:00:00Z"},
        {"season": 2026, "week": 1, "team": "KC", "opponent": "DEN", "gameday": "2026-09-14", "kickoff_utc": "2026-09-14T00:00:00Z"},
        {"season": 2026, "week": 1, "team": "DEN", "opponent": "KC", "gameday": "2026-09-14", "kickoff_utc": "2026-09-14T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "DAL", "opponent": "PHI", "gameday": "2026-09-17", "kickoff_utc": "2026-09-17T00:00:00Z"},
        {"season": 2026, "week": 2, "team": "PHI", "opponent": "DAL", "gameday": "2026-09-17", "kickoff_utc": "2026-09-17T00:00:00Z"},
    ]).to_csv(path, index=False)
    return path


def test_blank_date_uses_nearest_upcoming_week(tmp_path, monkeypatch):
    monkeypatch.delenv(RUNTIME_WEEK_ENV, raising=False)
    path = _write_schedule(tmp_path)
    week = resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=path,
        now=pd.Timestamp("2026-09-12T12:00:00Z"),
    )
    assert week == 2


def test_explicit_date_uses_nfl_schedule_not_iso_week(tmp_path, monkeypatch):
    monkeypatch.delenv(RUNTIME_WEEK_ENV, raising=False)
    path = _write_schedule(tmp_path)
    assert resolve_week(season=2026, slate_date="2026-09-10", team_week_map_path=path) == 1


def test_sunday_night_does_not_roll_to_week2_at_utc_midnight(tmp_path, monkeypatch):
    """Regression for Full Slate run 34791136744.

    2026-09-14 00:00 UTC is still Sunday evening in the NFL calendar zone. A
    date-only schedule must remain Week 1 rather than jumping to the next future
    Week-2 date.
    """
    monkeypatch.delenv(RUNTIME_WEEK_ENV, raising=False)
    path = _write_rollover_schedule(tmp_path / "team_week_map.csv")
    week = resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=path,
        now=pd.Timestamp("2026-09-14T00:00:09Z"),
    )
    assert week == 1


def test_frozen_runtime_week_is_stable_after_wall_clock_moves_to_next_week(tmp_path, monkeypatch):
    path = _write_rollover_schedule(tmp_path / "team_week_map.csv")
    monkeypatch.setenv(RUNTIME_WEEK_ENV, "1")
    week = resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=path,
        now=pd.Timestamp("2026-09-17T12:00:00Z"),
    )
    assert week == 1


def test_frozen_runtime_week_must_exist_in_authoritative_schedule(tmp_path, monkeypatch):
    path = _write_rollover_schedule(tmp_path / "team_week_map.csv")
    monkeypatch.setenv(RUNTIME_WEEK_ENV, "19")
    with pytest.raises(RuntimeError, match="Frozen NFL_RUNTIME_WEEK=19 is invalid"):
        resolve_week(season=2026, slate_date="", team_week_map_path=path)


def test_frozen_runtime_week_must_not_conflict_with_explicit_slate_date(tmp_path, monkeypatch):
    path = _write_rollover_schedule(tmp_path / "team_week_map.csv")
    monkeypatch.setenv(RUNTIME_WEEK_ENV, "1")
    with pytest.raises(RuntimeError, match="conflicts with SLATE_DATE=2026-09-17"):
        resolve_week(season=2026, slate_date="2026-09-17", team_week_map_path=path)


def test_full_slate_github_action_persists_first_resolved_week(tmp_path, monkeypatch):
    """The first canonical Full Slate resolution becomes a job-wide env value."""
    monkeypatch.chdir(tmp_path)
    schedule = _write_rollover_schedule(Path("data/team_week_map.csv"))
    github_env = tmp_path / "github_env"
    github_env.write_text("", encoding="utf-8")
    monkeypatch.delenv(RUNTIME_WEEK_ENV, raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "dkaps6/imtiredofthis/.github/workflows/full-slate.yml@refs/heads/main",
    )

    assert resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=schedule,
        now=pd.Timestamp("2026-09-14T00:00:09Z"),
    ) == 1
    assert github_env.read_text(encoding="utf-8") == "NFL_RUNTIME_WEEK=1\n"
    assert resolve_week(
        season=2026,
        slate_date="",
        team_week_map_path=schedule,
        now=pd.Timestamp("2026-09-17T12:00:00Z"),
    ) == 1
