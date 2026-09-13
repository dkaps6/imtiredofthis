import pandas as pd

from scripts.build._schedule_utils import _coerce_kickoff, _detect_columns


def test_gameday_gametime_fallback_localizes_eastern_not_naive_utc():
    """nflverse's gameday/gametime columns are naive Eastern local time
    (e.g. gametime="13:00" for a 1:00pm ET kickoff), not UTC. Labeling that
    naive string as UTC directly (pd.to_datetime(..., utc=True)) shifts every
    kickoff 4-5 hours earlier than reality and makes games falsely appear to
    have already kicked off hours before they actually have -- the bug that
    caused KICKED_OFF_LOCKED to fire on ~10 Week 1 games that hadn't started.
    """
    df = pd.DataFrame({
        "season": [2026],
        "week": [1],
        "home_team": ["KC"],
        "away_team": ["BAL"],
        "gameday": ["2026-09-13"],
        "gametime": ["13:00"],
    })
    columns = _detect_columns(df)
    assert columns.kickoff is None  # forces the gameday+gametime fallback path
    kickoff = _coerce_kickoff(df, columns)
    # 1:00pm ET during EDT (September) is 17:00 UTC, not 13:00 UTC.
    assert kickoff.iloc[0] == pd.Timestamp("2026-09-13T17:00:00", tz="UTC")


def test_gameday_gametime_fallback_handles_winter_est_offset():
    df = pd.DataFrame({
        "season": [2026],
        "week": [18],
        "home_team": ["KC"],
        "away_team": ["BAL"],
        "gameday": ["2027-01-03"],
        "gametime": ["13:00"],
    })
    columns = _detect_columns(df)
    kickoff = _coerce_kickoff(df, columns)
    # 1:00pm ET during EST (January) is 18:00 UTC, not 17:00 UTC -- DST must
    # be resolved via real timezone localization, not a fixed offset.
    assert kickoff.iloc[0] == pd.Timestamp("2027-01-03T18:00:00", tz="UTC")


def test_explicit_utc_kickoff_column_is_unaffected():
    df = pd.DataFrame({
        "season": [2026],
        "week": [1],
        "home_team": ["KC"],
        "away_team": ["BAL"],
        "start_time_utc": ["2026-09-13T17:00:00Z"],
    })
    columns = _detect_columns(df)
    assert columns.kickoff == "start_time_utc"
    kickoff = _coerce_kickoff(df, columns)
    assert kickoff.iloc[0] == pd.Timestamp("2026-09-13T17:00:00", tz="UTC")
