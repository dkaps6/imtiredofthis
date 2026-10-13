from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.operations.archive_market_snapshot_history_v1 import (
    archive_snapshot,
    build_snapshot,
)
from scripts.research.compute_market_snapshot_clv_v1 import (
    american_to_decimal,
    compare,
    timing_class,
)


def _priced(**overrides):
    row = {
        "event_id": "e1",
        "player": "Alpha Back",
        "player_clean_key": "alphaback",
        "team": "IND",
        "market": "rush_yards",
        "side": "OVER",
        "book": "draftkings",
        "vegas_line": 60.5,
        "vegas_odds": -110.0,
        "mc_proj": 65.0,
        "ensemble_proj": 64.0,
        "ensemble_status": "calibrated",
        "model_proj": 64.5,
        "fair_prob": 0.56,
        "edge_pct": 0.04,
        "season": 2026,
        "week": 4,
    }
    row.update(overrides)
    return row


def _raw(**overrides):
    row = {
        "event_id": "e1",
        "fetched_at": "2026-10-04T16:00:00Z",
        "commence_time": "2026-10-04T17:00:00Z",
    }
    row.update(overrides)
    return row


def test_build_snapshot_uses_provider_fetch_time_and_kickoff(tmp_path):
    priced = tmp_path / "priced.csv"
    raw = tmp_path / "raw.csv"
    pd.DataFrame([_priced()]).to_csv(priced, index=False)
    pd.DataFrame([_raw()]).to_csv(raw, index=False)

    out = build_snapshot(priced, raw, source_run_id="123", source_git_sha="abc")

    assert out.loc[0, "odds_fetched_at_utc"] == "2026-10-04T16:00:00Z"
    assert out.loc[0, "commence_time_utc"] == "2026-10-04T17:00:00Z"
    assert out.loc[0, "minutes_to_kickoff"] == pytest.approx(60.0)
    assert bool(out.loc[0, "pregame_snapshot_valid"])
    assert out.loc[0, "source_run_id"] == "123"


def test_snapshot_archive_is_append_only_per_run(tmp_path):
    priced = tmp_path / "priced.csv"
    raw = tmp_path / "raw.csv"
    root = tmp_path / "snapshots"
    pd.DataFrame([_priced()]).to_csv(priced, index=False)
    pd.DataFrame([_raw()]).to_csv(raw, index=False)

    first = archive_snapshot(
        priced_path=priced,
        raw_path=raw,
        source_run_id="123",
        source_git_sha="abc",
        snapshot_root=root,
        season=2026,
        week=4,
    )
    assert first["operation_status"] == "archived"

    second = archive_snapshot(
        priced_path=priced,
        raw_path=raw,
        source_run_id="123",
        source_git_sha="abc",
        snapshot_root=root,
        season=2026,
        week=4,
    )
    assert second["operation_status"] == "already_archived_identical"

    pd.DataFrame([_priced(vegas_line=61.5)]).to_csv(priced, index=False)
    with pytest.raises(RuntimeError, match="immutable snapshot collision"):
        archive_snapshot(
            priced_path=priced,
            raw_path=raw,
            source_run_id="123",
            source_git_sha="abc",
            snapshot_root=root,
            season=2026,
            week=4,
        )


def _snap(*, fetched, kickoff="2026-10-04T17:00:00Z", **overrides):
    row = _priced()
    row.update({
        "odds_fetched_at_utc": pd.Timestamp(fetched, tz="UTC"),
        "commence_time_utc": pd.Timestamp(kickoff, tz="UTC"),
        "minutes_to_kickoff": (
            pd.Timestamp(kickoff, tz="UTC") - pd.Timestamp(fetched, tz="UTC")
        ).total_seconds() / 60.0,
        "source_snapshot_file": overrides.pop("source_snapshot_file", "snap.csv"),
    })
    row.update(overrides)
    return row


def test_clv_uses_latest_later_same_book_prekickoff_quote():
    entry = pd.DataFrame([_snap(
        fetched="2026-10-04 15:00:00",
        source_snapshot_file="entry.csv",
        vegas_line=60.5,
        vegas_odds=-110,
    )])
    later = pd.DataFrame([
        _snap(
            fetched="2026-10-04 16:10:00",
            source_snapshot_file="late1.csv",
            vegas_line=61.5,
            vegas_odds=-110,
        ),
        _snap(
            fetched="2026-10-04 16:45:00",
            source_snapshot_file="close.csv",
            vegas_line=62.5,
            vegas_odds=-105,
        ),
        _snap(
            fetched="2026-10-04 17:01:00",
            source_snapshot_file="post.csv",
            vegas_line=63.5,
            vegas_odds=100,
        ),
        _snap(
            fetched="2026-10-04 16:55:00",
            source_snapshot_file="otherbook.csv",
            book="fanduel",
            vegas_line=64.5,
            vegas_odds=100,
        ),
    ])

    out = compare(entry, later)
    row = out.iloc[0]

    assert row.comparison_status == "VALID_T30_CLOSE"
    assert row.close_snapshot_file == "close.csv"
    assert row.close_line == 62.5
    assert row.side_aligned_line_clv == pytest.approx(2.0)
    assert row.close_minutes_to_kickoff == pytest.approx(15.0)
    assert row.price_clv_status == "NOT_COMPARABLE_LINE_CHANGED"


def test_under_line_clv_direction_and_same_line_price_clv():
    entry = pd.DataFrame([_snap(
        fetched="2026-10-04 16:00:00",
        source_snapshot_file="entry.csv",
        side="UNDER",
        vegas_line=60.5,
        vegas_odds=110,
    )])
    later = pd.DataFrame([_snap(
        fetched="2026-10-04 16:40:00",
        source_snapshot_file="close.csv",
        side="UNDER",
        vegas_line=60.5,
        vegas_odds=100,
    )])

    out = compare(entry, later)
    row = out.iloc[0]

    assert row.comparison_status == "VALID_T30_CLOSE"
    assert row.side_aligned_line_clv == pytest.approx(0.0)
    assert row.price_clv_status == "COMPARABLE_SAME_LINE"
    expected = (american_to_decimal(110) / american_to_decimal(100) - 1.0) * 100.0
    assert row.same_line_price_clv_pct == pytest.approx(expected)


def test_timing_labels_are_not_all_called_clv():
    assert timing_class(15.0) == "VALID_T30_CLOSE"
    assert timing_class(45.0) == "VALID_T60_LATE_MARKET"
    assert timing_class(90.0) == "PREGAME_MOVEMENT_ONLY"
    assert timing_class(0.0) == "INVALID_FOR_PREGAME_MOVEMENT"
