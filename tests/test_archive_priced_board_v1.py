"""Tests for the forward market-track-record archiver.

The archiver is purely downstream of a completed Full Slate pricing run: it
copies already-fetched rows into a durable ledger so a live-market track
record (CLV/ROI/calibration) can accumulate week over week at zero marginal
API cost. These tests cover the pure merge/stamp logic without touching the
real repository paths.
"""
from __future__ import annotations

import pandas as pd

from scripts.operations.archive_priced_board_v1 import (
    key_columns,
    load_priced_board,
    merge_into_ledger,
    stamp_provenance,
)


def _board_row(**overrides):
    row = {
        "player": "p.mahomes",
        "market": "pass_yards",
        "side": "OVER",
        "vegas_line": 265.5,
        "mc_proj": 268.0,
        "ensemble_proj": 267.0,
        "ensemble_status": "calibrated",
        "model_proj": 267.5,
        "fair_prob": 0.52,
        "edge_pct": 0.03,
        "team": "KC",
        "season": 2026,
        "week": 1,
    }
    row.update(overrides)
    return row


def test_stamp_provenance_adds_required_columns():
    df = pd.DataFrame([_board_row()])
    stamped = stamp_provenance(df, run_id="123", git_sha="abc")
    assert stamped.loc[0, "source_run_id"] == "123"
    assert stamped.loc[0, "source_git_sha"] == "abc"
    assert "archived_at_utc" in stamped.columns


def test_merge_into_ledger_keeps_latest_capture_per_key():
    first = pd.DataFrame([_board_row(vegas_line=265.5, archived_at_utc="2026-09-11T18:00:00Z")])
    second = pd.DataFrame([_board_row(vegas_line=264.5, archived_at_utc="2026-09-11T19:55:00Z")])

    merged = merge_into_ledger(first, second)

    assert len(merged) == 1
    assert merged.loc[0, "vegas_line"] == 264.5


def test_merge_into_ledger_keeps_distinct_keys():
    a = pd.DataFrame([_board_row(player="p.mahomes", archived_at_utc="t1")])
    b = pd.DataFrame([_board_row(player="j.allen", archived_at_utc="t1")])

    merged = merge_into_ledger(a, b)

    assert len(merged) == 2


def test_key_columns_includes_event_id_and_book_when_present():
    df = pd.DataFrame([_board_row(event_id="e1", book="draftkings")])
    keys = key_columns(df)
    assert "event_id" in keys
    assert "book" in keys


def test_load_priced_board_returns_empty_frame_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert load_priced_board(missing).empty


def test_load_priced_board_raises_on_missing_required_columns(tmp_path):
    path = tmp_path / "props_priced_clean.csv"
    pd.DataFrame([{"player": "p.mahomes"}]).to_csv(path, index=False)
    try:
        load_priced_board(path)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
