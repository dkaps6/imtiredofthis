"""Tests for durable live-board origin/content verification."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.operations.verify_market_board_origin_v1 import (
    canonical_content_digest,
    verify,
)


def test_digest_ignores_archive_provenance_and_row_order(tmp_path: Path):
    base = pd.DataFrame(
        [
            {"player": "A", "vegas_line": 10.5, "side": "OVER"},
            {"player": "B", "vegas_line": 20.5, "side": "UNDER"},
        ]
    )
    a = base.copy()
    a["archived_at_utc"] = "2026-09-01T00:00:00Z"
    a["source_run_id"] = "1"
    a["source_git_sha"] = "aaa"

    b = base.iloc[::-1].copy()
    b["archived_at_utc"] = "2026-09-02T00:00:00Z"
    b["source_run_id"] = "2"
    b["source_git_sha"] = "bbb"

    pa = tmp_path / "a.csv"
    pb = tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)

    assert canonical_content_digest(pa) == canonical_content_digest(pb)


def test_simultaneous_board_and_manifest_edit_cannot_bypass_immutable_anchor(
    tmp_path: Path,
):
    original = pd.DataFrame(
        [
            {
                "player": "A",
                "vegas_line": 10.5,
                "side": "OVER",
                "source_run_id": "123",
                "source_git_sha": "abc",
            }
        ]
    )
    original_path = tmp_path / "original.csv"
    original.to_csv(original_path, index=False)
    rows, cols, original_digest = canonical_content_digest(original_path)

    immutable = {
        "version": "MARKET_BOARD_IMMUTABLE_ORIGIN_ANCHORS_V1",
        "weeks": {
            "1": {
                "rows": rows,
                "content_columns": cols,
                "canonical_content_sha256": original_digest,
                "source_run_id": "123",
                "source_git_sha": "abc",
            }
        },
    }

    mutated = original.copy()
    mutated.loc[0, "vegas_line"] = 99.5
    board_path = tmp_path / "mutated.csv"
    mutated.to_csv(board_path, index=False)
    m_rows, m_cols, mutated_digest = canonical_content_digest(board_path)

    # This is the old bypass: mutate the board and update the editable manifest
    # to bless the new content in the same change.
    manifest = {
        "weeks": {
            "1": {
                "board_path": str(board_path),
                "rows": m_rows,
                "content_columns": m_cols,
                "canonical_content_sha256": mutated_digest,
                "source_run_id": "123",
                "source_git_sha": "abc",
                "origin_evidence": "editable",
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable anchor"):
        verify(manifest_path, immutable_anchors=immutable)
