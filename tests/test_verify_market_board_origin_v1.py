"""Tests for durable live-board origin/content verification."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.operations.verify_market_board_origin_v1 import canonical_content_digest


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
