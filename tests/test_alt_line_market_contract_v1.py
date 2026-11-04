"""Contract tests for the alternate-line capture and the anytime_td removal.

These lock credit-safety properties on the PAID odds path:
  * anytime_td is gone from the pull and stays gone (standing instruction);
  * alternate markets are opt-in and default OFF;
  * a push can never enable them, whatever the commit message says;
  * alternate keys are disjoint from the main-line list, so no market is
    paid for twice;
  * alternate rows never reach props_raw.csv or the pricing path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

fetch = pytest.importorskip("scripts.fetch_props_oddsapi")

WORKFLOW = Path(".github/workflows/full-slate.yml")

# Confirmed against the-odds-api.com/sports-odds-data/betting-markets.html
EXPECTED_ALT = {
    "player_pass_yds_alternate",
    "player_rush_yds_alternate",
    "player_reception_yds_alternate",
    "player_receptions_alternate",
    "player_rush_reception_yds_alternate",
}


def test_anytime_td_is_not_in_the_pull():
    """Removed 2026-11-04 by standing instruction. It must not drift back."""
    assert "player_anytime_td" not in fetch.MARKETS
    assert not any("anytime" in m for m in fetch.MARKETS)


def test_alt_line_markets_are_the_confirmed_keys():
    assert set(fetch.ALT_LINE_MARKETS) == EXPECTED_ALT


def test_alt_markets_are_disjoint_from_main_markets():
    """Overlap would pay for the same market twice in one run."""
    assert not set(fetch.ALT_LINE_MARKETS) & set(fetch.MARKETS)


def test_alt_capture_defaults_off(monkeypatch):
    """Absent the env var, no alternate-line credit may be spent."""
    monkeypatch.delenv("FETCH_ALT_LINES", raising=False)
    import importlib

    reloaded = importlib.reload(fetch)
    try:
        assert reloaded.FETCH_ALT_LINES is False
    finally:
        importlib.reload(fetch)


def test_alt_capture_writes_its_own_artifact_not_props_raw():
    assert reads_alt_path() != str(fetch.PROPS_RAW_DATA_PATH)
    assert "props_alt_raw" in reads_alt_path()


def reads_alt_path() -> str:
    return str(fetch.ALT_RAW_PATH)


def test_alt_rows_are_never_appended_to_the_main_frames():
    """The capture's return value must only be written to ALT_RAW_PATH.

    Guards the isolation property by source inspection: a ladder in
    props_raw.csv would multiply rows and break the pricing audits.
    """
    src = Path("scripts/fetch_props_oddsapi.py").read_text(encoding="utf-8")
    block = src[src.index("if FETCH_ALT_LINES:"):]
    block = block[: block.index("\n    else:")]
    assert "frames.append" not in block
    assert "ALT_RAW_PATH" in block


def test_a_push_can_never_enable_alt_lines():
    """fetch_live_odds has a commit-message escape hatch. Alt lines must not."""
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"^  FETCH_ALT_LINES: (.+)$", text, re.MULTILINE)
    assert m, "FETCH_ALT_LINES env not declared in the workflow"
    expr = m.group(1)
    assert "head_commit" not in expr
    assert "github.event_name" not in expr
    assert "inputs.fetch_alt_lines" in expr
    assert "'false'" in expr


def test_alt_boundary_is_archived_to_a_durable_path():
    """AGENTS.md 5a: a paid boundary that only exists as an artifact expires."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "data/market_track_record/alt_props" in text
    assert "fetch_alt_lines" in text
