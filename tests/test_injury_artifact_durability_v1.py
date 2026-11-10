"""The injury artifact must be auditable and durable.

These pin three defects found on 2026-11-10 while verifying that the
OUT/DOUBTFUL zeroing had actually landed in production.

The investigation began with a false alarm: the durable trace of
``data/injuries.csv`` carried only player/team/season/week/source, which reads
exactly like a production artifact that had lost its ``status`` column. It had
not — the trace subsets by an allowlist and none of the injury columns were on
it. But the false alarm was only possible because the trace could not show the
chain, and it cost real time.
"""
import importlib.util
import json
from pathlib import Path

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trace_allowlist_carries_the_injury_columns():
    """The trace audits the opportunity path; injury state is an INPUT to it."""
    m = _load("scripts/research/preserve_week7_opportunity_trace_v1.py", "trace")
    wanted = {c.lower() for c in m._WANTED}
    for col in ("status", "practice_status", "designation", "body_part",
                "report_available", "injury_status", "injury_designation"):
        assert col in wanted, f"{col} dropped from the trace; the injury chain is unauditable"


def test_injury_columns_are_not_sportsbook_denied():
    """Injury state is football state. The denylist governs market data only."""
    m = _load("scripts/research/preserve_week7_opportunity_trace_v1.py", "trace")
    for col in ("status", "practice_status", "designation", "injury_status"):
        assert not m._denied(col), f"{col} trips the sportsbook denylist"


def test_injuries_artifact_is_archived_for_replay():
    """It previously had no durable copy anywhere."""
    m = _load("scripts/operations/preserve_replay_input_set_v1.py", "replay")
    archived = set(m.REQUIRED) | set(m.OPTIONAL)
    assert "data/injuries.csv" in archived
    assert "data/injuries_source_status.json" in archived


def test_refresh_script_classifies_state_instead_of_asserting_it():
    """refresh_current_injuries_nflcom_v1 overwrites injuries.csv in production.

    It hard-coded state="official_report", which reintroduces the mislabel the
    builder fix removed: a practice-only pull published as an official report,
    with _injury_limited silently never firing.
    """
    text = Path("scripts/operations/refresh_current_injuries_nflcom_v1.py").read_text(encoding="utf-8")
    assert "classify_report_state(repaired)" in text
    assert '"state": "official_report"' not in text, "state is asserted rather than classified"


def test_every_injuries_writer_uses_the_declared_schema():
    """All writers must emit the full 11-column contract, not a subset."""
    from scripts.build.build_injuries_weekly import SCHEMA
    assert len(SCHEMA) == 11
    for rel in ("scripts/operations/refresh_current_injuries_nflcom_v1.py",
                "scripts/repair_injuries_nflcom_v1.py"):
        text = Path(rel).read_text(encoding="utf-8")
        assert "[SCHEMA]" in text, f"{rel} may write a column subset"
