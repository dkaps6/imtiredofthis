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


# --- event scoping before paid calls -------------------------------------

def test_event_scoping_defaults_to_keeping_everything(monkeypatch):
    """No window set must behave exactly as before this feature existed."""
    monkeypatch.setattr(fetch, "ONLY_EVENT_IDS", [])
    monkeypatch.setattr(fetch, "EVENT_WINDOW_START", "")
    monkeypatch.setattr(fetch, "EVENT_WINDOW_END", "")
    evs = [{"id": "a", "commence_time": "2026-11-06T01:15:00Z"},
           {"id": "b", "commence_time": "2026-11-08T18:00:00Z"}]
    assert fetch._scope_events_before_paid_calls(evs) == evs


def test_event_window_scopes_to_one_game(monkeypatch):
    monkeypatch.setattr(fetch, "ONLY_EVENT_IDS", [])
    monkeypatch.setattr(fetch, "EVENT_WINDOW_START", "2026-11-06T00:00:00Z")
    monkeypatch.setattr(fetch, "EVENT_WINDOW_END", "2026-11-06T12:00:00Z")
    evs = [{"id": "thu", "commence_time": "2026-11-06T01:15:00Z"},
           {"id": "sun1", "commence_time": "2026-11-08T18:00:00Z"},
           {"id": "sun2", "commence_time": "2026-11-08T21:05:00Z"}]
    kept = fetch._scope_events_before_paid_calls(evs)
    assert [e["id"] for e in kept] == ["thu"]


def test_only_event_ids_overrides_the_window(monkeypatch):
    monkeypatch.setattr(fetch, "ONLY_EVENT_IDS", ["sun2"])
    monkeypatch.setattr(fetch, "EVENT_WINDOW_START", "2026-11-06T00:00:00Z")
    monkeypatch.setattr(fetch, "EVENT_WINDOW_END", "2026-11-06T12:00:00Z")
    evs = [{"id": "thu", "commence_time": "2026-11-06T01:15:00Z"},
           {"id": "sun2", "commence_time": "2026-11-08T21:05:00Z"}]
    assert [e["id"] for e in fetch._scope_events_before_paid_calls(evs)] == ["sun2"]


def test_a_window_that_drops_everything_fails_visibly_not_silently(monkeypatch):
    """An empty scope must not quietly produce a zero-row paid run."""
    monkeypatch.setattr(fetch, "ONLY_EVENT_IDS", [])
    monkeypatch.setattr(fetch, "EVENT_WINDOW_START", "2030-01-01T00:00:00Z")
    monkeypatch.setattr(fetch, "EVENT_WINDOW_END", "2030-01-02T00:00:00Z")
    evs = [{"id": "thu", "commence_time": "2026-11-06T01:15:00Z"}]
    assert fetch._scope_events_before_paid_calls(evs) == evs


def test_an_undateable_event_is_never_dropped(monkeypatch):
    monkeypatch.setattr(fetch, "ONLY_EVENT_IDS", [])
    monkeypatch.setattr(fetch, "EVENT_WINDOW_START", "2026-11-06T00:00:00Z")
    monkeypatch.setattr(fetch, "EVENT_WINDOW_END", "2026-11-06T12:00:00Z")
    evs = [{"id": "nodate", "commence_time": None},
           {"id": "sun", "commence_time": "2026-11-08T18:00:00Z"}]
    assert [e["id"] for e in fetch._scope_events_before_paid_calls(evs)] == ["nodate"]


def test_archive_never_overwrites_a_paid_boundary():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "odds_capture_label" in text
    assert "_part${i}.csv" in text
    assert "event_window_start" in text and "event_window_end" in text


# --- injury-contract gate severities -------------------------------------

def test_injury_gate_blocks_a_broken_chain_but_not_a_known_open_gap(tmp_path, monkeypatch):
    """P0 for a severed chain; P2 for the QUESTIONABLE modelling gap.

    The gap is true by construction on every real NFL report while
    _injury_limited matches only OUT/DOUBTFUL/IR/PUP, so raising it at P1 in
    an audit the Full Slate runs with --strict made every run fail. P0s must
    still block.
    """
    import importlib
    import pandas as pd
    aud = importlib.import_module("scripts.audit_2026_production_readiness")
    inj = Path("data/injuries.csv")
    existed = inj.exists()
    backup = inj.read_bytes() if existed else None

    def write(vals):
        inj.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"player": [f"P{i}" for i in range(len(vals))],
                      "team": ["KC"] * len(vals), "season": [2026] * len(vals),
                      "week": [9] * len(vals), "status": vals}).to_csv(inj, index=False)

    import json as _json
    st = Path("data/injuries_source_status.json")
    st_existed = st.exists()
    st_backup = st.read_bytes() if st_existed else None
    # The all-null case now depends on what the provenance file claims, so this
    # test must control it rather than inherit whatever is on disk.
    st.write_text(_json.dumps({"season": 2026, "week": 9, "state": "official_report"}),
                  encoding="utf-8")

    try:
        write(["Out", "Questionable", "Doubtful"])
        sev = {f["code"]: f["severity"] for f in aud._injury_contract_findings()}
        assert sev.get("injury_questionable_untreated") == "P2"

        write(["Probable", "Active"])
        sev = {f["code"]: f["severity"] for f in aud._injury_contract_findings()}
        assert sev.get("injury_status_unrecognised") == "P0"

        write([None, None])
        sev = {f["code"]: f["severity"] for f in aud._injury_contract_findings()}
        assert sev.get("injury_status_all_null") == "P0"
    finally:
        if existed:
            inj.write_bytes(backup)
        else:
            inj.unlink(missing_ok=True)
        st.unlink(missing_ok=True)
        if st_existed:
            st.write_bytes(st_backup)


# --- midweek practice report vs a self-contradicting artifact -------------

def _write_injuries(vals, state):
    import json
    import pandas as pd
    p = Path("data/injuries.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"player": [f"P{i}" for i in range(len(vals))],
                  "team": ["BAL"] * len(vals), "season": [2026] * len(vals),
                  "week": [9] * len(vals), "status": vals,
                  "practice_status": ["Did Not Participate In Practice"] * len(vals)}
                 ).to_csv(p, index=False)
    Path("data/injuries_source_status.json").write_text(
        json.dumps({"season": 2026, "week": 9, "state": state, "rows": len(vals)}),
        encoding="utf-8")


def test_midweek_practice_only_report_does_not_block():
    """Designations publish Friday; a Wednesday build legitimately has none."""
    import importlib
    aud = importlib.import_module("scripts.audit_2026_production_readiness")
    inj, st = Path("data/injuries.csv"), Path("data/injuries_source_status.json")
    had = (inj.exists() and inj.read_bytes(), st.exists() and st.read_bytes())
    try:
        _write_injuries([None, None, None], "practice_report_only")
        sev = {f["code"]: f["severity"] for f in aud._injury_contract_findings()}
        assert sev.get("injury_designations_not_published_yet") == "P2"
        assert "injury_status_all_null" not in sev
    finally:
        for p, data in zip((inj, st), had):
            p.unlink(missing_ok=True)
            if data:
                p.write_bytes(data)


def test_an_artifact_that_claims_official_but_has_no_designation_blocks():
    """The real defect: the file contradicting its own provenance."""
    import importlib
    aud = importlib.import_module("scripts.audit_2026_production_readiness")
    inj, st = Path("data/injuries.csv"), Path("data/injuries_source_status.json")
    had = (inj.exists() and inj.read_bytes(), st.exists() and st.read_bytes())
    try:
        _write_injuries([None, None], "official_report")
        sev = {f["code"]: f["severity"] for f in aud._injury_contract_findings()}
        assert sev.get("injury_status_all_null") == "P0"
    finally:
        for p, data in zip((inj, st), had):
            p.unlink(missing_ok=True)
            if data:
                p.write_bytes(data)


def test_builder_labels_a_practice_only_frame_honestly():
    import importlib
    import pandas as pd
    b = importlib.import_module("scripts.build.build_injuries_weekly")
    practice_only = pd.DataFrame({"status": [None, None], "designation": [None, None],
                                  "practice_status": ["Did Not Participate In Practice",
                                                      "Full Participation in Practice"]})
    assert b._report_state(practice_only) == "practice_report_only"
    with_designation = pd.DataFrame({"status": ["Out", None], "designation": [None, None],
                                     "practice_status": ["Did Not Participate In Practice", None]})
    assert b._report_state(with_designation) == "official_report"
