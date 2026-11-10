"""The injury report state must describe the artifact, not the row count.

Root cause this pins: NFL clubs publish practice participation Wednesday and
Thursday and the GAME DESIGNATION (Out / Doubtful / Questionable) on FRIDAY.
``build_injuries_with_status`` used to label any non-empty pull
``official_report``. Downstream consumers trusted that label, found ``status``
all-null, and ``simulation_rules._injury_limited`` silently never fired -- so
every unavailable player kept full opportunity.

Concretely: a WR who was Out in weeks 7 and 8 and Questionable in week 9
reached a shipped card because his pre-injury production was treated as clean
evidence. AGENTS.md makes the player-state anomaly gate non-negotiable.
"""
import json

import pandas as pd
import pytest

from scripts.build.build_injuries_weekly import (
    GAME_DESIGNATIONS,
    SCHEMA,
    STATE_NONE,
    STATE_OFFICIAL,
    STATE_OUTAGE,
    STATE_PRACTICE_ONLY,
    VALID_STATES,
    classify_report_state,
    designation_count,
)


def _rows(*specs):
    """Build an injuries frame from (status, practice_status) pairs."""
    if not specs:
        return pd.DataFrame(columns=SCHEMA)
    out = []
    for i, (status, practice) in enumerate(specs):
        out.append(["Player %d" % i, "BUF", 2026, 9, status, practice,
                    "Foot", status, "2026-11-07", "nflverse", 1])
    return pd.DataFrame(out, columns=SCHEMA)


def test_schema_is_eleven_columns_and_carries_status():
    assert len(SCHEMA) == 11
    assert "status" in SCHEMA and "practice_status" in SCHEMA


def test_valid_states_are_exactly_four():
    assert VALID_STATES == (STATE_OFFICIAL, STATE_PRACTICE_ONLY, STATE_NONE, STATE_OUTAGE)


def test_empty_pull_is_no_official_report():
    assert classify_report_state(_rows()) == STATE_NONE


def test_practice_only_pull_is_not_official_report():
    """The defect: midweek rows with participation and no designation."""
    df = _rows((None, "Limited Participation in Practice"),
               (None, "Full Participation in Practice"))
    assert designation_count(df) == 0
    assert classify_report_state(df) == STATE_PRACTICE_ONLY
    assert classify_report_state(df) != STATE_OFFICIAL


def test_blank_string_status_is_not_a_designation():
    df = _rows(("", "Limited Participation in Practice"))
    assert classify_report_state(df) == STATE_PRACTICE_ONLY


def test_single_questionable_makes_it_official():
    """Egbuka: Out wk7, Out wk8, Questionable wk9."""
    df = _rows((None, "Full Participation in Practice"),
               ("Questionable", "Full Participation in Practice"))
    assert designation_count(df) == 1
    assert classify_report_state(df) == STATE_OFFICIAL


@pytest.mark.parametrize("status", ["Out", "out", "OUT", "Doubtful", "Questionable"])
def test_every_game_designation_is_recognised(status):
    assert classify_report_state(_rows((status, "DNP"))) == STATE_OFFICIAL


def test_practice_participation_is_never_a_designation():
    for practice in ("Limited Participation in Practice",
                     "Full Participation in Practice",
                     "Did Not Participate In Practice"):
        assert practice.lower() not in GAME_DESIGNATIONS
        assert classify_report_state(_rows((None, practice))) == STATE_PRACTICE_ONLY


def test_missing_status_column_counts_zero_designations():
    df = _rows(("Out", "DNP")).drop(columns=["status"])
    assert designation_count(df) == 0


def test_provider_validator_rejects_mislabelled_official_report():
    """validate_2026_provider_artifacts must fail closed on the mislabel."""
    from scripts.validate_2026_provider_artifacts import validate_injuries
    df = _rows((None, "Limited Participation in Practice"))
    status = {"season": 2026, "week": 9, "state": "official_report",
              "source": "nflverse", "rows": len(df)}
    with pytest.raises(RuntimeError, match="practice_only_report"):
        validate_injuries(df, status, 2026, 9)


def test_provider_validator_accepts_practice_only_state():
    from scripts.validate_2026_provider_artifacts import validate_injuries
    df = _rows((None, "Limited Participation in Practice"))
    status = {"season": 2026, "week": 9, "state": "practice_only_report",
              "source": "nflverse", "rows": len(df)}
    res = validate_injuries(df, status, 2026, 9)
    assert "practice_only_report" in json.dumps(res)


def test_provider_validator_accepts_genuine_official_report():
    from scripts.validate_2026_provider_artifacts import validate_injuries
    df = _rows(("Out", "Did Not Participate In Practice"))
    status = {"season": 2026, "week": 9, "state": "official_report",
              "source": "nflverse", "rows": len(df)}
    res = validate_injuries(df, status, 2026, 9)
    assert "official_report" in json.dumps(res)


def test_provider_validator_still_rejects_unknown_state():
    from scripts.validate_2026_provider_artifacts import validate_injuries
    df = _rows(("Out", "DNP"))
    status = {"season": 2026, "week": 9, "state": "something_new",
              "source": "nflverse", "rows": len(df)}
    with pytest.raises(RuntimeError, match="unknown injury provider state"):
        validate_injuries(df, status, 2026, 9)


def test_practice_only_state_is_accepted_by_every_hardcoded_set():
    """The three validators that fail closed on an unknown state."""
    import pathlib
    for rel in ("scripts/validate_2026_provider_artifacts.py",
                "scripts/validate_full_slate_data_quality_v1.py",
                "scripts/repair_injuries_nflcom_v1.py"):
        text = pathlib.Path(rel).read_text(encoding="utf-8")
        assert "practice_only_report" in text, f"{rel} would fail closed on the new state"


def test_repair_still_runs_for_practice_only_pulls():
    """Team identity and scope matter regardless of designations."""
    import pathlib
    text = pathlib.Path("scripts/repair_injuries_nflcom_v1.py").read_text(encoding="utf-8")
    assert 'state not in {"official_report", "practice_only_report"}' in text
