#!/usr/bin/env python3
"""Tests for the pregame decision-gate log salvage reader.

The reader's whole value is that it refuses to invent a fact a log does not
contain, so most of these pin the fail-closed paths rather than the happy one.
"""
from __future__ import annotations

import json

import pytest

from scripts.operations.salvage_production_decision_gates_v1 import (
    extract_json_blocks,
    resolve_block_reason,
    salvage_week,
    strip_timestamps,
)

SOURCE = {
    "run_id": "36293274478",
    "git_sha": "0982b62276303403e2ca58b16e6f4fc3e041f65d",
    "job_id": "108547303910",
    "conclusion": "success",
    "created_at": "2026-09-27T04:05:01Z",
}


def cert_meta(
    *,
    asof="2026-09-27T04:06:00+00:00",
    games=16,
    eligible=14,
    withheld=2,
    state_counts=None,
    withheld_teams=("LAR", "NE", "SEA", "SF"),
):
    return {
        "asof_utc": asof,
        "eligible_games": eligible,
        "games": games,
        "require_minutes_before_kickoff": 75.0,
        "sportsbook_inputs_used": 0,
        "state_counts": state_counts
        if state_counts is not None
        else {"KICKED_OFF_LOCKED": 2, "NOT_YET_REQUIRED": 14},
        "withheld_games": withheld,
        "withheld_teams": list(withheld_teams),
    }


def write_log(tmp_path, week, *blocks, extra=""):
    """Build a timestamp-prefixed Actions log containing the given blocks."""
    lines = ["2026-09-27T04:05:10.1234567Z Run full slate"]
    for block in blocks:
        for line in json.dumps(block, indent=2, sort_keys=True).splitlines():
            lines.append(f"2026-09-27T04:06:00.0000001Z {line}")
    if extra:
        for line in extra.splitlines():
            lines.append(f"2026-09-27T04:07:00.0000001Z {line}")
    path = tmp_path / f"week{week}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_strip_timestamps_removes_only_the_actions_prefix():
    raw = "2026-09-27T04:06:00.0000001Z   \"games\": 16,"
    assert strip_timestamps(raw) == '  "games": 16,'


def test_strip_timestamps_leaves_unprefixed_lines_alone():
    assert strip_timestamps("plain line") == "plain line"


def test_strip_timestamps_also_removes_ansi_colour_codes():
    raw = "2026-09-27T04:06:00.0000001Z \x1b[36;1m{\x1b[0m"
    assert strip_timestamps(raw) == "{"


def test_coloured_braces_do_not_hide_a_certification_block():
    """Real Actions logs colour echoed script lines, braces included."""
    body = json.dumps(cert_meta(), indent=2, sort_keys=True).splitlines()
    coloured = "\n".join(
        f"2026-09-27T04:06:00.0000001Z \x1b[36;1m{line}\x1b[0m" for line in body
    )
    blocks = extract_json_blocks(
        strip_timestamps(coloured), "require_minutes_before_kickoff"
    )
    assert len(blocks) == 1
    assert blocks[0]["games"] == 16


def test_extract_json_blocks_keeps_nested_objects_intact():
    text = json.dumps(cert_meta(), indent=2, sort_keys=True)
    blocks = extract_json_blocks(text, "require_minutes_before_kickoff")
    assert len(blocks) == 1
    assert blocks[0]["state_counts"] == {
        "KICKED_OFF_LOCKED": 2,
        "NOT_YET_REQUIRED": 14,
    }


def test_extract_json_blocks_ignores_objects_without_the_key():
    text = "\n".join(
        [
            json.dumps({"unrelated": 1}, indent=2),
            json.dumps(cert_meta(), indent=2, sort_keys=True),
        ]
    )
    assert len(extract_json_blocks(text, "require_minutes_before_kickoff")) == 1


def test_extract_json_blocks_survives_malformed_json():
    text = "{\n  not json at all\n}\n" + json.dumps(
        cert_meta(), indent=2, sort_keys=True
    )
    assert len(extract_json_blocks(text, "require_minutes_before_kickoff")) == 1


def test_resolve_block_reason_single_noneligible_state_is_attributable():
    reason, ok = resolve_block_reason(
        {"KICKED_OFF_LOCKED": 2, "NOT_YET_REQUIRED": 14}
    )
    assert (reason, ok) == ("KICKED_OFF_LOCKED", True)


def test_resolve_block_reason_mixed_states_is_not_attributable():
    reason, ok = resolve_block_reason(
        {
            "KICKED_OFF_LOCKED": 2,
            "REQUIRED_MISSING_FAIL_CLOSED": 1,
            "NOT_YET_REQUIRED": 13,
        }
    )
    assert (reason, ok) == ("WITHHELD_REASON_NOT_RECOVERABLE", False)


def test_resolve_block_reason_no_withholding():
    reason, ok = resolve_block_reason(
        {"REQUIRED_AND_CERTIFIED": 8, "NOT_YET_REQUIRED": 8}
    )
    assert (reason, ok) == ("NO_TEAMS_WITHHELD", True)


def test_resolve_block_reason_ignores_zero_count_states():
    reason, ok = resolve_block_reason(
        {"KICKED_OFF_LOCKED": 2, "REQUIRED_MISSING_FAIL_CLOSED": 0}
    )
    assert (reason, ok) == ("KICKED_OFF_LOCKED", True)


def test_recoverable_week_reads_back_every_required_fact(tmp_path):
    write_log(tmp_path, 3, cert_meta())
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["status"] == "RECOVERABLE"
    facts = got["source_facts"]
    assert facts["availability_asof_utc"] == "2026-09-27T04:06:00+00:00"
    assert facts["game_count"] == 16
    assert facts["eligible_games"] == 14
    assert facts["withheld_games"] == 2
    assert facts["withheld_teams"] == ["LAR", "NE", "SEA", "SF"]
    assert got["withheld_reason_recoverable"] is True
    assert all(r["reason"] == "KICKED_OFF_LOCKED" for r in got["block_rules"])
    assert all(r["scope"] == "TEAM_ALL" for r in got["block_rules"])


def test_recoverable_week_carries_the_provenance_envelope(tmp_path):
    write_log(tmp_path, 3, cert_meta())
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["source_run_id"] == SOURCE["run_id"]
    assert got["source_git_sha"] == SOURCE["git_sha"]
    assert got["source_job_id"] == SOURCE["job_id"]
    assert SOURCE["run_id"] in got["evidence_basis"]
    assert SOURCE["job_id"] in got["evidence_basis"]


def test_missing_log_is_not_recoverable(tmp_path):
    got = salvage_week(4, SOURCE, tmp_path / "week4.log")
    assert got["status"] == "NOT_RECOVERABLE"
    assert got["missing_fields"] == ["run_log"]
    assert "source_facts" not in got


def test_empty_log_is_not_recoverable(tmp_path):
    (tmp_path / "week4.log").write_text("", encoding="utf-8")
    got = salvage_week(4, SOURCE, tmp_path / "week4.log")
    assert got["status"] == "NOT_RECOVERABLE"
    assert got["missing_fields"] == ["run_log"]


def test_log_without_a_certification_block_is_not_recoverable(tmp_path):
    path = tmp_path / "week4.log"
    path.write_text(
        "2026-10-01T22:21:40.0000001Z priced 2528 rows\n", encoding="utf-8"
    )
    got = salvage_week(4, SOURCE, path)
    assert got["status"] == "NOT_RECOVERABLE"
    assert got["missing_fields"] == ["availability_timing_certification"]
    assert "must not be inferred" in " ".join(got["notes"])


def test_disagreeing_certification_blocks_fail_closed(tmp_path):
    write_log(
        tmp_path,
        5,
        cert_meta(eligible=14, withheld=2),
        cert_meta(eligible=13, withheld=3, games=16),
    )
    got = salvage_week(5, SOURCE, tmp_path / "week5.log")
    assert got["status"] == "NOT_RECOVERABLE"
    assert got["missing_fields"] == [
        "availability_timing_certification_agreement"
    ]
    assert got["certification_blocks_found"] == 2


def test_identical_repeated_blocks_are_recoverable(tmp_path):
    write_log(tmp_path, 5, cert_meta(), cert_meta())
    got = salvage_week(5, SOURCE, tmp_path / "week5.log")
    assert got["status"] == "RECOVERABLE"
    assert got["certification_blocks_found"] == 2
    assert "using the last" in " ".join(got["notes"])


def test_inconsistent_game_counts_fail_closed(tmp_path):
    write_log(tmp_path, 3, cert_meta(games=16, eligible=14, withheld=5))
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["status"] == "NOT_RECOVERABLE"
    assert got["missing_fields"] == ["game_count_consistency"]


@pytest.mark.parametrize(
    "field", ["asof_utc", "games", "eligible_games", "withheld_teams"]
)
def test_incomplete_certification_block_names_the_missing_field(tmp_path, field):
    meta = cert_meta()
    meta[field] = None
    write_log(tmp_path, 3, meta)
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["status"] == "NOT_RECOVERABLE"
    assert field in got["missing_fields"]


def test_mixed_withholding_keeps_the_blocked_set_but_not_the_reason(tmp_path):
    write_log(
        tmp_path,
        4,
        cert_meta(
            eligible=13,
            withheld=3,
            state_counts={
                "KICKED_OFF_LOCKED": 2,
                "REQUIRED_MISSING_FAIL_CLOSED": 1,
                "NOT_YET_REQUIRED": 13,
            },
            withheld_teams=("DEN", "LAR", "NE", "SEA", "SF", "WAS"),
        ),
    )
    got = salvage_week(4, SOURCE, tmp_path / "week4.log")
    # The blocked set is exact even though each team's reason is not.
    assert got["status"] == "RECOVERABLE"
    assert got["withheld_reason_recoverable"] is False
    assert [r["team"] for r in got["block_rules"]] == [
        "DEN",
        "LAR",
        "NE",
        "SEA",
        "SF",
        "WAS",
    ]
    assert all(
        r["reason"] == "WITHHELD_REASON_NOT_RECOVERABLE" for r in got["block_rules"]
    )


def test_absent_corroborating_facts_are_recorded_not_invented(tmp_path):
    write_log(tmp_path, 3, cert_meta())
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["corroborating_facts"]["lineage_version"] is None
    assert "lineage_version" in got["corroborating_facts_absent"]


def test_present_corroborating_facts_are_captured(tmp_path):
    write_log(
        tmp_path,
        3,
        cert_meta(),
        extra=(
            "sportsbook_current_roster_mismatches: 0\n"
            "temporary_possible_historical_aliases: 1\n"
            "Unresolved Position Rows 0\n"
            "lineage MARKET_MODEL_LINEAGE_V3_R22_WEEK1_PROMOTED ok\n"
        ),
    )
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    c = got["corroborating_facts"]
    assert c["sportsbook_current_roster_mismatches"] == "0"
    assert c["temporary_possible_historical_aliases"] == "1"
    assert c["core_unresolved_player_rows"] == "0"
    assert c["lineage_version"] == "MARKET_MODEL_LINEAGE_V3_R22_WEEK1_PROMOTED"
    assert got["corroborating_facts_absent"] == []


def test_a_failed_run_log_is_still_readable(tmp_path):
    """Week 4's run concluded 'failure' after certification had already run."""
    write_log(tmp_path, 4, cert_meta())
    source = dict(SOURCE, conclusion="failure")
    got = salvage_week(4, source, tmp_path / "week4.log")
    assert got["status"] == "RECOVERABLE"
    assert got["source_run_conclusion"] == "failure"


def test_zero_withheld_teams_produces_no_block_rules(tmp_path):
    write_log(
        tmp_path,
        3,
        cert_meta(
            eligible=16,
            withheld=0,
            state_counts={"REQUIRED_AND_CERTIFIED": 16},
            withheld_teams=(),
        ),
    )
    got = salvage_week(3, SOURCE, tmp_path / "week3.log")
    assert got["status"] == "RECOVERABLE"
    assert got["block_rules"] == []
    assert got["withheld_reason_recoverable"] is True
