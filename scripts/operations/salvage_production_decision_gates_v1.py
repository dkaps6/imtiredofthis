#!/usr/bin/env python3
"""Salvage pregame production decision-gate evidence from surviving run logs.

The canonical gate file (``data/market_track_record/PRODUCTION_DECISION_GATES_V1.json``)
covers 2026 weeks 1-2 only. Weeks 3-5 lost their priced boards when
``full-slate.yml`` uploaded them with ``retention-days: 7``, and the gate
evidence was never made durable either.

Actions *logs* outlive artifacts, and
``scripts/validate_current_player_availability_timing_v1.py`` prints its whole
certification meta to stdout. That printed block is the authoritative pregame
record of which games were already kicked off and which were failed closed, so
for any week whose log survives the gate facts can be read back exactly rather
than reconstructed by inference.

This reader is deliberately fail-closed per field. It reports what a log
actually contains and never fills a gap: a week with a missing or internally
inconsistent fact is reported ``NOT_RECOVERABLE`` with the missing fields named,
not completed from a neighbouring week, the schedule, or the priced board.

It writes a candidate file for audit. It does not mutate the canonical gate file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Each log line carries an Actions timestamp prefix that has to come off before
# the payload parses as JSON.
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?")

# Actions logs also carry ANSI colour codes. The certification meta is printed
# uncoloured, but stripping them unconditionally keeps a coloured line from
# ever hiding a brace from the block scanner.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Unique signature of the certification meta block printed by
# validate_current_player_availability_timing_v1.main().
_CERT_KEY = "require_minutes_before_kickoff"

# Facts the timing certification block must carry for a week to be recoverable.
_REQUIRED_CERT_FIELDS = (
    "asof_utc",
    "games",
    "eligible_games",
    "withheld_games",
    "state_counts",
    "withheld_teams",
)

# Optional corroborating facts emitted by other validators. Absence is recorded,
# never invented.
_OPTIONAL_PATTERNS = {
    "sportsbook_current_roster_mismatches": re.compile(
        r"sportsbook_current_roster_mismatches\D{0,12}(\d+)"
    ),
    "temporary_possible_historical_aliases": re.compile(
        r"temporary_possible_historical_aliases\D{0,12}(\d+)"
    ),
    "core_unresolved_player_rows": re.compile(
        r"(?:core_unresolved_player_rows|Unresolved Position Rows)\D{0,12}(\d+)"
    ),
    "lineage_version": re.compile(r"(MARKET_MODEL_LINEAGE_V\d+[A-Z0-9_]*)"),
}


def strip_timestamps(text: str) -> str:
    return "\n".join(
        _TS.sub("", _ANSI.sub("", line)) for line in text.splitlines()
    )


def extract_json_blocks(text: str, key: str) -> list[dict[str, Any]]:
    """Return every top-level JSON object in ``text`` that contains ``key``.

    The certification meta is printed with ``indent=2``, so a block starts at a
    line that is exactly ``{`` and ends at the matching ``}`` in column zero.
    Scanning for brace depth rather than regex keeps nested objects such as
    ``state_counts`` intact.
    """
    blocks: list[dict[str, Any]] = []
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if line.strip() != "{":
            continue
        depth = 0
        for end in range(start, len(lines)):
            depth += lines[end].count("{") - lines[end].count("}")
            if depth != 0:
                continue
            chunk = "\n".join(lines[start : end + 1])
            if key in chunk:
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    break
                if isinstance(parsed, dict) and key in parsed:
                    blocks.append(parsed)
            break
    return blocks


def resolve_block_reason(state_counts: dict[str, Any]) -> tuple[str, bool]:
    """Classify why teams were withheld, or refuse to classify.

    A team is withheld either because its game had already kicked off or because
    the T-75 official-inactive section failed closed. Both block the team, but
    they are different reasons and the log only decomposes them when exactly one
    non-eligible state occurred. With a mix, the blocked set stays exact while
    the per-team reason is not recoverable, so it is not guessed.
    """
    noneligible = {
        state: int(count)
        for state, count in state_counts.items()
        if state not in ("NOT_YET_REQUIRED", "REQUIRED_AND_CERTIFIED")
        and int(count) > 0
    }
    if not noneligible:
        return "NO_TEAMS_WITHHELD", True
    if len(noneligible) == 1:
        return next(iter(noneligible)), True
    return "WITHHELD_REASON_NOT_RECOVERABLE", False


def salvage_week(week: int, source: dict[str, str], log_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "week": week,
        "source_run_id": source["run_id"],
        "source_git_sha": source["git_sha"],
        "source_job_id": source["job_id"],
        "source_run_conclusion": source["conclusion"],
        "source_run_created_at": source["created_at"],
        "log_available": log_path.exists() and log_path.stat().st_size > 0,
        "missing_fields": [],
        "notes": [],
    }
    if not result["log_available"]:
        result["status"] = "NOT_RECOVERABLE"
        result["missing_fields"] = ["run_log"]
        result["notes"].append("run log absent or empty; nothing to read back")
        return result

    raw = log_path.read_text(encoding="utf-8", errors="replace")
    text = strip_timestamps(raw)
    result["log_lines"] = len(text.splitlines())

    blocks = extract_json_blocks(text, _CERT_KEY)
    result["certification_blocks_found"] = len(blocks)
    if not blocks:
        result["status"] = "NOT_RECOVERABLE"
        result["missing_fields"] = ["availability_timing_certification"]
        result["notes"].append(
            "no availability timing certification block printed in this log; "
            "pregame lock state cannot be read back and must not be inferred"
        )
        return result

    # A run can certify more than once. Only a single agreed answer is usable;
    # disagreement means the operative pregame state is ambiguous.
    final = blocks[-1]
    if len(blocks) > 1:
        signature = {
            json.dumps(
                {k: b.get(k) for k in _REQUIRED_CERT_FIELDS}, sort_keys=True
            )
            for b in blocks
        }
        if len(signature) > 1:
            result["status"] = "NOT_RECOVERABLE"
            result["missing_fields"] = ["availability_timing_certification_agreement"]
            result["notes"].append(
                f"{len(blocks)} certification blocks disagree on the required "
                "fields; the operative pregame gate is ambiguous and is not guessed"
            )
            return result
        result["notes"].append(
            f"{len(blocks)} identical certification blocks; using the last"
        )

    missing = [f for f in _REQUIRED_CERT_FIELDS if final.get(f) is None]
    if missing:
        result["status"] = "NOT_RECOVERABLE"
        result["missing_fields"] = missing
        result["notes"].append("certification block present but incomplete")
        return result

    state_counts = {str(k): int(v) for k, v in dict(final["state_counts"]).items()}
    withheld_teams = sorted({str(t) for t in final["withheld_teams"]})
    reason, reason_ok = resolve_block_reason(state_counts)

    result["source_facts"] = {
        "availability_asof_utc": str(final["asof_utc"]),
        "game_count": int(final["games"]),
        "eligible_games": int(final["eligible_games"]),
        "withheld_games": int(final["withheld_games"]),
        "require_minutes_before_kickoff": final.get(
            "require_minutes_before_kickoff"
        ),
        "sportsbook_inputs_used": final.get("sportsbook_inputs_used"),
        "certification_state_counts": state_counts,
        "withheld_teams": withheld_teams,
    }
    result["block_rules"] = [
        {"scope": "TEAM_ALL", "team": team, "reason": reason}
        for team in withheld_teams
    ]
    result["withheld_reason_recoverable"] = reason_ok
    if not reason_ok:
        result["notes"].append(
            "more than one non-eligible certification state occurred, so the "
            "blocked team set is exact but each team's reason is not recoverable"
        )

    # Internal consistency: the blocked set has to square with the game counts.
    if int(final["eligible_games"]) + int(final["withheld_games"]) != int(
        final["games"]
    ):
        result["status"] = "NOT_RECOVERABLE"
        result["missing_fields"] = ["game_count_consistency"]
        result["notes"].append(
            f"eligible({final['eligible_games']}) + withheld"
            f"({final['withheld_games']}) != games({final['games']})"
        )
        return result

    corroborating: dict[str, Any] = {}
    for name, pattern in _OPTIONAL_PATTERNS.items():
        hits = pattern.findall(text)
        corroborating[name] = hits[-1] if hits else None
    result["corroborating_facts"] = corroborating
    result["corroborating_facts_absent"] = sorted(
        name for name, value in corroborating.items() if value is None
    )

    result["status"] = "RECOVERABLE"
    result["evidence_basis"] = (
        "Printed availability timing certification meta from the surviving "
        f"Full Slate run log (run {source['run_id']}, job {source['job_id']}, "
        f"git {source['git_sha']})."
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument("--candidate-out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    season = int(manifest["season"])

    weeks: dict[str, Any] = {}
    for week_key, source in sorted(
        manifest["weeks"].items(), key=lambda kv: int(kv[0])
    ):
        log_path = args.logs_dir / f"week{int(week_key)}.log"
        weeks[week_key] = salvage_week(int(week_key), source, log_path)

    recoverable = sorted(k for k, v in weeks.items() if v["status"] == "RECOVERABLE")
    blocked = sorted(k for k, v in weeks.items() if v["status"] != "RECOVERABLE")

    report = {
        "version": "PRODUCTION_DECISION_GATE_SALVAGE_V1",
        "season": season,
        "purpose": (
            "Read-back audit of what pregame production decision-gate evidence "
            "survives in Actions run logs for weeks whose priced-board "
            "artifacts expired. Fail-closed per field; nothing is inferred."
        ),
        "recoverable_weeks": recoverable,
        "not_recoverable_weeks": blocked,
        "weeks": weeks,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    candidate = {
        "version": "PRODUCTION_DECISION_GATES_SALVAGE_CANDIDATE_V1",
        "status": "AWAITING_AUDIT",
        "purpose": (
            "Candidate gate entries salvaged from surviving run logs. These are "
            "NOT canonical and must not be consumed by the grader until audited "
            "and promoted into PRODUCTION_DECISION_GATES_V1.json."
        ),
        "season": season,
        "weeks": {
            week: {
                "source_run_id": weeks[week]["source_run_id"],
                "source_git_sha": weeks[week]["source_git_sha"],
                "source_job_id": weeks[week]["source_job_id"],
                "evidence_basis": weeks[week]["evidence_basis"],
                "source_facts": weeks[week]["source_facts"],
                "block_rules": weeks[week]["block_rules"],
                "withheld_reason_recoverable": weeks[week][
                    "withheld_reason_recoverable"
                ],
                "corroborating_facts": weeks[week]["corroborating_facts"],
            }
            for week in recoverable
        },
    }
    args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_out.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=== PRODUCTION DECISION GATE SALVAGE ===")
    print(f"  season: {season}")
    for week_key in sorted(weeks, key=int):
        w = weeks[week_key]
        print(f"\n  week {week_key}: {w['status']}")
        print(
            f"    run {w['source_run_id']} ({w['source_run_conclusion']}, "
            f"{w['source_run_created_at']}) job {w['source_job_id']}"
        )
        print(
            f"    log available: {w['log_available']}"
            + (f", {w['log_lines']} lines" if w.get("log_lines") else "")
        )
        if w["status"] == "RECOVERABLE":
            f = w["source_facts"]
            print(f"    asof_utc            : {f['availability_asof_utc']}")
            print(
                f"    games               : {f['game_count']} "
                f"(eligible {f['eligible_games']}, withheld {f['withheld_games']})"
            )
            print(f"    state counts        : {f['certification_state_counts']}")
            print(
                f"    withheld teams ({len(f['withheld_teams'])}): "
                f"{', '.join(f['withheld_teams']) or 'none'}"
            )
            print(
                f"    reason recoverable  : {w['withheld_reason_recoverable']}"
                f" -> {w['block_rules'][0]['reason'] if w['block_rules'] else 'n/a'}"
            )
            if w["corroborating_facts_absent"]:
                print(
                    "    corroborating absent: "
                    + ", ".join(w["corroborating_facts_absent"])
                )
        else:
            print(f"    missing: {', '.join(w['missing_fields'])}")
        for note in w["notes"]:
            print(f"    note: {note}")

    print(
        f"\n  recoverable: {recoverable or 'none'}"
        f"\n  not recoverable: {blocked or 'none'}"
    )
    print(f"\nwrote {args.report_out}")
    print(f"wrote {args.candidate_out}")
    print(
        "\nThe candidate file is AWAITING_AUDIT and is not consumed by the "
        "grader. Promotion into PRODUCTION_DECISION_GATES_V1.json is a "
        "separate, audited step."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
