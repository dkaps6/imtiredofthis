#!/usr/bin/env python3
"""Certify injury-team scope for an offline preserved Full Slate replay.

The paid run can contain a valid nflverse injury table whose positive rows cover
fewer than all scheduled teams.  That is not proof that the missing teams had no
injuries, so the production quality gate correctly fails closed.

For replay only, this adapter uses the existing NFL.com parser to prove the full
scheduled-team report scope while leaving the preserved nflverse injury rows
byte-for-byte unchanged.  It does not fetch or mutate sportsbook data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import scripts.repair_injuries_nflcom_v1 as nflcom
from scripts.runtime_context import resolve_season, resolve_week

AUDIT = Path("data/preserved_replay_injury_scope_audit.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def certify_scope() -> dict:
    season = int(resolve_season())
    week = int(resolve_week())

    if not nflcom.STATUS.exists() or nflcom.STATUS.stat().st_size == 0:
        raise RuntimeError("preserved replay missing injuries_source_status.json")
    if not nflcom.INJURIES.exists() or nflcom.INJURIES.stat().st_size == 0:
        raise RuntimeError("preserved replay missing injuries.csv")

    status = json.loads(nflcom.STATUS.read_text(encoding="utf-8"))
    if str(status.get("state", "")) != "official_report":
        raise RuntimeError(f"preserved replay injury state is not official_report: {status.get('state')}")

    before_hash = _sha256(nflcom.INJURIES)
    preserved = pd.read_csv(nflcom.INJURIES, low_memory=False)
    preserved_rows = int(len(preserved))
    preserved_teams = int(preserved["team"].nunique()) if "team" in preserved.columns else 0

    html = nflcom.fetch_injuries_html()
    detected = int(nflcom.detect_current_week_from_html(html))
    if detected != week:
        raise RuntimeError(f"NFL.com injury scope week mismatch expected={week} detected={detected}")

    source_rows = nflcom._parse_context_tables(html, season, week)
    nflcom._validate(source_rows, season, week)
    scope = nflcom._parse_team_scope(html, season, week, source_rows)

    after_hash = _sha256(nflcom.INJURIES)
    if before_hash != after_hash:
        raise RuntimeError("injury scope certification mutated preserved injuries.csv")

    status = dict(status)
    status.update({
        "all_scheduled_teams_checked": True,
        "scheduled_teams_checked": int(len(scope)),
        "teams_with_report_rows": int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum()),
        "teams_explicit_no_injuries_reported": int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        ),
        "scope_basis": "nflcom_current_week_explicit_team_scope_for_preserved_replay",
        "scope_source": "nfl.com",
        "scope_ledger": str(nflcom.SCOPE),
        "preserved_injury_rows_source": str(status.get("source", "")),
    })
    nflcom.STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "PRESERVED_INJURY_ROWS_SCOPE_CERTIFIED",
        "season": season,
        "week": week,
        "preserved_rows": preserved_rows,
        "preserved_teams_with_rows": preserved_teams,
        "preserved_injuries_sha256": before_hash,
        "scheduled_teams_checked": int(len(scope)),
        "teams_with_report_rows": int(status["teams_with_report_rows"]),
        "teams_explicit_no_injuries_reported": int(status["teams_explicit_no_injuries_reported"]),
        "sportsbook_inputs_used": False,
        "injury_rows_mutated": False,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[preserved_injury_scope] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    certify_scope()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
