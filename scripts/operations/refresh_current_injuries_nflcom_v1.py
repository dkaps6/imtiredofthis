#!/usr/bin/env python3
"""Refresh current football-only injury rows from NFL.com with explicit active-team scope.

Used by the Week-5 offline sportsbook replay after the preserved nflverse injury
snapshot was proven incomplete relative to the current NFL.com injury page.
No sportsbook input is read or written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.repair_injuries_nflcom_v1 as nflcom
from scripts.build.build_injuries_weekly import SCHEMA, detect_current_week_from_html, fetch_injuries_html
from scripts.runtime_context import resolve_season, resolve_week

DATA = Path("data")
AUDIT = DATA / "current_injury_nflcom_refresh_audit.json"


def refresh() -> dict:
    season = int(resolve_season())
    week = int(resolve_week())

    html = fetch_injuries_html()
    detected = int(detect_current_week_from_html(html))
    if detected != week:
        raise RuntimeError(f"NFL.com injury refresh week mismatch expected={week} detected={detected}")

    repaired = nflcom._parse_context_tables(html, season, week)
    nflcom._validate(repaired, season, week)
    repaired = repaired[SCHEMA].copy()
    scope = nflcom._parse_team_scope(html, season, week, repaired)

    repaired.to_csv(nflcom.INJURIES, index=False)

    audit_rows = repaired[["player", "team", "status", "practice_status", "body_part", "source"]].copy()
    audit_rows["team_identity_resolved"] = 1
    audit_rows.to_csv(nflcom.AUDIT, index=False)

    status = {
        "season": season,
        "week": week,
        "state": "official_report",
        "source": "nfl.com_context_refresh",
        "rows": int(len(repaired)),
        "successful_checks": ["nfl.com"],
        "provider_errors": [],
        "team_identity_repaired": True,
        "team_identity_unresolved_rows": 0,
        "all_scheduled_teams_checked": True,
        "scheduled_teams_checked": int(len(scope)),
        "teams_with_report_rows": int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum()),
        "teams_explicit_no_injuries_reported": int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        ),
        "scope_ledger": str(nflcom.SCOPE),
        "scope_basis": "live_nflcom_current_week_explicit_active_team_scope",
        "sportsbook_inputs_used": 0,
    }
    nflcom.STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "CURRENT_NFLCOM_INJURY_REFRESH_CERTIFIED",
        "season": season,
        "week": week,
        "rows": int(len(repaired)),
        "teams_with_report_rows": int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum()),
        "teams_explicit_no_injuries_reported": int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        ),
        "scheduled_teams_checked": int(len(scope)),
        "sportsbook_inputs_used": 0,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[current_injury_nflcom_refresh] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    refresh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
