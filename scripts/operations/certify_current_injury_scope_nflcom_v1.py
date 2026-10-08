#!/usr/bin/env python3
"""Certify current injury-team scope without mutating preserved nflverse injury rows.

The current Full Slate injury table may contain rows only for teams with listed
injuries.  For production certification we separately prove that every active
scheduled team was represented by the official NFL.com weekly injury page, and
that teams absent from the preserved nflverse row set are explicitly marked
"No Injuries Reported" by NFL.com.

This is a football-provider scope check only.  It never calls a sportsbook and
never changes data/injuries.csv.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scripts._opponent_map import TEAM_NICKNAME_TO_ABBR, canon_team
from scripts.build.build_injuries_weekly import detect_current_week_from_html, fetch_injuries_html
from scripts.runtime_context import resolve_season, resolve_week

DATA = Path("data")
INJURIES = DATA / "injuries.csv"
STATUS = DATA / "injuries_source_status.json"
SCHEDULE = DATA / "team_week_map.csv"
SCOPE = DATA / "injury_team_scope.csv"
AUDIT = DATA / "current_injury_scope_nflcom_audit.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scheduled_teams(season: int, week: int) -> set[str]:
    if not SCHEDULE.exists() or SCHEDULE.stat().st_size <= 0:
        raise RuntimeError("team_week_map missing for injury scope certification")
    sched = pd.read_csv(SCHEDULE, low_memory=False)
    required = {"season", "week", "team"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"team_week_map missing columns: {sorted(missing)}")
    active = sched.loc[
        pd.to_numeric(sched["season"], errors="coerce").eq(season)
        & pd.to_numeric(sched["week"], errors="coerce").eq(week)
    ].copy()
    teams = set(active["team"].map(canon_team).dropna().astype(str))
    if not teams or len(teams) % 2:
        raise RuntimeError(f"active injury scope requires even nonzero team count; got {len(teams)}")
    return teams


def _nickname_map() -> dict[str, str]:
    out = {str(k).strip().lower(): canon_team(v) for k, v in TEAM_NICKNAME_TO_ABBR.items()}
    out.setdefault("49ers", "SF")
    return out


def _page_scope(html: str, scheduled: set[str]) -> tuple[set[str], set[str]]:
    labels = _nickname_map()
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()
    explicit_none: set[str] = set()
    current = ""
    for node in soup.find_all(string=True):
        text = " ".join(str(node).split()).strip()
        if not text:
            continue
        key = text.lower()
        maybe = labels.get(key, "")
        if maybe in scheduled:
            current = maybe
            found.add(maybe)
            continue
        if current and key == "no injuries reported":
            explicit_none.add(current)
    return found, explicit_none


def certify() -> dict:
    season = int(resolve_season())
    week = int(resolve_week())
    if not INJURIES.exists() or INJURIES.stat().st_size <= 0:
        raise RuntimeError("injuries.csv missing/empty")
    if not STATUS.exists() or STATUS.stat().st_size <= 0:
        raise RuntimeError("injuries_source_status.json missing/empty")

    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if str(status.get("state", "")) != "official_report":
        raise RuntimeError(f"injury provider state not official_report: {status.get('state')}")

    preserved_hash = _sha256(INJURIES)
    injuries = pd.read_csv(INJURIES, low_memory=False)
    if "team" not in injuries.columns:
        raise RuntimeError("injuries.csv missing team column")
    injuries["team"] = injuries["team"].map(canon_team)
    scheduled = _scheduled_teams(season, week)
    off_schedule = sorted(set(injuries["team"].dropna().astype(str)) - scheduled)
    if off_schedule:
        raise RuntimeError(f"preserved injury rows contain teams outside active schedule: {off_schedule}")

    html = fetch_injuries_html()
    detected = int(detect_current_week_from_html(html))
    if detected != week:
        raise RuntimeError(f"NFL.com injury page week mismatch expected={week} detected={detected}")
    found, explicit_none = _page_scope(html, scheduled)

    counts = injuries.groupby("team").size().astype(int).to_dict()
    records = []
    unresolved = []
    now = datetime.now(timezone.utc).isoformat()
    for team in sorted(scheduled):
        rows = int(counts.get(team, 0))
        label_found = team in found
        none = team in explicit_none
        if rows > 0 and label_found:
            state = "OFFICIAL_REPORT_ROWS"
        elif rows == 0 and label_found and none:
            state = "NO_INJURIES_REPORTED_BY_SOURCE"
        else:
            state = "SCOPE_UNRESOLVED"
            unresolved.append({
                "team": team,
                "preserved_injury_rows": rows,
                "page_team_label_found": label_found,
                "no_injuries_reported_marker": none,
            })
        records.append({
            "season": season,
            "week": week,
            "team": team,
            "scope_state": state,
            "injury_rows": rows,
            "page_team_label_found": int(label_found),
            "no_injuries_reported_marker": int(none),
            "source": "nfl.com_scope_for_preserved_nflverse_rows",
            "checked_at_utc": now,
        })

    if unresolved:
        raise RuntimeError(f"NFL.com scope cannot certify preserved injury rows: {unresolved}")

    scope = pd.DataFrame(records)
    if len(scope) != len(scheduled) or set(scope["team"]) != scheduled:
        raise RuntimeError("injury scope ledger does not exactly match active schedule")
    scope.to_csv(SCOPE, index=False)

    if _sha256(INJURIES) != preserved_hash:
        raise RuntimeError("injury scope certification mutated preserved injuries.csv")

    status = dict(status)
    status.update({
        "all_scheduled_teams_checked": True,
        "scheduled_teams_checked": int(len(scope)),
        "teams_with_report_rows": int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum()),
        "teams_explicit_no_injuries_reported": int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        ),
        "scope_basis": "live_nflcom_explicit_active_team_scope_for_preserved_nflverse_rows",
        "scope_source": "nfl.com",
        "scope_ledger": str(SCOPE),
        "scope_checked_at_utc": now,
    })
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "CURRENT_INJURY_SCOPE_CERTIFIED_PRESERVED_ROWS_UNCHANGED",
        "season": season,
        "week": week,
        "scheduled_teams_checked": len(scope),
        "teams_with_report_rows": int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum()),
        "teams_explicit_no_injuries_reported": int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        ),
        "preserved_injuries_sha256": preserved_hash,
        "injury_rows_mutated": False,
        "sportsbook_inputs_used": False,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[current_injury_scope] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    certify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
