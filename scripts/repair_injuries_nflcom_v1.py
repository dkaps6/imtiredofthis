#!/usr/bin/env python3
"""Repair NFL.com injury identity and certify scheduled-team report scope.

NFL.com renders team labels immediately outside injury tables. Pandas ``read_html``
therefore sees player rows but not club identity, and teams with an explicit
"No Injuries Reported" state have no table at all. This adapter reattaches team
identity and writes a 32-team scope ledger. It never infers that an absent row is
a healthy player; it records only what the official source explicitly reports.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scripts._opponent_map import CANON_TEAM_CODES, TEAM_NICKNAME_TO_ABBR, canon_team
from scripts.build.build_injuries_weekly import (
    SCHEMA,
    detect_current_week_from_html,
    fetch_injuries_html,
    normalize_nflcom_dataframe,
)
from scripts.runtime_context import resolve_season, resolve_week

DATA = Path("data")
INJURIES = DATA / "injuries.csv"
STATUS = DATA / "injuries_source_status.json"
AUDIT = DATA / "injuries_identity_audit.csv"
SCOPE = DATA / "injury_team_scope.csv"
SCHEDULE = DATA / "team_week_map.csv"


def _clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _nickname_map() -> dict[str, str]:
    out = {str(k).strip().lower(): str(v) for k, v in TEAM_NICKNAME_TO_ABBR.items()}
    out.setdefault("49ers", "SF")
    return out


def _scheduled_teams(season: int, week: int) -> set[str]:
    if not SCHEDULE.exists() or SCHEDULE.stat().st_size == 0:
        raise RuntimeError("team_week_map missing before injury scope certification")
    sched = pd.read_csv(SCHEDULE, low_memory=False)
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "team"}
    missing = required - set(sched.columns)
    if missing:
        raise RuntimeError(f"team_week_map missing injury scope columns: {sorted(missing)}")
    active = sched.loc[
        pd.to_numeric(sched["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(sched["week"], errors="coerce").eq(int(week))
    ].copy()
    teams = set(active["team"].map(canon_team).dropna().astype(str))
    if len(teams) != 32:
        raise RuntimeError(f"injury scope requires 32 scheduled teams; got {len(teams)}")
    return teams


def _nearest_team_label(table, labels: dict[str, str]) -> str:
    for node in table.find_all_previous(string=True, limit=250):
        text = " ".join(str(node).split()).strip()
        if not text:
            continue
        team = labels.get(text.lower(), "")
        if team:
            return canon_team(team)
    return ""


def _parse_context_tables(html: str, season: int, week: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    labels = _nickname_map()
    parts: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        try:
            parsed = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        columns = {str(c).strip().lower() for c in parsed.columns}
        if not any("player" in c for c in columns):
            continue
        if not any("status" in c for c in columns):
            continue
        team = _nearest_team_label(table, labels)
        if not team:
            continue
        parsed = parsed.copy()
        parsed["team_raw"] = team
        normalized = normalize_nflcom_dataframe(parsed, season, week)
        if not normalized.empty:
            parts.append(normalized)
    if not parts:
        return pd.DataFrame(columns=SCHEMA)
    return pd.concat(parts, ignore_index=True).drop_duplicates(["player", "team"], keep="last")


def _parse_team_scope(html: str, season: int, week: int, injuries: pd.DataFrame) -> pd.DataFrame:
    """Account for every scheduled team using explicit NFL.com page states."""
    scheduled = _scheduled_teams(season, week)
    labels = _nickname_map()
    soup = BeautifulSoup(html, "html.parser")

    # Walk source text in DOM order. A team label starts a team section; the
    # explicit marker belongs to that section until the next recognized team.
    found_labels: set[str] = set()
    no_report_markers: set[str] = set()
    current_team = ""
    for node in soup.find_all(string=True):
        text = " ".join(str(node).split()).strip()
        if not text:
            continue
        maybe_team = canon_team(labels.get(text.lower(), "")) if text.lower() in labels else ""
        if maybe_team in scheduled:
            current_team = maybe_team
            found_labels.add(maybe_team)
            continue
        if current_team and text.lower() == "no injuries reported":
            no_report_markers.add(current_team)

    row_counts = (
        injuries.assign(team=injuries["team"].map(canon_team)).groupby("team").size().to_dict()
        if not injuries.empty
        else {}
    )
    report_teams = {str(t) for t, n in row_counts.items() if int(n) > 0}
    records: list[dict] = []
    unresolved: list[str] = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for team in sorted(scheduled):
        rows = int(row_counts.get(team, 0))
        label_found = team in found_labels
        explicit_none = team in no_report_markers
        if rows > 0 and label_found:
            state = "OFFICIAL_REPORT_ROWS"
        elif rows == 0 and label_found and explicit_none:
            state = "NO_INJURIES_REPORTED_BY_SOURCE"
        else:
            state = "SCOPE_UNRESOLVED"
            unresolved.append(team)
        records.append({
            "season": int(season),
            "week": int(week),
            "team": team,
            "scope_state": state,
            "injury_rows": rows,
            "page_team_label_found": int(label_found),
            "no_injuries_reported_marker": int(explicit_none),
            "source": "nfl.com",
            "checked_at_utc": checked_at,
        })

    scope = pd.DataFrame(records)
    if unresolved:
        raise RuntimeError(f"NFL.com injury scope unresolved scheduled teams: {unresolved}")
    if len(scope) != 32 or set(scope["team"]) != scheduled:
        raise RuntimeError("NFL.com injury scope ledger does not exactly match scheduled teams")
    if report_teams - scheduled:
        raise RuntimeError(f"NFL.com injury rows include non-scheduled teams: {sorted(report_teams - scheduled)}")
    scope.to_csv(SCOPE, index=False)
    return scope


def _validate(df: pd.DataFrame, season: int, week: int) -> None:
    if df.empty:
        raise RuntimeError("NFL.com injury context repair produced zero rows")
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise RuntimeError(f"repaired injuries missing columns: {sorted(missing)}")
    teams = df["team"].map(canon_team)
    bad_team = ~teams.isin(CANON_TEAM_CODES)
    if bad_team.any():
        sample = df.loc[bad_team, ["player", "team"]].head(20).to_dict("records")
        raise RuntimeError(f"repaired injuries contain unresolved teams: {sample}")
    if not pd.to_numeric(df["season"], errors="coerce").eq(int(season)).all():
        raise RuntimeError("repaired injuries contain stale season")
    if not pd.to_numeric(df["week"], errors="coerce").eq(int(week)).all():
        raise RuntimeError("repaired injuries contain stale week")


def repair_if_needed(season: int | None = None, week: int | None = None) -> dict:
    season = int(season if season is not None else resolve_season())
    week = int(week if week is not None else resolve_week())
    if not STATUS.exists() or STATUS.stat().st_size == 0:
        raise RuntimeError("injuries_source_status.json missing; injury provider state unknown")
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    state = str(status.get("state", "")).strip()
    source = str(status.get("source", "")).strip().lower()

    current = pd.read_csv(INJURIES) if INJURIES.exists() and INJURIES.stat().st_size else pd.DataFrame(columns=SCHEMA)
    team_missing = True
    if not current.empty and "team" in current.columns:
        team_missing = current["team"].map(_clean).eq("").any()

    if state != "official_report":
        result = {
            "disposition": "INJURY_IDENTITY_NOT_REQUIRED",
            "state": state,
            "source": source,
            "rows": int(len(current)),
        }
        print("[injury_identity] " + json.dumps(result, sort_keys=True))
        return result

    # NFL.com context is also needed to prove teams that explicitly have no rows.
    if "nfl.com" in source or team_missing:
        html = fetch_injuries_html()
        detected = int(detect_current_week_from_html(html))
        if detected != week:
            raise RuntimeError(f"NFL.com injury repair week mismatch expected={week} detected={detected}")
        repaired = _parse_context_tables(html, season, week)
        _validate(repaired, season, week)
        repaired = repaired[SCHEMA].copy()
        scope = _parse_team_scope(html, season, week, repaired)
        repaired.to_csv(INJURIES, index=False)

        audit = repaired[["player", "team", "status", "practice_status", "body_part", "source"]].copy()
        audit["team_identity_resolved"] = 1
        audit.to_csv(AUDIT, index=False)

        status["source"] = "nfl.com_context_repair"
        status["rows"] = int(len(repaired))
        status["team_identity_repaired"] = True
        status["team_identity_unresolved_rows"] = 0
        status["all_scheduled_teams_checked"] = True
        status["scheduled_teams_checked"] = int(len(scope))
        status["teams_with_report_rows"] = int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum())
        status["teams_explicit_no_injuries_reported"] = int(
            scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum()
        )
        status["scope_ledger"] = str(SCOPE)
        STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {
            "disposition": "INJURY_IDENTITY_AND_SCOPE_CERTIFIED",
            "state": "official_report",
            "source": status["source"],
            "rows": int(len(repaired)),
            "teams_with_rows": int(repaired["team"].nunique()),
            "scheduled_teams_checked": int(len(scope)),
            "teams_explicit_no_injuries_reported": int(status["teams_explicit_no_injuries_reported"]),
        }
        print("[injury_identity] " + json.dumps(result, sort_keys=True))
        return result

    if not team_missing:
        _validate(current, season, week)
        result = {
            "disposition": "INJURY_IDENTITY_READY_SCOPE_UNPROVEN",
            "state": state,
            "source": source,
            "rows": int(len(current)),
        }
        print("[injury_identity] " + json.dumps(result, sort_keys=True))
        return result

    raise RuntimeError(f"Official injury report contains unresolved team identity from unsupported source={source}")


def main() -> int:
    repair_if_needed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
