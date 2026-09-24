#!/usr/bin/env python3
"""Certify injury-team scope for an offline preserved Full Slate replay.

Historical replay must be deterministic.  This adapter therefore consumes the
Week-2 NFL.com scope ledger preserved by a previously successful replay and
pinned in an immutable Git commit.  It validates that evidence against the
preserved nflverse injury rows and the replay's authoritative schedule while
leaving injuries.csv byte-for-byte unchanged.

No live NFL.com request and no sportsbook request is made here.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import pandas as pd

from scripts._opponent_map import canon_team
import scripts.repair_injuries_nflcom_v1 as nflcom
from scripts.runtime_context import resolve_season, resolve_week

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path("data/preserved_replay_injury_scope_audit.json")
PINNED_EVIDENCE_COMMIT = "8839a34e549c04a3d688fc6437a032e386a216fb"
PINNED_SCOPE_PATH = "data/preserved_replay/WEEK2_INJURY_SCOPE_2026_WK02.csv"
PINNED_PROVENANCE_PATH = "data/preserved_replay/WEEK2_INJURY_SCOPE_PROVENANCE_V1.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_show_bytes(path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{PINNED_EVIDENCE_COMMIT}:{path}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"unable to read pinned replay evidence {PINNED_EVIDENCE_COMMIT}:{path}: {detail}"
        )
    return proc.stdout


def _load_pinned_scope() -> tuple[pd.DataFrame, dict, bytes]:
    provenance_bytes = _git_show_bytes(PINNED_PROVENANCE_PATH)
    provenance = json.loads(provenance_bytes.decode("utf-8"))
    scope_bytes = _git_show_bytes(PINNED_SCOPE_PATH)
    expected_scope_hash = str(provenance.get("scope_csv_sha256", ""))
    actual_scope_hash = _sha256_bytes(scope_bytes)
    if actual_scope_hash != expected_scope_hash:
        raise RuntimeError(
            "pinned Week-2 injury scope digest mismatch: "
            f"{actual_scope_hash} != {expected_scope_hash}"
        )
    scope = pd.read_csv(io.BytesIO(scope_bytes), low_memory=False)
    return scope, provenance, scope_bytes


def _validate_pinned_scope(
    scope: pd.DataFrame,
    provenance: dict,
    *,
    season: int,
    week: int,
) -> pd.DataFrame:
    if int(provenance.get("season", -1)) != int(season):
        raise RuntimeError(
            f"pinned injury scope season mismatch expected={season} "
            f"evidence={provenance.get('season')}"
        )
    if int(provenance.get("week", -1)) != int(week):
        raise RuntimeError(
            f"pinned injury scope week mismatch expected={week} "
            f"evidence={provenance.get('week')}"
        )
    if str(provenance.get("source", "")).strip().lower() != "nfl.com":
        raise RuntimeError("pinned injury scope does not identify NFL.com as official source")

    required = {
        "season",
        "week",
        "team",
        "scope_state",
        "injury_rows",
        "page_team_label_found",
        "no_injuries_reported_marker",
        "source",
    }
    scope = scope.copy()
    scope.columns = [str(c).strip().lower() for c in scope.columns]
    missing = required - set(scope.columns)
    if missing:
        raise RuntimeError(f"pinned injury scope missing columns: {sorted(missing)}")
    if scope.empty:
        raise RuntimeError("pinned injury scope is empty")

    scope_season = pd.to_numeric(scope["season"], errors="coerce")
    scope_week = pd.to_numeric(scope["week"], errors="coerce")
    if not scope_season.eq(int(season)).all() or not scope_week.eq(int(week)).all():
        raise RuntimeError("pinned injury scope contains out-of-scope season/week rows")

    scope["team"] = scope["team"].map(canon_team)
    scheduled = nflcom._scheduled_teams(season, week)
    teams = set(scope["team"].dropna().astype(str))
    if len(scope) != 32 or teams != scheduled:
        raise RuntimeError(
            "pinned injury scope does not exactly match the authoritative scheduled-team set"
        )
    if scope["team"].duplicated().any():
        raise RuntimeError("pinned injury scope contains duplicate teams")

    allowed_states = {"OFFICIAL_REPORT_ROWS", "NO_INJURIES_REPORTED_BY_SOURCE"}
    states = set(scope["scope_state"].astype(str))
    if not states <= allowed_states:
        raise RuntimeError(f"pinned injury scope contains unresolved/unknown states: {sorted(states)}")

    source_values = set(scope["source"].astype(str).str.strip().str.lower())
    if source_values != {"nfl.com"}:
        raise RuntimeError(f"pinned injury scope source drift: {sorted(source_values)}")

    injury_rows = pd.to_numeric(scope["injury_rows"], errors="coerce")
    if injury_rows.isna().any() or injury_rows.lt(0).any():
        raise RuntimeError("pinned injury scope has invalid injury_rows")
    has_rows = scope["scope_state"].eq("OFFICIAL_REPORT_ROWS")
    explicit_none = scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE")
    if (has_rows & injury_rows.le(0)).any():
        raise RuntimeError("pinned injury scope marks report rows without positive injury_rows")
    if (explicit_none & injury_rows.ne(0)).any():
        raise RuntimeError("pinned injury scope marks no-injury state with nonzero injury_rows")

    with_rows = int(has_rows.sum())
    with_none = int(explicit_none.sum())
    if with_rows != int(provenance.get("teams_with_report_rows", -1)):
        raise RuntimeError("pinned injury scope report-row team count drift")
    if with_none != int(provenance.get("teams_explicit_no_injuries_reported", -1)):
        raise RuntimeError("pinned injury scope explicit-none team count drift")
    if len(scope) != int(provenance.get("scheduled_teams_checked", -1)):
        raise RuntimeError("pinned injury scope scheduled-team count drift")
    return scope


def certify_scope() -> dict:
    season = int(resolve_season())
    week = int(resolve_week())

    if not nflcom.STATUS.exists() or nflcom.STATUS.stat().st_size == 0:
        raise RuntimeError("preserved replay missing injuries_source_status.json")
    if not nflcom.INJURIES.exists() or nflcom.INJURIES.stat().st_size == 0:
        raise RuntimeError("preserved replay missing injuries.csv")

    status = json.loads(nflcom.STATUS.read_text(encoding="utf-8"))
    if str(status.get("state", "")) != "official_report":
        raise RuntimeError(
            f"preserved replay injury state is not official_report: {status.get('state')}"
        )

    before_hash = _sha256(nflcom.INJURIES)
    preserved = pd.read_csv(nflcom.INJURIES, low_memory=False)
    preserved_rows = int(len(preserved))
    preserved_teams = int(preserved["team"].nunique()) if "team" in preserved.columns else 0

    scope, provenance, scope_bytes = _load_pinned_scope()
    expected_injury_hash = str(provenance.get("preserved_injuries_sha256", ""))
    if before_hash != expected_injury_hash:
        raise RuntimeError(
            "preserved injuries.csv does not match the injury rows certified by "
            f"the successful Week-2 replay: {before_hash} != {expected_injury_hash}"
        )
    scope = _validate_pinned_scope(scope, provenance, season=season, week=week)

    nflcom.SCOPE.parent.mkdir(parents=True, exist_ok=True)
    nflcom.SCOPE.write_bytes(scope_bytes)

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
        "scope_basis": "pinned_successful_week2_replay_scope_evidence",
        "scope_source": "nfl.com",
        "scope_ledger": str(nflcom.SCOPE),
        "scope_evidence_commit": PINNED_EVIDENCE_COMMIT,
        "scope_evidence_replay_run_id": str(provenance.get("source_replay_run_id", "")),
        "scope_evidence_artifact_id": str(provenance.get("source_artifact_id", "")),
        "preserved_injury_rows_source": str(status.get("source", "")),
    })
    nflcom.STATUS.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = {
        "disposition": "PRESERVED_INJURY_ROWS_SCOPE_CERTIFIED",
        "season": season,
        "week": week,
        "preserved_rows": preserved_rows,
        "preserved_teams_with_rows": preserved_teams,
        "preserved_injuries_sha256": before_hash,
        "scheduled_teams_checked": int(len(scope)),
        "teams_with_report_rows": int(status["teams_with_report_rows"]),
        "teams_explicit_no_injuries_reported": int(
            status["teams_explicit_no_injuries_reported"]
        ),
        "scope_evidence_commit": PINNED_EVIDENCE_COMMIT,
        "scope_evidence_replay_run_id": str(provenance.get("source_replay_run_id", "")),
        "scope_evidence_artifact_id": str(provenance.get("source_artifact_id", "")),
        "scope_ledger_sha256": _sha256_bytes(scope_bytes),
        "sportsbook_inputs_used": False,
        "injury_rows_mutated": False,
        "live_nflcom_request_used": False,
    }
    AUDIT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[preserved_injury_scope] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    certify_scope()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
