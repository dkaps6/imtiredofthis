#!/usr/bin/env python3
"""Repair NFL.com injury team identity when the table omits a Team column.

NFL.com renders the team label immediately outside each injury table. Pandas
``read_html`` therefore sees the player rows but not the club identity. This
adapter reattaches that page-context team label, then reuses the canonical
injury normalizer. It is provider plumbing only; it does not infer injuries or
alter football projections.
"""
from __future__ import annotations

import io
import json
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


def _clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def _nickname_map() -> dict[str, str]:
    out = {str(k).strip().lower(): str(v) for k, v in TEAM_NICKNAME_TO_ABBR.items()}
    # NFL.com sometimes uses the bare numeric nickname.
    out.setdefault("49ers", "SF")
    return out


def _nearest_team_label(table, labels: dict[str, str]) -> str:
    # Text nodes immediately preceding the table are more reliable than walking
    # generic parent containers whose text can contain multiple teams/games.
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

    if not team_missing:
        _validate(current, season, week)
        result = {
            "disposition": "INJURY_IDENTITY_READY",
            "state": state,
            "source": source,
            "rows": int(len(current)),
        }
        print("[injury_identity] " + json.dumps(result, sort_keys=True))
        return result

    if "nfl.com" not in source:
        raise RuntimeError(
            f"Official injury report contains unresolved team identity from non-NFL.com source={source}"
        )

    html = fetch_injuries_html()
    detected = int(detect_current_week_from_html(html))
    if detected != week:
        raise RuntimeError(f"NFL.com injury repair week mismatch expected={week} detected={detected}")
    repaired = _parse_context_tables(html, season, week)
    _validate(repaired, season, week)
    repaired = repaired[SCHEMA].copy()
    repaired.to_csv(INJURIES, index=False)

    audit = repaired[["player", "team", "status", "practice_status", "body_part", "source"]].copy()
    audit["team_identity_resolved"] = 1
    audit.to_csv(AUDIT, index=False)

    status["source"] = "nfl.com_context_repair"
    status["rows"] = int(len(repaired))
    status["team_identity_repaired"] = True
    status["team_identity_unresolved_rows"] = 0
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "disposition": "INJURY_IDENTITY_REPAIRED",
        "state": "official_report",
        "source": status["source"],
        "rows": int(len(repaired)),
        "teams": int(repaired["team"].nunique()),
    }
    print("[injury_identity] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    repair_if_needed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
