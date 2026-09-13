#!/usr/bin/env python3
"""Acquire current-week official-ish NFL inactive status from ESPN's core API.

Replaces the dead nfl.com/inactives scraper (nfl_official_inactives_v1.py):
that page has shipped a "Please check back soon" placeholder with zero
structured content all 2026 season -- confirmed empty across every game on
a live game day, well past the point official inactives should exist.
Same output contract as the script it replaces, so downstream consumers
(build_current_player_availability_v1.py) need no changes:
data/official_inactives_v1.csv + data/official_inactives_v1_status.json.

Source: ESPN's core API team injuries endpoint
(sports.core.api.espn.com/v2/.../teams/{id}/injuries), an unbounded,
newest-first-sorted per-team injury log -- unlike the site API's per-game
'injuries' convenience field, which is silently capped at exactly 5 entries
per team regardless of query params (verified against all 16 games on a
live Sunday: every single team showed count=5, an impossible coincidence
for real independent per-team data -- a display limit, not real data).

Known limitation: this is an injury-report feed. A player inactive for a
pure roster/numbers decision with no reported injury (a "healthy scratch")
will not appear here and is treated as available. This is the same
residual risk already accepted elsewhere in this pipeline for teams
outside the required pre-kickoff window (NOT_YET_REQUIRED teams are
treated as eligible with zero official confirmation at all), and
prop-relevant players are overwhelmingly captured by the injury report
when they are actually out.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from scripts._opponent_map import canon_team

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TEAM_INJURIES_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
OUT = Path("data/official_inactives_v1.csv")
STATUS = Path("data/official_inactives_v1_status.json")
RECENT_WINDOW_DAYS = 6
ITEMS_PER_TEAM = 25
REQUEST_TIMEOUT = 30


def _get(url: str, **kwargs):
    # ESPN's edge returns 403 for an identifying custom User-Agent (verified
    # live); the plain default requests UA is what actually works.
    return requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)


def _athlete_id_from_ref(ref: str) -> str:
    m = re.search(r"/athletes/(\d+)", str(ref or ""))
    return m.group(1) if m else ""


def fetch_scoreboard_teams() -> list[dict]:
    """Return [{'team_id': ..., 'team': abbr}, ...] for this week's games."""
    r = _get(SCOREBOARD_URL)
    r.raise_for_status()
    events = r.json().get("events", [])
    seen: set[str] = set()
    teams: list[dict] = []
    for e in events:
        comps = e.get("competitions", [{}])
        competitors = comps[0].get("competitors", []) if comps else []
        for comp in competitors:
            team = comp.get("team", {})
            tid, abbr = team.get("id"), team.get("abbreviation")
            if tid and abbr and tid not in seen:
                seen.add(tid)
                teams.append({"team_id": str(tid), "team": str(abbr)})
    return teams


def fetch_team_injury_items(team_id: str, *, limit: int = ITEMS_PER_TEAM) -> list[dict]:
    """Fetch the newest `limit` injury log entries for one team (already
    sorted newest-first by the API; no full pagination needed)."""
    r = _get(TEAM_INJURIES_URL.format(team_id=team_id), params={"limit": limit})
    r.raise_for_status()
    refs = r.json().get("items", [])
    items = []
    for ref in refs:
        href = ref.get("$ref")
        if not href:
            continue
        rr = _get(href)
        rr.raise_for_status()
        items.append(rr.json())
    return items


def fetch_athlete_name(athlete_ref: str) -> str:
    href = (athlete_ref or "").strip()
    if not href:
        return ""
    r = _get(href)
    r.raise_for_status()
    d = r.json()
    return str(d.get("displayName") or d.get("fullName") or "").strip()


def current_out_athlete_ids(items: list[dict], *, now: datetime) -> list[dict]:
    """Collapse a newest-first injury log to each athlete's latest entry,
    keep only entries that are recent and definitively 'Out'."""
    seen_athletes: set[str] = set()
    out: list[dict] = []
    for it in items:
        athlete_ref = (it.get("athlete") or {}).get("$ref", "")
        athlete_id = _athlete_id_from_ref(athlete_ref)
        if not athlete_id or athlete_id in seen_athletes:
            continue
        seen_athletes.add(athlete_id)
        date_s = it.get("date")
        try:
            dt = datetime.fromisoformat(str(date_s).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if now - dt > timedelta(days=RECENT_WINDOW_DAYS):
            continue
        status = str(it.get("status", "")).strip().lower()
        if status != "out":
            continue
        out.append({"athlete_id": athlete_id, "athlete_ref": athlete_ref, "status_date": dt.isoformat()})
    return out


def build(team_records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """team_records: [{'team': abbr, 'section_complete': bool, 'players': [name, ...]}, ...]"""
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    complete_teams: list[str] = []
    for rec in team_records:
        team = canon_team(rec["team"])
        complete = bool(rec["section_complete"])
        if complete:
            complete_teams.append(team)
        # Ledger row: one per team so "no listed players" is distinguishable
        # from "we never checked this team."
        rows.append({
            "team": team, "player": "", "listed_position": "",
            "section_complete": int(complete),
            "source_url": TEAM_INJURIES_URL,
            "source_asof_utc": now,
        })
        for name in rec.get("players", []):
            rows.append({
                "team": team, "player": name, "listed_position": "",
                "section_complete": int(complete),
                "source_url": TEAM_INJURIES_URL,
                "source_asof_utc": now,
            })
    frame = pd.DataFrame(rows, columns=["team", "player", "listed_position", "section_complete", "source_url", "source_asof_utc"])
    listed_players = int(frame["player"].astype(str).str.strip().ne("").sum())
    status = {
        "generated_at_utc": now,
        "source": "espn_official_inactives_core_api",
        "endpoint_reachable": True,
        "complete_team_sections": len(complete_teams),
        "complete_teams": sorted(complete_teams),
        "listed_players": listed_players,
        "payload_valid": bool(complete_teams),
    }
    return frame, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--status", type=Path, default=STATUS)
    a = ap.parse_args()

    now = datetime.now(timezone.utc)
    try:
        teams = fetch_scoreboard_teams()
    except Exception as exc:
        frame = pd.DataFrame(columns=["team", "player", "listed_position", "section_complete", "source_url", "source_asof_utc"])
        status = {
            "generated_at_utc": now.isoformat(), "source": "espn_official_inactives_core_api",
            "endpoint_reachable": False, "complete_team_sections": 0, "complete_teams": [],
            "listed_players": 0, "payload_valid": False, "error": f"{type(exc).__name__}:{exc}",
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(a.out, index=False)
        a.status.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    team_records = []
    for t in teams:
        try:
            items = fetch_team_injury_items(t["team_id"])
            out_athletes = current_out_athlete_ids(items, now=now)
            names = []
            for oa in out_athletes:
                name = fetch_athlete_name(oa["athlete_ref"])
                if name:
                    names.append(name)
            team_records.append({"team": t["team"], "section_complete": True, "players": names})
        except Exception:
            team_records.append({"team": t["team"], "section_complete": False, "players": []})

    frame, status = build(team_records)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    a.status.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
