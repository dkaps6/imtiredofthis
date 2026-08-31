#!/usr/bin/env python3
"""Fetch live sportsbook data and scope it to the canonical active NFL slate.

This wrapper exists so Full Slate never consumes stale or off-slate sportsbook
artifacts. It clears previous odds outputs before fetching, lets the existing
OddsAPI adapter do its provider work, then filters every event-bearing artifact
to the authoritative season/week schedule already built by Full Slate.

A legitimate preseason/early-week state where no player prop markets are posted
is non-fatal. The wrapper writes data/live_odds_status.json with available=false
so the football model can continue while sportsbook comparison/pricing is
skipped cleanly. Provider/auth failures remain fatal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_week

DATA = Path("data")
OUTPUTS = Path("outputs")
STATUS = DATA / "live_odds_status.json"
TEAM_WEEK_MAP = DATA / "team_week_map.csv"

CRITICAL_EVENT_ARTIFACTS = [
    OUTPUTS / "odds_game.csv",
    DATA / "odds_game.csv",
    OUTPUTS / "props_raw.csv",
    DATA / "props_raw.csv",
    OUTPUTS / "props_enriched.csv",
    DATA / "props_enriched.csv",
    OUTPUTS / "props_raw_wide.csv",
    DATA / "opponent_map_from_props.csv",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _clear_stale_odds_artifacts() -> None:
    """Remove tracked/leftover sportsbook files before any live fetch."""
    targets = set(CRITICAL_EVENT_ARTIFACTS)
    targets.update(OUTPUTS.glob("props_*.csv"))
    raw_dir = OUTPUTS / "props_raw"
    if raw_dir.exists():
        targets.update(raw_dir.glob("*.csv"))
    targets.add(STATUS)
    for path in sorted(targets):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to clear stale sportsbook artifact {path}: {exc}") from exc


def _active_game_pairs(schedule: pd.DataFrame, season: int, week: int) -> set[tuple[str, str]]:
    required = {"season", "week", "team", "opponent"}
    missing = required - set(schedule.columns)
    if missing:
        raise RuntimeError(f"team_week_map missing columns required for live odds gate: {sorted(missing)}")
    x = schedule.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].eq(int(week))].copy()
    if x.empty:
        raise RuntimeError(f"No canonical schedule rows for live odds season={season} week={week}")
    pairs: set[tuple[str, str]] = set()
    for team, opp in zip(x["team"], x["opponent"]):
        a, b = canon_team(team), canon_team(opp)
        if not a or not b:
            raise RuntimeError(f"Unresolvable team identity in live odds schedule: {team!r} vs {opp!r}")
        pairs.add(tuple(sorted((a, b))))
    return pairs


def _allowed_event_ids(game_odds: pd.DataFrame, active_pairs: set[tuple[str, str]]) -> set[str]:
    if game_odds.empty:
        return set()
    required = {"event_id", "home_team", "away_team"}
    missing = required - set(game_odds.columns)
    if missing:
        raise RuntimeError(f"odds_game missing columns required for active-slate gate: {sorted(missing)}")
    allowed: set[str] = set()
    for row in game_odds.itertuples(index=False):
        home = canon_team(getattr(row, "home_team"))
        away = canon_team(getattr(row, "away_team"))
        if home and away and tuple(sorted((home, away))) in active_pairs:
            allowed.add(str(getattr(row, "event_id")))
    return allowed


def _filter_event_csv(path: Path, allowed_event_ids: set[str]) -> int:
    if not path.exists():
        return 0
    df = _safe_read_csv(path)
    if df.empty:
        # Preserve a parseable file if a schema exists; otherwise remove the
        # unusable/headerless artifact rather than leave a false-positive file.
        if len(df.columns):
            df.to_csv(path, index=False)
        else:
            path.unlink(missing_ok=True)
        return 0
    if "event_id" not in df.columns:
        return len(df)
    event_ids = df["event_id"].astype("string").fillna("").str.strip()
    scoped = df.loc[event_ids.isin(allowed_event_ids)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    scoped.to_csv(path, index=False)
    return len(scoped)


def _scope_all_event_artifacts(allowed_event_ids: set[str]) -> None:
    paths = set(CRITICAL_EVENT_ARTIFACTS)
    paths.update(OUTPUTS.glob("props_*.csv"))
    raw_dir = OUTPUTS / "props_raw"
    if raw_dir.exists():
        paths.update(raw_dir.glob("*.csv"))
    for path in sorted(paths):
        _filter_event_csv(path, allowed_event_ids)


def _actual_prop_rows(props: pd.DataFrame) -> int:
    if props.empty:
        return 0
    player_col = next(
        (c for c in ("canonical_player_name", "player_canonical", "player") if c in props.columns),
        None,
    )
    if player_col is None:
        return 0
    player_ok = props[player_col].astype("string").fillna("").str.strip().ne("")
    if "bookmaker_missing" in props.columns:
        missing = pd.to_numeric(props["bookmaker_missing"], errors="coerce").fillna(0).eq(1)
        player_ok &= ~missing
    return int(player_ok.sum())


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[live_odds_gate] "
        f"status={payload.get('status')} available={payload.get('available')} "
        f"season={payload.get('season')} week={payload.get('week')} "
        f"active_events={payload.get('active_event_count')} actual_prop_rows={payload.get('actual_prop_rows')}"
    )


def run_gate(season: int, date: str = "") -> dict:
    week = int(resolve_week())
    schedule = _safe_read_csv(TEAM_WEEK_MAP)
    active_pairs = _active_game_pairs(schedule, int(season), week)

    _clear_stale_odds_artifacts()

    cmd = [
        sys.executable,
        "scripts/fetch_props_oddsapi.py",
        "--season",
        str(int(season)),
        "--date",
        str(date or ""),
    ]
    proc = subprocess.run(cmd, env=dict(os.environ), check=False)
    if proc.returncode != 0:
        payload = {
            "status": "provider_error",
            "available": False,
            "season": int(season),
            "week": week,
            "slate_date": date or "",
            "active_game_count": len(active_pairs),
            "active_event_count": 0,
            "actual_prop_rows": 0,
            "game_odds_rows": 0,
            "fetch_returncode": int(proc.returncode),
        }
        _write_status(payload)
        raise RuntimeError(f"OddsAPI fetch failed with exit code {proc.returncode}")

    raw_game_odds = _safe_read_csv(OUTPUTS / "odds_game.csv")
    allowed_ids = _allowed_event_ids(raw_game_odds, active_pairs)
    _scope_all_event_artifacts(allowed_ids)

    scoped_games = _safe_read_csv(OUTPUTS / "odds_game.csv")
    scoped_props = _safe_read_csv(OUTPUTS / "props_raw.csv")
    actual_props = _actual_prop_rows(scoped_props)

    if not allowed_ids:
        state = "no_active_slate_markets"
        available = False
    elif actual_props <= 0:
        state = "no_player_prop_markets"
        available = False
    else:
        state = "available"
        available = True

    payload = {
        "status": state,
        "available": bool(available),
        "season": int(season),
        "week": week,
        "slate_date": date or "",
        "active_game_count": len(active_pairs),
        "active_event_count": len(allowed_ids),
        "actual_prop_rows": int(actual_props),
        "game_odds_rows": int(len(scoped_games)),
        "fetch_returncode": 0,
    }
    _write_status(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", default="")
    args = parser.parse_args()
    run_gate(int(args.season), str(args.date or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
