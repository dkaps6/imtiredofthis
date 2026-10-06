#!/usr/bin/env python3
"""Fail-closed Week-4 postgame source-completeness gate.

Verifies all 2026 REG Week 4 games are final in nflverse and that the three
postgame sources required by the canonical GSIS-aware grader (player weekly
stats, weekly rosters, and snap counts) cover every scheduled team before
Week-4 scoring is allowed to proceed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

SEASON = 2026
WEEK = 4


def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _team_col(df: pd.DataFrame) -> str:
    for c in ("recent_team", "team", "team_abbr", "club", "club_code"):
        if c in df.columns:
            return c
    raise RuntimeError(f"no team column found; columns={sorted(df.columns)}")


def _team_set(df: pd.DataFrame) -> set[str]:
    c = _team_col(df)
    return {
        canon_team(str(v).strip())
        for v in df[c].dropna().tolist()
        if str(v).strip()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import nflreadpy as nfl

    schedule = _lower(_to_pandas(nfl.load_schedules(seasons=[SEASON])))
    if "season" in schedule.columns:
        schedule = schedule.loc[pd.to_numeric(schedule["season"], errors="coerce").eq(SEASON)]
    if "game_type" in schedule.columns:
        schedule = schedule.loc[schedule["game_type"].astype(str).str.upper().eq("REG")]
    schedule = schedule.loc[pd.to_numeric(schedule.get("week"), errors="coerce").eq(WEEK)].copy()
    if schedule.empty:
        raise RuntimeError("nflverse schedule has no 2026 REG Week 4 games")

    for c in ("home_team", "away_team", "home_score", "away_score"):
        if c not in schedule.columns:
            raise RuntimeError(f"schedule missing required column {c}")

    incomplete = schedule.loc[
        pd.to_numeric(schedule["home_score"], errors="coerce").isna()
        | pd.to_numeric(schedule["away_score"], errors="coerce").isna()
    ].copy()
    if not incomplete.empty:
        cols = [c for c in ("game_id", "home_team", "away_team", "home_score", "away_score") if c in incomplete.columns]
        raise RuntimeError(
            "Week 4 schedule still has non-final games: "
            + incomplete[cols].to_dict(orient="records").__repr__()
        )

    scheduled_teams = {
        canon_team(str(v).strip())
        for c in ("home_team", "away_team")
        for v in schedule[c].dropna().tolist()
        if str(v).strip()
    }

    stats = _lower(_to_pandas(nfl.load_player_stats(seasons=[SEASON], summary_level="week")))
    if "season" in stats.columns:
        stats = stats.loc[pd.to_numeric(stats["season"], errors="coerce").eq(SEASON)]
    stats = stats.loc[pd.to_numeric(stats.get("week"), errors="coerce").eq(WEEK)].copy()

    rosters = _lower(_to_pandas(nfl.load_rosters_weekly(SEASON)))
    if "season" in rosters.columns:
        rosters = rosters.loc[pd.to_numeric(rosters["season"], errors="coerce").eq(SEASON)]
    rosters = rosters.loc[pd.to_numeric(rosters.get("week"), errors="coerce").eq(WEEK)].copy()

    snaps = _lower(_to_pandas(nfl.load_snap_counts(seasons=[SEASON])))
    if "season" in snaps.columns:
        snaps = snaps.loc[pd.to_numeric(snaps["season"], errors="coerce").eq(SEASON)]
    snaps = snaps.loc[pd.to_numeric(snaps.get("week"), errors="coerce").eq(WEEK)].copy()

    coverage = {
        "player_stats": _team_set(stats),
        "weekly_rosters": _team_set(rosters),
        "snap_counts": _team_set(snaps),
    }
    missing = {name: sorted(scheduled_teams - teams) for name, teams in coverage.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise RuntimeError(f"Week 4 postgame source coverage incomplete: {missing}")

    result = {
        "status": "WEEK4_OUTCOME_SOURCES_READY",
        "season": SEASON,
        "week": WEEK,
        "schedule_games": int(len(schedule)),
        "scheduled_teams": int(len(scheduled_teams)),
        "all_games_final": True,
        "player_stats_rows": int(len(stats)),
        "weekly_roster_rows": int(len(rosters)),
        "snap_count_rows": int(len(snaps)),
        "player_stats_team_coverage": int(len(coverage["player_stats"] & scheduled_teams)),
        "weekly_roster_team_coverage": int(len(coverage["weekly_rosters"] & scheduled_teams)),
        "snap_count_team_coverage": int(len(coverage["snap_counts"] & scheduled_teams)),
        "missing_team_coverage": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
