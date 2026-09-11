#!/usr/bin/env python3
"""Build the frozen QB public-intent source-audit schedule and review manifest.

This utility intentionally produces NO source dispositions. It constructs only the
preregistered deterministic 2023-2025 team-week universe (weeks 2,5,8,11,14,17)
and a PENDING_REVIEW collection manifest. That prevents unsearched rows from being
silently labeled NO_RECONSTRUCTABLE_SOURCE and keeps source qualification separate
from football outcomes/model data.

Historical kickoff timestamps come from ``scripts.providers.build_schedule``; this
script does not alter the frozen sample or infer missing kickoff times itself.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.providers.build_schedule import build_or_get_schedule

SEASONS = (2023, 2024, 2025)
SAMPLED_WEEKS = {2, 5, 8, 11, 14, 17}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule-out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    args = ap.parse_args()

    schedule_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    seen_games: set[tuple[int, int, str, str]] = set()
    seen_team_weeks: set[tuple[int, int, str]] = set()

    for season in SEASONS:
        df = build_or_get_schedule(season)
        df = df[df["week"].astype(int).isin(SAMPLED_WEEKS)].copy()
        for _, r in df.iterrows():
            week = int(r["week"])
            home = str(r["home_team"]).strip().upper()
            away = str(r["away_team"]).strip().upper()
            ko = r["kickoff_utc"]
            if getattr(ko, "isoformat", None):
                kickoff = ko.isoformat()
            else:
                kickoff = str(ko)
            if not kickoff or kickoff.lower() in {"nat", "nan", "none"}:
                raise RuntimeError(f"missing kickoff for {season} W{week} {away}@{home}")

            gkey = (season, week, home, away)
            if gkey in seen_games:
                continue
            seen_games.add(gkey)
            schedule_rows.append({
                "season": str(season), "week": str(week), "home_team": home,
                "away_team": away, "kickoff": kickoff,
            })
            for team, opp in ((home, away), (away, home)):
                tkey = (season, week, team)
                if tkey in seen_team_weeks:
                    raise RuntimeError(f"duplicate team-week in frozen sample: {tkey}")
                seen_team_weeks.add(tkey)
                manifest_rows.append({
                    "season": str(season), "week": str(week), "team": team,
                    "opponent": opp, "kickoff": kickoff,
                    "review_status": "PENDING_REVIEW",
                    "official_search_complete": "false",
                    "local_search_complete": "false",
                    "candidate_locator": "",
                    "notes": "",
                })

    schedule_rows.sort(key=lambda x: (int(x["season"]), int(x["week"]), x["kickoff"], x["home_team"]))
    manifest_rows.sort(key=lambda x: (int(x["season"]), int(x["week"]), x["team"]))

    # Structural checks independent of outcomes.
    seasons_present = {int(r["season"]) for r in manifest_rows}
    weeks_present = {int(r["week"]) for r in manifest_rows}
    if seasons_present != set(SEASONS):
        raise RuntimeError(f"season coverage mismatch: {sorted(seasons_present)}")
    if weeks_present != SAMPLED_WEEKS:
        raise RuntimeError(f"sampled-week coverage mismatch: {sorted(weeks_present)}")
    if len(manifest_rows) != 2 * len(schedule_rows):
        raise RuntimeError("team-week/game cardinality mismatch")

    _write_csv(args.schedule_out, ["season", "week", "home_team", "away_team", "kickoff"], schedule_rows)
    _write_csv(
        args.manifest_out,
        ["season", "week", "team", "opponent", "kickoff", "review_status",
         "official_search_complete", "local_search_complete", "candidate_locator", "notes"],
        manifest_rows,
    )

    print(f"games={len(schedule_rows)} team_weeks={len(manifest_rows)}")
    print("sportsbook_fields_used=0 predictive_models_fit=0 production_changes=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
