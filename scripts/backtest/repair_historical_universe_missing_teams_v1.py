#!/usr/bin/env python3
"""Mechanical repair for missing scheduled teams in historical pregame universes.

Uses ONLY the same target-week nflverse weekly roster snapshot already used by the
historical input builder. It does not use box scores, participation outcomes, or
future-week rosters. The repair is only activated when a scheduled team has zero
rows in the generated pregame universe, typically because legacy roster status
coding caused the generic ACT/INA filter to remove the entire team.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

OFF_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "TE"}


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _parse_weeks(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", default="1-17")
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()

    import nflreadpy as nfl

    schedule = pd.read_csv(args.data_dir / "schedule_history.csv", low_memory=False)
    schedule.columns = [str(c).strip().lower() for c in schedule.columns]
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)

    rosters = _to_pandas(nfl.load_rosters_weekly(int(args.season)))
    rosters.columns = [str(c).strip().lower() for c in rosters.columns]
    name_col = "full_name" if "full_name" in rosters.columns else "football_name" if "football_name" in rosters.columns else None
    if name_col is None:
        raise RuntimeError("weekly rosters missing full_name/football_name")
    required = {"season", "week", "team", "position"}
    if not required.issubset(rosters.columns):
        raise RuntimeError(f"weekly rosters missing columns: {sorted(required - set(rosters.columns))}")
    rosters["team"] = rosters["team"].map(canon_team)
    rosters["position"] = rosters["position"].astype(str).str.upper().str.strip()

    audit_rows: list[dict] = []
    for week in _parse_weeks(args.weeks):
        path = args.data_dir / "pregame_universe" / f"{args.season}_week_{week:02d}.csv"
        u = pd.read_csv(path, low_memory=False)
        u.columns = [str(c).strip().lower() for c in u.columns]
        u["team"] = u["team"].map(canon_team)

        sched = schedule.loc[
            pd.to_numeric(schedule["season"], errors="coerce").eq(int(args.season))
            & pd.to_numeric(schedule["week"], errors="coerce").eq(int(week))
        ].copy()
        scheduled = set(sched["team"].astype(str))
        existing = set(u["team"].astype(str))
        missing = sorted(scheduled - existing)
        if not missing:
            continue

        week_roster = rosters.loc[
            pd.to_numeric(rosters["season"], errors="coerce").eq(int(args.season))
            & pd.to_numeric(rosters["week"], errors="coerce").eq(int(week))
            & rosters["position"].isin(OFF_POSITIONS)
        ].copy()

        additions: list[dict] = []
        for team in missing:
            rr = week_roster.loc[week_roster["team"].eq(team)].copy()
            if rr.empty:
                raw_teams = sorted(set(week_roster["team"].astype(str)))
                raise RuntimeError(
                    f"same-week roster repair found no offensive rows for missing team {team} "
                    f"season={args.season} week={week}; roster_teams={raw_teams}"
                )
            matchup = sched.loc[sched["team"].eq(team)].iloc[0]
            status_counts = rr["status"].fillna("<NA>").astype(str).value_counts().to_dict() if "status" in rr.columns else {}
            for _, r in rr.iterrows():
                player = str(r.get(name_col, "") or "").strip()
                if not player:
                    continue
                additions.append({
                    "player": player,
                    "team": team,
                    "opponent": str(matchup["opponent"]),
                    "position": str(r["position"]),
                    "role": "",
                    "game_id": str(matchup.get("game_id", "") or ""),
                    "season": int(args.season),
                    "week": int(week),
                    "pregame_source": "nflverse_weekly_roster_missing_team_mechanical_repair",
                })
            audit_rows.append({
                "season": int(args.season),
                "week": int(week),
                "team": team,
                "rows_added": int(len(rr)),
                "status_counts": str(status_counts),
                "same_week_roster_only": True,
                "future_rows_used": 0,
                "outcome_rows_used": 0,
            })

        if additions:
            add = pd.DataFrame(additions)
            u = pd.concat([u, add], ignore_index=True, sort=False)
            u = u.drop_duplicates(["team", "player"], keep="first").reset_index(drop=True)
            u.to_csv(path, index=False)
            print(f"[historical_universe_repair] {args.season} W{week:02d} missing={missing} rows_added={len(add)}")

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows, columns=[
        "season", "week", "team", "rows_added", "status_counts",
        "same_week_roster_only", "future_rows_used", "outcome_rows_used",
    ])
    audit.to_csv(args.audit, index=False)
    if audit.empty:
        print(f"[historical_universe_repair] season={args.season} no missing teams")
    else:
        print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
