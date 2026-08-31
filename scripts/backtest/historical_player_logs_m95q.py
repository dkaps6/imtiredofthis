#!/usr/bin/env python3
"""M95Q-only historical player-log builder with exact regular-season calendars.

The generic historical loader uses a universal Week 1-18 ceiling. That is
correct for 2021+ but nflverse weekly data can expose postseason observations as
Week 18 for pre-2021 seasons. M95Q spans 2018-2024, so this wrapper derives each
season's maximum regular-season week from the already-built REG-only schedule
artifact and filters weekly player rows to that season-specific calendar before
attaching opponents.

Mechanical scope only: no projection/model/feature/gate changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly
from scripts.backtest.historical_player_logs import _load_historical_weekly


def build_logs(seasons: list[int], schedule_history: pd.DataFrame) -> pd.DataFrame:
    sched = schedule_history.copy()
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "team", "opponent"}
    if not required.issubset(sched.columns):
        raise RuntimeError(f"M95Q historical schedule missing {sorted(required - set(sched.columns))}")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched["team"] = sched["team"].map(canon_team)
    sched["opponent"] = sched["opponent"].map(canon_team)
    if "game_id" not in sched.columns:
        sched["game_id"] = ""
    sched = sched[["season", "week", "team", "opponent", "game_id"]].drop_duplicates(
        ["season", "week", "team"]
    )

    frames: list[pd.DataFrame] = []
    for season in sorted(set(int(s) for s in seasons)):
        ss = sched.loc[sched["season"].eq(season)].copy()
        if ss.empty:
            raise RuntimeError(f"M95Q schedule has no rows for historical season {season}")
        max_week = int(pd.to_numeric(ss["week"], errors="coerce").max())
        if max_week not in (17, 18):
            raise RuntimeError(f"M95Q unexpected regular-season max week {max_week} for {season}")

        normalized = _normalize_weekly(_load_historical_weekly(season), season)
        weeks = pd.to_numeric(normalized["week"], errors="coerce")
        dropped_after_calendar = int(weeks.gt(max_week).sum())
        normalized = normalized.loc[weeks.between(1, max_week)].copy()
        normalized = normalized.merge(
            ss, on=["season", "week", "team"], how="left", validate="many_to_one"
        )
        missing = normalized["opponent"].isna() | normalized["opponent"].astype(str).eq("")
        if missing.any():
            sample = normalized.loc[missing, ["season", "week", "team"]].drop_duplicates().head(10)
            raise RuntimeError(
                "M95Q unresolved opponent inside valid regular-season calendar: "
                + repr(sample.to_dict(orient="records"))
            )
        print(
            f"[m95q_player_logs] season={season} max_regular_week={max_week} "
            f"rows={len(normalized)} dropped_after_calendar={dropped_after_calendar}"
        )
        frames.append(normalized)

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if out.empty:
        raise RuntimeError("M95Q historical player log builder produced zero rows")
    keys = ["season", "week", "team", "player_clean_key"]
    if out.duplicated(keys).any():
        raise RuntimeError("M95Q historical player logs contain duplicate season/week/team/player rows")
    return out.sort_values(keys).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not args.schedule.exists():
        raise RuntimeError(f"missing historical schedule: {args.schedule}")
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    out = build_logs(seasons, pd.read_csv(args.schedule))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[m95q_player_logs] wrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
