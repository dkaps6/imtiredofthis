#!/usr/bin/env python3
"""Historical player logs bounded to schedule-valid regular-season weeks.

Mechanical compatibility helper for older NFL seasons. nflverse weekly player
stats can expose rows whose numeric week exceeds that season's actual regular
season. The canonical schedule artifact is the authority for valid REG weeks.
This wrapper preserves all existing normalization/opponent logic while filtering
weekly player rows to the exact season/week set present in schedule_history.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.historical_player_logs import _load_historical_weekly
from scripts.player_form_v2 import _normalize_weekly


def build_schedule_bounded_logs(*, seasons: Iterable[int], schedule_history: pd.DataFrame) -> pd.DataFrame:
    sched = schedule_history.copy()
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "team", "opponent"}
    if not required.issubset(sched.columns):
        raise RuntimeError(f"historical schedule missing columns: {sorted(required - set(sched.columns))}")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched = sched.loc[sched["week"].between(1, 18)].copy()
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
        valid_weeks = sorted(pd.to_numeric(ss["week"], errors="coerce").dropna().astype(int).unique().tolist())
        if not valid_weeks:
            raise RuntimeError(f"schedule has zero valid regular-season weeks for {season}")
        normalized = _normalize_weekly(_load_historical_weekly(season), season)
        week_num = pd.to_numeric(normalized["week"], errors="coerce")
        normalized = normalized.loc[week_num.isin(valid_weeks)].copy()
        normalized = normalized.merge(
            ss,
            on=["season", "week", "team"],
            how="left",
            validate="many_to_one",
        )
        missing = normalized["opponent"].isna() | normalized["opponent"].astype(str).eq("")
        if missing.any():
            sample = normalized.loc[missing, ["season", "week", "team"]].drop_duplicates().head(10)
            raise RuntimeError(
                "schedule-bounded historical logs could not resolve opponent: "
                + repr(sample.to_dict(orient="records"))
            )
        frames.append(normalized)
        print(f"[schedule_bounded_logs] season={season} valid_weeks={valid_weeks} rows={len(normalized)}")

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if out.empty:
        raise RuntimeError("schedule-bounded historical log builder produced zero rows")
    if out.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("schedule-bounded historical logs contain duplicate player-game rows")
    return out.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    seasons = [int(x.strip()) for x in a.seasons.split(",") if x.strip()]
    out = build_schedule_bounded_logs(seasons=seasons, schedule_history=pd.read_csv(a.schedule))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[schedule_bounded_logs] wrote {len(out)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
