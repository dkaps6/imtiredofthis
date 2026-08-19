"""Build canonical historical player-game logs for walk-forward backtests.

This intentionally reuses PlayerForm v2's nflverse normalization, but attaches
opponents from the historical schedule artifact rather than today's production
team_week_map.csv. The output contains regular-season completed-game
observations only; the walk-forward context layer is responsible for enforcing
the prediction cutoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _load_weekly, _normalize_weekly

REGULAR_SEASON_MAX_WEEK = 18


def build_historical_player_logs(
    *, seasons: Iterable[int], schedule_history: pd.DataFrame
) -> pd.DataFrame:
    sched = schedule_history.copy()
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    required = {"season", "week", "team", "opponent"}
    if not required.issubset(sched.columns):
        raise RuntimeError(f"historical schedule missing columns: {sorted(required - set(sched.columns))}")
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce").astype("Int64")
    sched = sched.loc[sched["week"].between(1, REGULAR_SEASON_MAX_WEEK)].copy()
    sched["team"] = sched["team"].map(canon_team)
    sched["opponent"] = sched["opponent"].map(canon_team)
    if "game_id" not in sched.columns:
        sched["game_id"] = ""
    sched = sched[["season", "week", "team", "opponent", "game_id"]].drop_duplicates(
        ["season", "week", "team"]
    )

    frames: list[pd.DataFrame] = []
    for season in sorted(set(int(s) for s in seasons)):
        normalized = _normalize_weekly(_load_weekly(season), season)
        # nflverse weekly player stats include postseason weeks (19+). This
        # backtest is explicitly regular season Weeks 1-18, and schedule_history
        # is intentionally REG-only, so postseason observations must be excluded
        # before opponent attachment rather than treated as missing schedule data.
        normalized = normalized.loc[
            pd.to_numeric(normalized["week"], errors="coerce").between(1, REGULAR_SEASON_MAX_WEEK)
        ].copy()
        normalized = normalized.merge(
            sched.loc[sched["season"].eq(season)],
            on=["season", "week", "team"],
            how="left",
            validate="many_to_one",
        )
        missing = normalized["opponent"].isna() | normalized["opponent"].astype(str).eq("")
        if missing.any():
            sample = normalized.loc[missing, ["season", "week", "team"]].drop_duplicates().head(10)
            raise RuntimeError(
                "historical player logs could not resolve opponent for regular-season schedule rows: "
                + sample.to_dict(orient="records").__repr__()
            )
        frames.append(normalized)
        print(f"[backtest_player_logs] season={season} regular_season_rows={len(normalized)}")

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if out.empty:
        raise RuntimeError("historical player log builder produced zero rows")
    if out.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("historical player logs contain duplicate season/week/team/player rows")
    return out.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--out", type=Path, default=Path("data/backtests/player_game_logs_history.csv"))
    args = p.parse_args()
    if not args.schedule.exists():
        raise RuntimeError(f"missing historical schedule: {args.schedule}")
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    out = build_historical_player_logs(seasons=seasons, schedule_history=pd.read_csv(args.schedule))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[backtest_player_logs] wrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
