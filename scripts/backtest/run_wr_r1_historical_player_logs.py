#!/usr/bin/env python3
"""Mechanical WR-R1 wrapper for pre-2021 regular-season week numbering.

The canonical historical_player_logs builder uses a universal week<=18 filter,
which is correct for 2021+ but admits nflverse postseason Week 18 rows for older
17-week regular seasons such as 2019. WR-R1 already has an exact REG-only
schedule artifact, so this wrapper restricts raw weekly rows to season/week keys
that actually exist in that schedule before invoking the unchanged M38 builder.

No model input is added and no scientific rule is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.backtest import historical_player_logs as hp


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    schedule = pd.read_csv(args.schedule)
    schedule.columns = [str(c).strip().lower() for c in schedule.columns]
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
    allowed = {
        int(season): set(schedule.loc[schedule["season"].eq(int(season)), "week"].dropna().astype(int).tolist())
        for season in seasons
    }

    original = hp._load_historical_weekly

    def exact_reg_week_loader(season: int):
        raw = original(int(season))
        x = raw.copy()
        x.columns = [str(c).strip().lower() for c in x.columns]
        if "week" not in x.columns:
            raise RuntimeError(f"weekly stats missing week for {season}")
        week = pd.to_numeric(x["week"], errors="coerce")
        keep = week.isin(allowed.get(int(season), set()))
        return raw.loc[keep.to_numpy()].copy()

    with patch.object(hp, "_load_historical_weekly", side_effect=exact_reg_week_loader):
        out = hp.build_historical_player_logs(seasons=seasons, schedule_history=schedule)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[wr-r1-player-logs] wrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
