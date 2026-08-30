#!/usr/bin/env python3
"""M77 chronology hardening wrapper.

Ensures the personnel discontinuity baseline always compares against the team's
actual previous regular-season game, even if that prior game is absent from the
stable-primary QB evaluation cohort. No feature/model/gate changes.
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest import test_qb_exact_personnel_discontinuity as m77


def build_team_snapshots(base: pd.DataFrame, depth: dict[int, pd.DataFrame], schedule: pd.DataFrame):
    if schedule.empty:
        raise RuntimeError("schedule unavailable for M77 personnel chronology")
    targets = schedule.loc[m77.num(schedule["season"]).isin(sorted(base["season"].unique()))].copy()
    targets = targets.drop_duplicates(["season", "week", "team"]).sort_values(["season", "team", "week"])

    snapshots: dict[tuple[int, int, str], pd.DataFrame] = {}
    for r in targets.itertuples(index=False):
        key = (int(r.season), int(r.week), str(r.team))
        snapshots[key] = m77.m76.latest_depth_for_target(
            depth[int(r.season)], int(r.season), int(r.week), str(r.team), r.kickoff
        )

    prev_key: dict[tuple[int, int, str], tuple[int, int, str] | None] = {}
    for (_, _), g in targets.groupby(["season", "team"], sort=False):
        keys = [(int(x.season), int(x.week), str(x.team)) for x in g.itertuples(index=False)]
        for i, key in enumerate(keys):
            prev_key[key] = keys[i - 1] if i > 0 else None
    return snapshots, prev_key


m77.build_team_snapshots = build_team_snapshots

if __name__ == "__main__":
    raise SystemExit(m77.main())
