#!/usr/bin/env python3
"""Mechanical execution wrapper for frozen RB-R7 science.

The original R7 experiment correctly built 2020 as a 17-week season, but reused
R6's training helper, which hard-codes ``range(1, 19)`` and therefore attempted
to open a nonexistent 2020 Week 18 universe artifact.  This wrapper changes only
that execution assumption: training weeks are discovered from the already-built
pregame universe artifacts for the requested season.

No R7 scientific feature, alpha, residual definition, seed, allocation mechanism,
confirmation fold, threshold, or integrity gate is changed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import evaluate_rb_r7_fixed_room_within_participation_v1 as r7
from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6


def _available_training_weeks(*, season: int, data_dir: Path) -> list[int]:
    root = data_dir / "pregame_universe"
    weeks: list[int] = []
    for p in root.glob(f"{season}_week_*.csv"):
        try:
            weeks.append(int(p.stem.rsplit("_", 1)[-1]))
        except Exception:
            continue
    weeks = sorted(set(w for w in weeks if 1 <= w <= 18))
    if not weeks:
        raise RuntimeError(f"R7B found no training universe weeks for {season} in {root}")
    return weeks


def _fit_within_schedule_bounded(*, season: int, data_dir: Path, logs: pd.DataFrame, snaps: pd.DataFrame):
    within_parts: list[pd.DataFrame] = []
    future_total = 0
    weeks = _available_training_weeks(season=season, data_dir=data_dir)

    for week in weeks:
        baseline = r6._build_bundle_frame(
            season=season,
            week=week,
            prior_season=season - 1,
            data_dir=data_dir,
            logs=logs,
        )
        rb, _, future = r6._rb_features(baseline, snaps)
        future_total += int(future)
        actual_t = r6._actual_target_frame(logs, season, week)

        rbx = rb.merge(
            actual_t,
            on=["team", "player_clean_key"],
            how="left",
            validate="one_to_one",
        )
        rbx["actual_targets"] = pd.to_numeric(
            rbx["actual_targets"], errors="coerce"
        ).fillna(0.0)
        rb_actual = (
            rbx.groupby(["event_id", "team"], as_index=False)["actual_targets"]
            .sum()
            .rename(columns={"actual_targets": "actual_rb_targets"})
        )
        rbx = rbx.merge(rb_actual, on=["event_id", "team"], how="left")
        rbx["actual_rb_targets"] = pd.to_numeric(
            rbx["actual_rb_targets"], errors="coerce"
        ).fillna(0.0)
        rbx["actual_rb_within_share"] = np.where(
            rbx["actual_rb_targets"].gt(0),
            rbx["actual_targets"] / rbx["actual_rb_targets"],
            0.0,
        )
        rbx["within_residual_target"] = (
            np.log(rbx["actual_rb_within_share"].clip(lower=0.0) + r6.EPS)
            - np.log(rbx["b0_rb_within_share"].clip(lower=0.0) + r6.EPS)
        ).clip(-r6.TRAIN_CLIP, r6.TRAIN_CLIP)
        rbx["season"] = int(season)
        rbx["week"] = int(week)
        within_parts.append(
            rbx.loc[rbx["actual_rb_targets"].gt(0) & rbx["b0_rb_pool"].gt(0)].copy()
        )
        print(
            f"[rb-r7b] training season={season} week={week:02d} "
            f"rb_rows={len(rbx)}"
        )

    if future_total:
        raise RuntimeError(
            f"R7B training season {season} used same/future participation: {future_total}"
        )
    within_train = pd.concat(within_parts, ignore_index=True)
    if within_train.empty:
        raise RuntimeError(f"R7B empty within-RB training casebook for {season}")

    model = make_pipeline(StandardScaler(), Ridge(alpha=r7.ALPHA))
    model.fit(within_train[r7.FEATURES], within_train["within_residual_target"])
    return model, within_train


r7._fit_within = _fit_within_schedule_bounded


if __name__ == "__main__":
    raise SystemExit(r7.main())
