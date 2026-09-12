#!/usr/bin/env python3
"""Build the clean historical full-stack projection trace used for Vegas grading.

This replaces the untracked ad-hoc game_id attachment that corrupted the prior
benchmark. Component predictions are football-only and are built first. Frozen
production ensemble weights are then applied. Finally, game identity is attached
only from the authoritative historical schedule on (season, week, team), with
hard fail-closed checks before any sportsbook archive is loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity
from scripts.modeling.ensemble_v2 import apply_ensemble


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component-file", action="append", required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("data/model_ensemble_weights.csv"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    components = pd.concat([_read(Path(p), "component file") for p in a.component_file], ignore_index=True)
    required = {"season", "week", "team", "opponent", "player_clean_key", "market", "mc_proj", "actual"}
    missing = sorted(required - set(components.columns))
    if missing:
        raise RuntimeError(f"component trace missing columns: {missing}")

    components["season"] = pd.to_numeric(components["season"], errors="raise").astype(int)
    components["week"] = pd.to_numeric(components["week"], errors="raise").astype(int)
    components["team"] = components["team"].map(canon_team)
    components["opponent"] = components["opponent"].map(canon_team)
    if components["team"].eq("").any() or components["opponent"].eq("").any():
        raise RuntimeError("component trace contains uncanonicalizable team/opponent")

    keys = ["season", "week", "team", "player_clean_key", "market"]
    if components.duplicated(keys).any():
        sample = components.loc[components.duplicated(keys, keep=False), keys].head(20).to_dict(orient="records")
        raise RuntimeError(f"component trace duplicate identity rows: {sample}")

    weights = _read(a.weights, "ensemble weights")
    full = apply_ensemble(components, weights=weights)

    schedule = _read(a.schedule, "historical schedule")
    schedule_required = {"season", "week", "team", "opponent", "game_id"}
    missing_schedule = sorted(schedule_required - set(schedule.columns))
    if missing_schedule:
        raise RuntimeError(f"historical schedule missing columns: {missing_schedule}")
    schedule["season"] = pd.to_numeric(schedule["season"], errors="raise").astype(int)
    schedule["week"] = pd.to_numeric(schedule["week"], errors="raise").astype(int)
    schedule["team"] = schedule["team"].map(canon_team)
    schedule["opponent"] = schedule["opponent"].map(canon_team)
    schedule = schedule[["season", "week", "team", "opponent", "game_id"]].drop_duplicates()
    if schedule.duplicated(["season", "week", "team"]).any():
        sample = schedule.loc[
            schedule.duplicated(["season", "week", "team"], keep=False),
            ["season", "week", "team", "opponent", "game_id"],
        ].head(20).to_dict(orient="records")
        raise RuntimeError(f"historical schedule duplicate team-week rows: {sample}")
    assert_benchmark_identity(schedule, label="authoritative historical schedule", require_team=True, require_opponent=True)

    # Never trust or preserve a pre-existing game_id on the component trace.
    # This is the root-cause guard against the untracked `_gid` post-processing
    # step that corrupted the original benchmark.
    full = full.drop(columns=["game_id"], errors="ignore")
    full = full.rename(columns={"opponent": "component_opponent"})
    full = full.merge(
        schedule.rename(columns={"opponent": "schedule_opponent"}),
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if full["game_id"].isna().any() or full["game_id"].astype(str).str.strip().eq("").any():
        sample = full.loc[
            full["game_id"].isna() | full["game_id"].astype(str).str.strip().eq(""),
            ["season", "week", "team", "player_clean_key", "market"],
        ].head(20).to_dict(orient="records")
        raise RuntimeError(f"authoritative schedule failed to resolve game_id: {sample}")

    comp_opp = full["component_opponent"].map(canon_team)
    sched_opp = full["schedule_opponent"].map(canon_team)
    mismatch = comp_opp.ne(sched_opp)
    if mismatch.any():
        sample = full.loc[
            mismatch,
            ["season", "week", "team", "component_opponent", "schedule_opponent", "game_id"],
        ].head(20).to_dict(orient="records")
        raise RuntimeError(f"component/schedule opponent mismatch: {sample}")

    full["opponent"] = sched_opp
    full = full.drop(columns=["component_opponent", "schedule_opponent"])
    identity = assert_benchmark_identity(
        full,
        label="clean full-stack projection trace",
        require_team=True,
        require_opponent=True,
    )

    full = full.sort_values(keys).reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(a.out, index=False)
    print(f"[clean_full_stack_trace] wrote {len(full)} rows -> {a.out}")
    print(f"[clean_full_stack_trace] identity={identity}")
    print(full.groupby(["season", "market"]).size().rename("rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
