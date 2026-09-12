#!/usr/bin/env python3
"""Attach authoritative game_id to the M89/M90 QB synthesis test-season trace.

This is the missing link identified in Issue #535: run_m89_pregame_synthesis.py
emits m89_2024_2025_synthesis_trace.csv (base_proj/football_synthesis/
market_assisted predictions) with NO game_id and NO market column, so it can
never be joined to real Vegas props via grade_full_stack_vegas_benchmark_v1.py.
No committed script ever bridged this gap -- the previously-committed
qb_synthesis_summary.csv was produced by an untracked, since-lost ad-hoc step
(the same class of bug independently found and fixed for the non-QB benchmark
in Issue #535 / PR #541).

This script closes that gap using the exact same fail-closed pattern PR #541
used for non-QB: never trust any pre-existing game_id (there isn't one here),
attach it only from the authoritative historical schedule on
(season, week, team), and hard-fail on any unresolved or opponent-inconsistent
row before anything downstream can silently grade against the wrong game.

Research only. No production, model, weight, or threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True, help="m89_2024_2025_synthesis_trace.csv")
    ap.add_argument("--schedule", type=Path, required=True, help="authoritative combined historical schedule")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    trace = _read(a.trace, "M89/M90 synthesis trace")
    required = {"season", "week", "team", "opponent", "player_clean_key", "actual_pass_yards", "base_proj", "football_synthesis", "market_assisted", "mc_proj", "ml_proj", "state_proj"}
    missing = sorted(required - set(trace.columns))
    if missing:
        raise RuntimeError(f"QB synthesis trace missing columns: {missing}")

    trace["season"] = pd.to_numeric(trace["season"], errors="raise").astype(int)
    trace["week"] = pd.to_numeric(trace["week"], errors="raise").astype(int)
    trace["team"] = trace["team"].map(canon_team)
    trace["opponent"] = trace["opponent"].map(canon_team)
    if trace["team"].eq("").any() or trace["opponent"].eq("").any():
        raise RuntimeError("QB synthesis trace contains uncanonicalizable team/opponent")

    keys = ["season", "week", "team", "player_clean_key"]
    if trace.duplicated(keys).any():
        sample = trace.loc[trace.duplicated(keys, keep=False), keys].head(20).to_dict(orient="records")
        raise RuntimeError(f"QB synthesis trace duplicate identity rows: {sample}")

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
        raise RuntimeError("historical schedule duplicate team-week rows")
    assert_benchmark_identity(schedule, label="authoritative historical schedule", require_team=True, require_opponent=True)

    # Never trust or preserve a pre-existing game_id -- there isn't one on
    # this trace, but this guard makes the invariant explicit and permanent.
    out = trace.drop(columns=["game_id"], errors="ignore")
    out = out.rename(columns={"opponent": "trace_opponent"})
    out = out.merge(
        schedule.rename(columns={"opponent": "schedule_opponent"}),
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if out["game_id"].isna().any() or out["game_id"].astype(str).str.strip().eq("").any():
        sample = out.loc[
            out["game_id"].isna() | out["game_id"].astype(str).str.strip().eq(""),
            ["season", "week", "team", "player_clean_key"],
        ].head(20).to_dict(orient="records")
        raise RuntimeError(f"authoritative schedule failed to resolve game_id for QB synthesis rows: {sample}")

    mismatch = out["trace_opponent"].ne(out["schedule_opponent"])
    if mismatch.any():
        sample = out.loc[
            mismatch, ["season", "week", "team", "trace_opponent", "schedule_opponent", "game_id"],
        ].head(20).to_dict(orient="records")
        raise RuntimeError(f"QB trace/schedule opponent mismatch: {sample}")

    out["opponent"] = out["schedule_opponent"]
    out = out.drop(columns=["trace_opponent", "schedule_opponent"])
    out["market"] = "pass_yards"
    out["actual"] = pd.to_numeric(out["actual_pass_yards"], errors="coerce")

    identity = assert_benchmark_identity(out, label="QB synthesis trace with attached game identity", require_team=True, require_opponent=True)

    out = out.sort_values(keys).reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[qb_synthesis_identity] wrote {len(out)} rows -> {a.out}")
    print(f"[qb_synthesis_identity] identity={identity}")
    print(out.groupby(["season"]).size().rename("rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
