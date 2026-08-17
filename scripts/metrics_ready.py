#!/usr/bin/env python3
"""Pre-pricing readiness validation for Full Slate."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.artifact_contracts import CONTRACTS
from scripts.runtime_context import resolve_season, resolve_week

REQUIRED_BEFORE_PRICING = (
    "roles_ourlads",
    "team_week_map",
    "opponent_map",
    "props_raw",
    "team_form",
    "player_game_logs",
    "player_form",
    "player_form_consensus",
    "metrics_ready",
)


def _check(key: str) -> pd.DataFrame:
    c = CONTRACTS[key]
    path = c.path
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Required artifact missing or empty: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read required artifact {path}: {exc}") from exc
    if len(frame) < c.min_rows:
        raise RuntimeError(f"{path} has {len(frame)} rows; expected >= {c.min_rows}")
    missing = [col for col in c.required_columns if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")
    print(f"[metrics_ready] ✓ {path} rows={len(frame)}")
    return frame


def validate_metrics_ready_csv(path: Path | None = None, *, season: int | None = None, week: int | None = None) -> None:
    contract = CONTRACTS["metrics_ready"]
    target = path or contract.path
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"metrics_ready.csv missing or empty at {target}")
    frame = pd.read_csv(target)
    missing = [c for c in contract.required_columns if c not in frame.columns]
    if missing:
        raise RuntimeError("metrics_ready.csv missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise RuntimeError("metrics_ready.csv has 0 rows")

    season = int(season if season is not None else resolve_season())
    s = pd.to_numeric(frame["season"], errors="coerce")
    scoped = frame.loc[s.eq(season)].copy()
    if scoped.empty:
        raise RuntimeError(f"metrics_ready.csv contains no rows for active season {season}")

    if week is not None and "week" in scoped.columns:
        w = pd.to_numeric(scoped["week"], errors="coerce")
        if not w.eq(int(week)).any():
            raise RuntimeError(f"metrics_ready.csv contains no rows for active week {week}")

    usage = [c for c in ("target_share", "tgt_share", "rush_share") if c in scoped.columns]
    if not usage:
        raise RuntimeError("metrics_ready.csv contains no player usage columns")
    odds = [c for c in ("over_odds", "under_odds") if c in scoped.columns]
    if not odds:
        raise RuntimeError("metrics_ready.csv must contain over_odds and/or under_odds")

    # route_rate/YPRR are deliberately not required to be non-null.  PlayerForm
    # v2 refuses to fabricate routes when the source does not provide them.
    for col in ("player", "team", "opponent"):
        missing_count = int(scoped[col].isna().sum()) if col in scoped.columns else len(scoped)
        if missing_count:
            raise RuntimeError(f"metrics_ready.csv has {missing_count} missing {col} values")

    print(f"[metrics_ready] ✅ validated rows={len(scoped)} season={season} week={week if week is not None else '<any>'}")


def check_required_inputs() -> None:
    for key in REQUIRED_BEFORE_PRICING:
        _check(key)
    for key, contract in CONTRACTS.items():
        if contract.required or key in REQUIRED_BEFORE_PRICING:
            continue
        if not contract.path.exists() or contract.path.stat().st_size == 0:
            print(f"[metrics_ready] WARN optional artifact missing/empty: {contract.path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--date", default="")
    args = parser.parse_args(argv)
    season = int(args.season if args.season is not None else resolve_season())
    week = resolve_week(season=season, slate_date=args.date or "")
    check_required_inputs()
    validate_metrics_ready_csv(season=season, week=week)


if __name__ == "__main__":
    main()
