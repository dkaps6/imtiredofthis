#!/usr/bin/env python3
"""Validate current-role PlayerForm against the resolved Full Slate target week.

The former workflow assertion was frozen to the Week-1 launch and rejected every
2026 Week-1-or-later game log. That is correct for a Week-1 target but wrong for
Week 2+, where completed earlier weeks are legitimate strictly-prior history.

This validator preserves the same no-leakage rule at the correct runtime grain:
for the active season, any game log from target_week or later is illegal; earlier
weeks are allowed. It also preserves the definitive-unavailable identity guard.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.runtime_context import resolve_season, resolve_week

AVAILABILITY = Path("data/current_player_availability.csv")
PLAYER_FORM = Path("data/player_form_consensus.csv")
GAME_LOGS = Path("data/player_game_logs.csv")


def validate(*, season: int, target_week: int) -> dict[str, int]:
    for path in (AVAILABILITY, PLAYER_FORM, GAME_LOGS):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing required PlayerForm audit artifact: {path}")

    availability = pd.read_csv(AVAILABILITY, low_memory=False)
    form = pd.read_csv(PLAYER_FORM, low_memory=False)
    logs = pd.read_csv(GAME_LOGS, low_memory=False)

    bad = set(
        zip(
            availability.loc[
                pd.to_numeric(availability["definitive_unavailable"], errors="coerce").fillna(0).eq(1),
                "team",
            ].astype(str),
            availability.loc[
                pd.to_numeric(availability["definitive_unavailable"], errors="coerce").fillna(0).eq(1),
                "player_clean_key",
            ].astype(str),
        )
    )
    got = set(zip(form["team"].astype(str), form["player_clean_key"].astype(str)))
    leaked_unavailable = sorted(bad & got)
    if leaked_unavailable:
        raise RuntimeError(f"unavailable PlayerForm identities: {leaked_unavailable[:20]}")

    if not {"season", "week"}.issubset(logs.columns):
        raise RuntimeError("player_game_logs missing season/week strict-prior keys")
    log_season = pd.to_numeric(logs["season"], errors="coerce")
    log_week = pd.to_numeric(logs["week"], errors="coerce")
    illegal = log_season.eq(int(season)) & log_week.ge(int(target_week))
    if illegal.fillna(False).any():
        sample_cols = [c for c in ("player", "player_clean_key", "team", "season", "week", "game_id") if c in logs.columns]
        sample = logs.loc[illegal.fillna(False), sample_cols].head(20).to_dict("records")
        raise RuntimeError(
            f"PlayerForm published current/future target-week history season={season} "
            f"target_week={target_week}: {sample}"
        )

    same_season_prior = log_season.eq(int(season)) & log_week.lt(int(target_week))
    result = {
        "target_season": int(season),
        "target_week": int(target_week),
        "same_season_prior_rows": int(same_season_prior.fillna(False).sum()),
        "illegal_current_or_future_rows": 0,
        "unavailable_identity_rows": 0,
    }
    print("PLAYERFORM_CURRENT_ROLES_AND_RUNTIME_STRICT_PRIOR_PASS", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    week = int(args.week if args.week is not None else resolve_week(season=season))
    validate(season=season, target_week=week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
