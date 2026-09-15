#!/usr/bin/env python3
"""Remove verified, final-board-only prop quarantines from the priced output.

Some players have a real roster/PlayerForm entry and price normally through
the whole football-only simulation stack, but have an unresolved, verified
identity/authority conflict for this specific week (see
data/manual_final_board_quarantine.csv) that makes showing their props on the
published board unsafe -- for example a live starter-authority announcement
that conflicts with the injury report, with no clean way to know who is
actually playing. Unlike data/manual_prop_quarantine.csv (which removes a
player's offers before metrics/pricing, for a player the football-only
universe has no entry for at all), this quarantine is applied only to the
final priced board, after every per-team completeness check the pricing
stack enforces has already run against the full, unfiltered board -- so the
rest of the slate, including that player's own team's other markets, still
prices exactly as it would without this file.

Every row is scoped to an explicit season/week and matched by (team, player)
together, not player name alone -- a quarantine is never allowed to silently
carry over into a different week/season, or to suppress a same-named player
on an unrelated team.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.repair_live_prop_identity_v1 import _name_keys
from scripts.runtime_context import resolve_season, resolve_slate_date, resolve_week

OUT = Path("outputs/props_priced_clean.csv")
QUARANTINE = Path("data/manual_final_board_quarantine.csv")
STATUS = Path("data/final_priced_props_quarantine_status.json")


def _load_quarantine_keys(
    path: Path = QUARANTINE, *, season: int | None = None, week: int | None = None
) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    required_cols = {"player", "team", "season", "week", "reason", "verified_source", "verified_date"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"final-board prop quarantine missing columns: {sorted(missing_cols)}")
    if not isinstance(df.index, pd.RangeIndex):
        raise RuntimeError(
            "final-board prop quarantine parsed with a non-default index; this usually means an "
            "unquoted extra comma in a field (e.g. reason) shifted a row"
        )
    if df.empty:
        return set()

    season = int(season if season is not None else resolve_season())
    slate = resolve_slate_date()
    week = int(week if week is not None else resolve_week(season=season, slate_date=slate))

    df_season = pd.to_numeric(df["season"], errors="coerce")
    df_week = pd.to_numeric(df["week"], errors="coerce")
    if df_season.isna().any() or df_week.isna().any():
        raise RuntimeError("final-board prop quarantine contains non-numeric season/week")
    scoped = df.loc[df_season.eq(season) & df_week.eq(week)]
    if scoped.empty:
        return set()

    keys: set[tuple[str, str]] = set()
    for row in scoped.itertuples(index=False):
        team = canon_team(getattr(row, "team"))
        for key in _name_keys(getattr(row, "player")):
            keys.add((team, key))
    return keys


def quarantine_final_priced_props(
    *,
    out_path: Path = OUT,
    quarantine_path: Path = QUARANTINE,
    status_path: Path = STATUS,
    season: int | None = None,
    week: int | None = None,
) -> dict:
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"required priced output missing/empty: {out_path}")
    df = pd.read_csv(out_path, low_memory=False)
    if "player" not in df.columns or "team" not in df.columns:
        raise RuntimeError("priced output missing player/team columns")

    quarantine_keys = _load_quarantine_keys(quarantine_path, season=season, week=week)
    removed = 0
    if quarantine_keys:
        teams = df["team"].map(canon_team)
        name_keys = df["player"].map(_name_keys)
        mask = [
            bool({(team, key) for key in keys} & quarantine_keys)
            for team, keys in zip(teams, name_keys)
        ]
        mask = pd.Series(mask, index=df.index)
        removed = int(mask.sum())
        if removed:
            df = df.loc[~mask].copy()
            df.to_csv(out_path, index=False)

    status = {
        "quarantine_source": str(quarantine_path),
        "quarantined_team_name_keys": sorted(f"{team}:{key}" for team, key in quarantine_keys),
        "rows_removed": removed,
        "rows_remaining": int(len(df)),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[quarantine_final_priced_props] " + json.dumps(status, sort_keys=True))
    return status


def main() -> int:
    quarantine_final_priced_props()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
