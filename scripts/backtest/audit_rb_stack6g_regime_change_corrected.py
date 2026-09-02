#!/usr/bin/env python3
"""Mechanical wrapper for STACK6G playcaller_recent_change boolean coercion.

Frozen protocol and support gates are unchanged. See
`docs/migrations/RB_STACK6G_IMPLEMENTATION_CORRECTION.md`.
"""
from __future__ import annotations

import pandas as pd

from scripts.backtest import audit_rb_stack6g_regime_change as base
from scripts.backtest import build_qb_playcaller_opening_leverage as m68


def corrected_playcaller_table(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for season in base.SEASONS:
        g = base.schedule_games(schedule, season).sort_values(["team", "week"])
        mapped = 0
        changes = 0
        for team, tg in g.groupby("team", sort=True):
            prev = ""
            tenure = 0
            for _, r in tg.sort_values("week").iterrows():
                caller = m68.caller_for(season, int(r.week), team)
                if caller:
                    mapped += 1
                changed = int(bool(prev and caller and caller != prev))
                if not caller:
                    tenure = 0
                elif caller == prev:
                    tenure += 1
                else:
                    tenure = 1
                if changed:
                    changes += 1
                recent_change = int(bool(changed or (bool(caller) and tenure in (2, 3) and prev == caller)))
                rows.append({
                    "season": season,
                    "week": int(r.week),
                    "team": team,
                    "target_playcaller": caller,
                    "prior_game_playcaller": prev,
                    "playcaller_changed": changed,
                    "playcaller_tenure_games": tenure,
                    "playcaller_recent_change": recent_change,
                })
                prev = caller
        summary.append({
            "season": season,
            "team_games": len(g),
            "mapped_team_games": mapped,
            "mapping_coverage": float(mapped / len(g)) if len(g) else 0.0,
            "documented_change_team_games": changes,
            "mapping_available": int(mapped > 0),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


base.playcaller_table = corrected_playcaller_table

if __name__ == "__main__":
    raise SystemExit(base.main())
