#!/usr/bin/env python3
"""Authoritative M75 runner with the observed nflverse PFR-def schema adapter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_personnel_tracking_matchup as m
from scripts.backtest import run_qb_personnel_tracking_matchup as runner


def build_pfr_secondary_week_fixed(pfr, players):
    if pfr.empty:
        return pd.DataFrame(), {"usable": False, "reason": "pfr_empty"}
    q = m.regular_week_rows(pfr)
    required = ["team", "def_targets", "def_yards_allowed_per_tgt"]
    missing = [c for c in required if c not in q]
    if missing:
        return pd.DataFrame(), {
            "usable": False,
            "reason": f"missing_required_pfr_coverage_columns:{'|'.join(missing)}",
            "columns": "|".join(q.columns),
        }

    q, pos_source = m.attach_positions(q, players)
    q["team"] = q.team.map(m.canon)
    q["targets_n"] = m.num(q.def_targets).fillna(0)
    q["ypt_n"] = m.num(q.def_yards_allowed_per_tgt)
    q["cmp_pct_n"] = m.num(q.def_completion_pct) if "def_completion_pct" in q else np.nan
    q["rating_n"] = m.num(q.def_passer_rating_allowed) if "def_passer_rating_allowed" in q else np.nan
    q["adot_n"] = m.num(q.def_adot) if "def_adot" in q else np.nan
    q["yac_n"] = m.num(q.def_yards_after_catch) if "def_yards_after_catch" in q else np.nan

    pos = q.position.fillna("").astype(str).str.upper() if "position" in q else pd.Series("", index=q.index)
    dbmask = pos.isin(["CB", "DB", "S", "FS", "SS", "NB"])
    position_filtered = bool(dbmask.any())
    if position_filtered:
        q = q[dbmask].copy()

    rows = []
    for (season, week, team), g in q.groupby(["season", "week", "team"], dropna=False):
        g = g[g.targets_n.gt(0)].copy()
        if g.empty:
            continue
        meaningful = g[g.targets_n.ge(2)]
        rows.append({
            "season": int(season),
            "week": int(week),
            "team": team,
            "db_targets": float(g.targets_n.sum()),
            "db_ypt": m.wmean(g.ypt_n, g.targets_n),
            "db_cmp_pct": m.wmean(g.cmp_pct_n, g.targets_n),
            "db_rating": m.wmean(g.rating_n, g.targets_n),
            "db_adot": m.wmean(g.adot_n, g.targets_n),
            "db_yac": m.wmean(g.yac_n, g.targets_n),
            "db_weak_ypt": float(m.num(meaningful.ypt_n).max()) if len(meaningful) else np.nan,
            "db_weak_rating": float(m.num(meaningful.rating_n).max()) if len(meaningful) else np.nan,
            "db_coverage_players": int(len(g)),
        })

    return pd.DataFrame(rows), {
        "usable": bool(len(rows)),
        "reason": "ok" if len(rows) else "no_team_week_rows",
        "rows": len(rows),
        "position_filtered": position_filtered,
        "position_source": pos_source,
        "target_col": "def_targets",
        "ypt_col": "def_yards_allowed_per_tgt",
        "cmp_pct_col": "def_completion_pct",
        "rating_col": "def_passer_rating_allowed",
        "adot_col": "def_adot",
        "yac_col": "def_yards_after_catch",
    }


# Adapter-only correction before any PFR/interaction scientific result exists.
m.build_pfr_secondary_week = build_pfr_secondary_week_fixed

if __name__ == "__main__":
    raise SystemExit(runner.main())
