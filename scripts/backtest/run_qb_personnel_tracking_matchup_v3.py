#!/usr/bin/env python3
"""Authoritative M75 runner with count-correct PFR defensive coverage aggregation.

PFR coverage YAC is a total, while Yds/Tgt, Cmp%, Rat and DADOT are rates.
This adapter reconstructs team-secondary weekly features from underlying DB
coverage counts before the frozen strictly-prior M75 transforms are built.
No M75 model, target, threshold or family definition changes here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_personnel_tracking_matchup as m
from scripts.backtest import run_qb_personnel_tracking_matchup as runner


def passer_rating(cmp_, att, yds, td, ints):
    if not np.isfinite(att) or att <= 0:
        return np.nan
    a = ((cmp_ / att) - 0.3) * 5.0
    b = ((yds / att) - 3.0) * 0.25
    c = (td / att) * 20.0
    d = 2.375 - ((ints / att) * 25.0)
    vals = [min(2.375, max(0.0, float(x))) for x in [a, b, c, d]]
    return float(sum(vals) / 6.0 * 100.0)


def build_pfr_secondary_week_fixed(pfr, players):
    if pfr.empty:
        return pd.DataFrame(), {"usable": False, "reason": "pfr_empty"}
    q = m.regular_week_rows(pfr)
    required = [
        "team", "def_targets", "def_completions_allowed", "def_yards_allowed",
        "def_yards_allowed_per_tgt", "def_receiving_td_allowed",
        "def_yards_after_catch", "def_adot",
    ]
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
    q["cmp_n"] = m.num(q.def_completions_allowed).fillna(0)
    q["yards_n"] = m.num(q.def_yards_allowed).fillna(0)
    q["td_n"] = m.num(q.def_receiving_td_allowed).fillna(0)
    q["int_n"] = m.num(q.def_ints).fillna(0) if "def_ints" in q else 0.0
    q["yac_total_n"] = m.num(q.def_yards_after_catch).fillna(0)
    q["adot_n"] = m.num(q.def_adot)
    q["individual_ypt"] = m.num(q.def_yards_allowed_per_tgt)
    q["individual_rating"] = m.num(q.def_passer_rating_allowed) if "def_passer_rating_allowed" in q else np.nan

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
        tgt = float(g.targets_n.sum())
        cmp_ = float(g.cmp_n.sum())
        yds = float(g.yards_n.sum())
        td = float(g.td_n.sum())
        ints = float(g.int_n.sum())
        yac = float(g.yac_total_n.sum())
        meaningful = g[g.targets_n.ge(2)]
        rows.append({
            "season": int(season),
            "week": int(week),
            "team": team,
            "db_targets": tgt,
            "db_ypt": float(yds / tgt) if tgt > 0 else np.nan,
            "db_cmp_pct": float(cmp_ / tgt) if tgt > 0 else np.nan,
            "db_rating": passer_rating(cmp_, tgt, yds, td, ints),
            "db_adot": m.wmean(g.adot_n, g.targets_n),
            # PFR YAC is a total; normalize by completions so the frozen
            # NGS YACOE x defensive-YAC interaction compares rate-like traits.
            "db_yac": float(yac / cmp_) if cmp_ > 0 else np.nan,
            "db_weak_ypt": float(m.num(meaningful.individual_ypt).max()) if len(meaningful) else np.nan,
            "db_weak_rating": float(m.num(meaningful.individual_rating).max()) if len(meaningful) else np.nan,
            "db_coverage_players": int(len(g)),
        })

    return pd.DataFrame(rows), {
        "usable": bool(len(rows)),
        "reason": "ok" if len(rows) else "no_team_week_rows",
        "rows": len(rows),
        "position_filtered": position_filtered,
        "position_source": pos_source,
        "aggregation": "DB-only counts: ypt=yards/targets; cmp%=cmp/targets; rating=reconstructed; adot=target-weighted; yac=total_yac/completions",
        "target_col": "def_targets",
        "completion_col": "def_completions_allowed",
        "yards_col": "def_yards_allowed",
        "td_col": "def_receiving_td_allowed",
        "int_col": "def_ints",
        "adot_col": "def_adot",
        "yac_col": "def_yards_after_catch",
    }


m.build_pfr_secondary_week = build_pfr_secondary_week_fixed

if __name__ == "__main__":
    raise SystemExit(runner.main())
