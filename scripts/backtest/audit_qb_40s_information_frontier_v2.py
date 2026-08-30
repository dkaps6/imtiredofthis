#!/usr/bin/env python3
"""M76 current-head hardening wrapper.

Keeps the frozen M76 science intact while requiring populated historical
measurements (not mere key-row existence) for strict-prior snap/PFR evidence.
"""
from __future__ import annotations

import pandas as pd

import audit_qb_40s_information_frontier as m76


def history_frame(df: pd.DataFrame, metric_kind: str):
    if df.empty:
        return pd.DataFrame(), False, []
    season = m76.first_col(df, ["season"])
    week = m76.first_col(df, ["week"])
    team = m76.first_col(df, ["team", "team_abbr"])
    player = m76.first_col(df, ["gsis_id"])
    if metric_kind == "snap":
        metrics_cols = [c for c in ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct"] if c in df.columns]
    else:
        keys = ("pressure", "hurr", "qb_hit", "sack", "blitz")
        metrics_cols = [c for c in df.columns if any(k in c for k in keys) and c.startswith("def_")]
        if len(metrics_cols) < 2:
            metrics_cols = [c for c in df.columns if any(k in c for k in keys)]
    ok = bool(all([season, week, team, player]) and len(metrics_cols) >= 1)
    if not ok:
        return pd.DataFrame(), False, metrics_cols

    out = pd.DataFrame({
        "season": m76.num(df[season]),
        "week": m76.num(df[week]),
        "team": df[team].map(m76.team_value),
        "player_id": m76.clean_id(df[player]),
    })
    metric_numeric = pd.DataFrame({c: m76.num(df[c]) for c in metrics_cols}, index=df.index)
    out["metric_nonnull"] = metric_numeric.notna().any(axis=1)
    out = out.dropna(subset=["season", "week", "team", "player_id"])
    out["season"] = out.season.astype(int)
    out["week"] = out.week.astype(int)
    key_nonnull = len(out) / len(df) if len(df) else 0.0
    populated_ratio = float(out.metric_nonnull.mean()) if len(out) else 0.0
    return out, bool(key_nonnull >= 0.90 and populated_ratio >= 0.80), metrics_cols


def has_prior(hist: pd.DataFrame, team: str, player: str, season: int, week: int) -> bool:
    if hist.empty or "metric_nonnull" not in hist.columns:
        return False
    q = hist.loc[
        hist.team.eq(team)
        & hist.player_id.eq(player)
        & hist.metric_nonnull.fillna(False)
    ]
    return bool(((q.season < season) | ((q.season == season) & (q.week < week))).any())


m76.history_frame = history_frame
m76.has_prior = has_prior

if __name__ == "__main__":
    raise SystemExit(m76.main())
