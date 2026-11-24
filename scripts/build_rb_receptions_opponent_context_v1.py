#!/usr/bin/env python3
"""Build data/rb_receptions_opponent_context.csv for the slate week (RB_RECEPTIONS_OPPONENT_ADJ_V1).

Strictly prior: only this season's weeks before the slate week. Week 1 writes an empty-but-valid
context (no prior games, so every D is 0 and the adjustment is a no-op).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.rb_receptions_opponent_adj_v1 import CONTEXT, VERSION, build_context
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.runtime_context import resolve_season, resolve_slate_date, resolve_week


def _opponents(season: int) -> pd.DataFrame:
    m = pd.read_csv("data/team_week_map.csv")
    m.columns = [str(c).lower() for c in m.columns]
    m = m[pd.to_numeric(m["season"], errors="coerce") == int(season)].copy()
    m["team"] = m["team"].map(canon_team); m["opponent"] = m["opponent"].map(canon_team)
    m["season"] = pd.to_numeric(m["season"]).astype(int); m["week"] = pd.to_numeric(m["week"]).astype(int)
    return m[["season", "week", "team", "opponent"]].drop_duplicates()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    a = ap.parse_args()
    season = int(a.season if a.season is not None else resolve_season())
    week = int(a.week if a.week is not None else resolve_week(season=season, slate_date=resolve_slate_date() or ""))
    if week <= 1:
        ctx = pd.DataFrame(columns=["season", "week", "defense", "games", "allowed_pg", "league_pg", "D", "version"])
    else:
        st = load_weekly_player_stats(season)
        st.columns = [str(c).lower() for c in st.columns]
        team_col = "team" if "team" in st.columns else "recent_team"
        st = st.assign(team=st[team_col].map(canon_team), receptions=pd.to_numeric(st["receptions"], errors="coerce").fillna(0))
        ctx = build_context(st, _opponents(season), season, week)
        if ctx.empty or ctx["defense"].nunique() < 20:
            raise RuntimeError(f"{VERSION}: only {ctx['defense'].nunique() if not ctx.empty else 0} defenses with prior games for week {week}")
    CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    ctx.to_csv(CONTEXT, index=False)
    print(json.dumps({"version": VERSION, "season": season, "week": week, "defenses": int(ctx["defense"].nunique()) if len(ctx) else 0,
                      "D_min": float(ctx["D"].min()) if len(ctx) else 0.0, "D_max": float(ctx["D"].max()) if len(ctx) else 0.0,
                      "sportsbook_inputs_used": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
