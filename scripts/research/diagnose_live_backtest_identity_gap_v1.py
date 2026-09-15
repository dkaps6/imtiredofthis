#!/usr/bin/env python3
"""Diagnostic (not production): for the specific players that failed to match
a real actual-stat row in the Week 1 real-board backtest, compare the priced
board's player display name against nflreadpy's live 2026-season weekly
stats AND weekly rosters for that same team, to check whether this is the
same canonicalization-drift bug class found in the WR-R15 historical
research (a real player present under a different name/key) versus a
genuine data-availability gap.

Source-only. No production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas


def _to_pd(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priced-file", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    priced = pd.read_csv(args.priced_file, low_memory=False)
    priced.columns = [c.strip().lower() for c in priced.columns]
    priced["team"] = priced["team"].map(canon_team)
    priced["opponent"] = priced["opponent"].map(canon_team)

    unresolved_keys = [
        "brianrobinsonjr", "travisetiennejr", "charliekolar", "odellbeckhamjr",
        "calvinridley", "michaelpittmanjr", "jalentolbert", "chrisgodwin",
    ]
    targets = priced.loc[priced["player_clean_key"].isin(unresolved_keys),
                          ["player", "player_clean_key", "team", "opponent"]].drop_duplicates()
    print(f"unresolved players on the board: {len(targets)}")
    print(targets.to_string(index=False))

    import nflreadpy as nfl
    raw_stats = nfl.load_player_stats(seasons=[args.season], summary_level="week")
    stats = _normalize_weekly(_to_pandas(raw_stats), args.season)
    stats["team"] = stats["team"].map(canon_team)
    stats.columns = [c.strip().lower() for c in stats.columns]

    raw_roster = nfl.load_rosters_weekly(args.season)
    roster = _to_pd(raw_roster)
    roster.columns = [str(c).strip().lower() for c in roster.columns]

    print(f"\ntotal weekly stat rows loaded for {args.season}: {len(stats)}")
    print(f"total weekly roster rows loaded for {args.season}: {len(roster)}")
    print(f"weeks present in weekly stats: {sorted(pd.to_numeric(stats['week'], errors='coerce').dropna().astype(int).unique().tolist())}")

    for r in targets.itertuples(index=False):
        print(f"\n=== {r.player!r} ({r.player_clean_key!r}), board team={r.team}, opponent={r.opponent} ===")

        # 1) exact-team stat rows this season, any week -- catches suffix/format drift
        team_stats = stats.loc[stats["team"].eq(r.team)]
        print(f"  weekly-stats rows for team={r.team} this season: {len(team_stats)}")
        # look for anything with a similar-sounding key (same first token)
        first_token = r.player.split()[0].lower()
        close = team_stats.loc[team_stats["player"].astype(str).str.lower().str.startswith(first_token)]
        if len(close):
            print(f"  candidates on {r.team} whose display name starts with {first_token!r}:")
            for c in close[["player", "player_clean_key", "week"]].drop_duplicates().itertuples(index=False):
                print(f"    player={c.player!r} key={c.player_clean_key!r} week={c.week}")
        else:
            print(f"  no {r.team} stat rows starting with {first_token!r} at all this season")

        # 2) roster check: is this person even on the roster under any spelling?
        if "team" in roster.columns:
            rteam_col = "team" if "team" in roster.columns else None
            rost_team = roster.loc[roster[rteam_col].map(canon_team).eq(r.team)] if rteam_col else roster
        else:
            rost_team = roster
        name_col = next((c for c in ("full_name", "football_name", "player_name", "player") if c in rost_team.columns), None)
        if name_col:
            rmatches = rost_team.loc[rost_team[name_col].astype(str).str.lower().str.contains(first_token, na=False)]
            if len(rmatches):
                cols = [c for c in (name_col, "position", "status", "week") if c in rmatches.columns]
                print(f"  roster rows on {r.team} matching {first_token!r}:")
                print(rmatches[cols].drop_duplicates().head(5).to_string(index=False))
            else:
                print(f"  NO roster row on {r.team} matching {first_token!r} at all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
