#!/usr/bin/env python3
"""Diagnostic (not part of the frozen Phase-4B pipeline): for a sample of
WR2+ feature identities that Phase-4B's preflight reports as
UNRESOLVED_WEEKLY_IDENTITY, print the feature file's player display name
next to every WR-position player nflreadpy's weekly stats has for that same
(season, week, team) -- to distinguish "genuinely absent from the source"
from "present under a different player_clean_key" (a canonicalization
drift between the frozen R15 artifact and the current live pipeline).

Source-only. No receiving-yard outcome interpretation, no production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

WR_POS = {"WR", "LWR", "RWR", "SWR"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=25)
    args = ap.parse_args()

    feat = pd.read_csv(
        args.features,
        usecols=["event_id", "player", "player_clean_key", "team", "season", "week", "baseline_wr_rank"],
    )
    feat["season"] = pd.to_numeric(feat["season"], errors="raise").astype(int)
    feat["week"] = pd.to_numeric(feat["week"], errors="raise").astype(int)
    feat["baseline_wr_rank"] = pd.to_numeric(feat["baseline_wr_rank"], errors="raise").astype(int)
    feat["team"] = feat["team"].map(canon_team)
    feat["player_clean_key"] = feat["player_clean_key"].astype(str)
    feat = feat.loc[feat["baseline_wr_rank"].ge(2)].copy()
    print(f"WR2+ feature rows: {len(feat)}")

    import nflreadpy as nfl
    frames = []
    for season in (2023, 2024):
        raw = nfl.load_player_stats(seasons=[season], summary_level="week")
        x = _normalize_weekly(_to_pandas(raw), season)
        x = x.loc[pd.to_numeric(x["week"], errors="coerce").between(1, 18)].copy()
        x["season"] = season
        x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
        x["team"] = x["team"].map(canon_team)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
        x["position"] = x["position"].astype("string").fillna("").str.upper().str.strip()
        frames.append(x[["season", "week", "team", "player_clean_key", "player", "position", "targets"]])
    weekly = pd.concat(frames, ignore_index=True)
    print(f"weekly stat rows (2023+2024): {len(weekly)}")

    key_cols = ["season", "week", "team", "player_clean_key"]
    merged = feat.merge(
        weekly[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True
    )
    unresolved = merged.loc[merged["_merge"].eq("left_only")].drop(columns=["_merge"])
    print(f"unresolved WR2+ identities: {len(unresolved)} / {len(feat)} ({len(unresolved)/len(feat):.2%})")

    sample = unresolved.sample(n=min(args.sample, len(unresolved)), random_state=20260915)
    print(f"\n=== sampling {len(sample)} unresolved identities against same-team-game WR roster ===\n")
    exact_name_present_diff_key = 0
    fuzzy_close = 0
    no_wr_at_all_that_game = 0
    for r in sample.itertuples(index=False):
        roster = weekly.loc[
            weekly["season"].eq(r.season) & weekly["week"].eq(r.week)
            & weekly["team"].eq(r.team) & weekly["position"].isin(WR_POS)
        ][["player", "player_clean_key", "targets"]]
        print(f"UNRESOLVED: season={r.season} week={r.week} team={r.team} "
              f"feature_player={r.player!r} feature_key={r.player_clean_key!r}")
        if roster.empty:
            print("  -> NO WR-position players at all in weekly stats for this team-game (bye/no-data/roster gap)")
            no_wr_at_all_that_game += 1
        else:
            print(f"  weekly WR roster for this team-game ({len(roster)} players):")
            for wr in roster.itertuples(index=False):
                same_name = str(wr.player).strip().lower() == str(r.player).strip().lower()
                marker = "  <-- EXACT NAME MATCH, DIFFERENT KEY" if same_name else ""
                print(f"    player={wr.player!r} key={wr.player_clean_key!r} targets={wr.targets}{marker}")
                if same_name:
                    exact_name_present_diff_key += 1
        print()

    print("=== summary over sample ===")
    print(f"exact display-name match present under a DIFFERENT player_clean_key: {exact_name_present_diff_key}")
    print(f"no WR-position player at all for that team-game in weekly stats: {no_wr_at_all_that_game}")
    print(f"sample size: {len(sample)}")
    if exact_name_present_diff_key > 0:
        print("\n=> CONFIRMS a player_clean_key canonicalization drift between the R15 artifact and the "
              "current live _normalize_weekly/canonicalize_player_name_safe pipeline is at least PART of the cause.")
    if no_wr_at_all_that_game == len(sample) and exact_name_present_diff_key == 0:
        print("\n=> No evidence of key drift in this sample; unresolved rows look like genuine source gaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
