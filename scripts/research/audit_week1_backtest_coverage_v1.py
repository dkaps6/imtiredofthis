#!/usr/bin/env python3
"""Coverage + miss audit for the Week 1 real-board backtest.

Answers three completeness questions before trusting the grading summary:
1. team/game coverage of the raw priced board vs the real Week 1 schedule
   (including whether the Thursday-night game is present, given the source
   board was captured the prior Friday);
2. how many priced rows are missing a vegas_line (no market to grade against);
3. how many graded bet rows failed to match a real actual-stat row, and why
   (player truly absent from weekly stats that week vs a join/key problem).

Then produces a biggest-misses breakdown by market (proxy for position group)
and by game, using the already-graded detail CSV.

Source-only. No production change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priced-file", type=Path, required=True)
    ap.add_argument("--graded-detail", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    import nflreadpy as nfl
    from scripts._opponent_map import canon_team

    priced = pd.read_csv(args.priced_file, low_memory=False)
    priced.columns = [c.strip().lower() for c in priced.columns]

    print("=" * 70)
    print("1) TEAM / GAME COVERAGE")
    print("=" * 70)

    sched_raw = nfl.load_schedules(args.season)
    sched = _to_pandas(sched_raw)
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    wk = sched.loc[sched["week"].eq(args.week) & sched["game_type"].astype(str).str.upper().isin(["REG", "REGULAR", ""])].copy()
    wk["home_team"] = wk["home_team"].map(canon_team)
    wk["away_team"] = wk["away_team"].map(canon_team)
    print(f"real season={args.season} week={args.week} schedule: {len(wk)} games, "
          f"{len(set(wk.home_team) | set(wk.away_team))} teams")
    for r in wk.sort_values("gameday").itertuples(index=False):
        gd = getattr(r, "gameday", "?")
        gt = getattr(r, "gametime", "?")
        print(f"  {gd} {gt}  {r.away_team} @ {r.home_team}")

    board_teams = set(priced["team"].map(canon_team).dropna()) if "team" in priced.columns else set()
    board_opps = set(priced["opponent"].map(canon_team).dropna()) if "opponent" in priced.columns else set()
    board_all_teams = board_teams | board_opps
    sched_teams = set(wk.home_team) | set(wk.away_team)
    missing_from_board = sched_teams - board_all_teams
    extra_in_board = board_all_teams - sched_teams
    print(f"\nteams referenced in priced board (team+opponent cols): {len(board_all_teams)}")
    print(f"real Week {args.week} schedule teams NOT referenced anywhere in board: {sorted(missing_from_board) or 'NONE'}")
    print(f"teams referenced in board but NOT in real Week {args.week} schedule: {sorted(extra_in_board) or 'NONE'}")

    # Per-team row counts, to spot a team with real odds but suspiciously thin coverage.
    if "team" in priced.columns:
        counts = priced["team"].map(canon_team).value_counts()
        print("\npriced rows per team (lowest 8, to check for thin/partial coverage):")
        print(counts.sort_values().head(8))

    print("\n" + "=" * 70)
    print("2) VEGAS LINE COMPLETENESS")
    print("=" * 70)
    total_rows = len(priced)
    if "vegas_line" in priced.columns:
        missing_line = priced["vegas_line"].isna().sum()
        print(f"total priced rows: {total_rows}")
        print(f"rows missing vegas_line entirely: {missing_line} ({missing_line/total_rows:.2%})")
        if missing_line:
            print("markets with missing vegas_line:")
            print(priced.loc[priced["vegas_line"].isna(), "market"].value_counts())
    else:
        print("NO vegas_line COLUMN FOUND -- cannot check")

    # model_proj present but no book/odds at all for that player-market (i.e.
    # the model priced someone the sportsbook never offered a prop for).
    if {"model_proj", "vegas_line"}.issubset(priced.columns):
        has_model_no_line = priced["model_proj"].notna() & priced["vegas_line"].isna()
        print(f"\nrows with a model projection but NO vegas line at all: {int(has_model_no_line.sum())}")

    print("\ndistinct players priced (any market):", priced["player_clean_key"].nunique() if "player_clean_key" in priced.columns else "n/a")
    print("distinct player-market combos priced:", len(priced[["player_clean_key", "market"]].drop_duplicates()) if "player_clean_key" in priced.columns else "n/a")

    print("\n" + "=" * 70)
    print("3) ACTUAL-OUTCOME MATCH COMPLETENESS")
    print("=" * 70)
    # IMPORTANT: the graded-detail CSV the grading script writes is already
    # filtered to has_actual==True rows only (grade_matched_rows() drops
    # unmatched rows before writing it) -- reading it back can never show an
    # unmatched row. Reconstruct the true pre-filter population instead by
    # calling the grading script's own functions directly on the archived
    # board + real actual stats, so "0 unmatched" can't be a false read of
    # an already-filtered file.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.operations.grade_market_track_record_v1 import (
        load_boards, select_model_bet, load_actual_stats, match_bets_to_actuals,
    )

    board = load_boards(args.season, [args.week])
    bets = select_model_bet(board)
    actual = load_actual_stats(args.season, [args.week])
    full_detail = match_bets_to_actuals(bets, actual)
    full_detail.columns = [c.strip().lower() for c in full_detail.columns]

    unmatched = full_detail.loc[~full_detail["has_actual"]] if "has_actual" in full_detail.columns else pd.DataFrame()
    print(f"total selected bet rows across the 5 gradeable markets: {len(full_detail)}")
    print(f"rows with NO matching actual-stat row: {len(unmatched)}")
    if len(unmatched):
        print("\nunmatched rows (player/team/market/week) -- checking why:")
        cols = [c for c in ["player", "team", "opponent", "market", "week", "season"] if c in unmatched.columns]
        print(unmatched[cols].drop_duplicates().to_string(index=False))

        # For each unmatched player, check if they appear in weekly stats AT ALL
        # this season (any week), to distinguish "did not play this week" from
        # "genuine identity/join gap".
        if "player_clean_key" in unmatched.columns:
            season_stats = load_actual_stats(args.season, [])
            season_stats.columns = [c.strip().lower() for c in season_stats.columns]
            for key in unmatched["player_clean_key"].dropna().unique():
                rows = season_stats.loc[season_stats["player_clean_key"].eq(key)] if "player_clean_key" in season_stats.columns else pd.DataFrame()
                if rows.empty:
                    print(f"  {key}: 0 rows anywhere in {args.season} weekly stats under this normalized key "
                          f"(genuine identity/key gap -- worth checking manually)")
                else:
                    weeks_played = sorted(pd.to_numeric(rows.get("week"), errors="coerce").dropna().astype(int).unique().tolist())
                    print(f"  {key}: appears in weeks {weeks_played} this season (not week {args.week} -> inactive/DNP/bye that week)")

    # Read the actual graded-detail CSV for the biggest-misses section below
    # (correct to use here since it's already the has_actual==True subset,
    # which is exactly what biggest-misses needs).
    detail = pd.read_csv(args.graded_detail, low_memory=False)
    detail.columns = [c.strip().lower() for c in detail.columns]

    print("\n" + "=" * 70)
    print("4) BIGGEST MISSES")
    print("=" * 70)
    graded = detail.loc[detail.get("has_actual", pd.Series(dtype=bool)).fillna(False)].copy() if "has_actual" in detail.columns else detail.copy()
    if "model_error" not in graded.columns and {"model_proj", "actual"}.issubset(graded.columns):
        graded["model_error"] = graded["model_proj"] - graded["actual"]
    graded["abs_model_error"] = graded["model_error"].abs()

    print("\n--- Top 20 single-row biggest misses (by |model_proj - actual|) ---")
    cols = [c for c in ["player", "team", "opponent", "market", "vegas_line", "model_proj", "actual", "model_error", "bet_result"] if c in graded.columns]
    top_misses = graded.sort_values("abs_model_error", ascending=False).head(20)
    print(top_misses[cols].to_string(index=False))

    print("\n--- MAE / bias by market (proxy for position group) ---")
    by_market = graded.groupby("market").agg(
        n=("abs_model_error", "size"),
        model_mae=("abs_model_error", "mean"),
        model_bias=("model_error", "mean"),
    )
    if "vegas_line" in graded.columns and "actual" in graded.columns:
        graded["vegas_error"] = graded["vegas_line"] - graded["actual"]
        by_market["vegas_mae"] = graded.groupby("market")["vegas_error"].apply(lambda s: s.abs().mean())
    print(by_market.sort_values("model_mae", ascending=False))

    if "team" in graded.columns and "opponent" in graded.columns:
        print("\n--- MAE by game (team+opponent pair, deduped) ---")
        graded["game_key"] = graded.apply(lambda r: "_".join(sorted([str(r["team"]), str(r["opponent"])])), axis=1)
        by_game = graded.groupby("game_key").agg(n=("abs_model_error", "size"), model_mae=("abs_model_error", "mean"))
        print(by_game.sort_values("model_mae", ascending=False).head(20))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
