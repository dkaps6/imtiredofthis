#!/usr/bin/env python3
"""Grade the current production MC engine against real 2024-2025 Vegas lines.

Joins the free-archive historical market props (repaired reconciliation,
scripts/backtest/prepare_free_market_prop_archive_v1.py) to independently
built historical projections (scripts/backtest/build_market_vegas_benchmark_projections_v1.py)
and reuses the exact grading arithmetic from the forward market track record
grader so both the retroactive and forward answers are computed the same way.

Important scope note: the projection graded here is the shared Monte Carlo
opportunity/simulation engine underneath every position (the common
foundation every market's projection starts from), not the full live
pricing stack -- it does not include the ML/State ensemble blend or the
position-specific frozen overlays (QB M89/M90 synthesis, RB P3, RB R26),
which are either fit on top of this engine or qualified for a narrower
scope (P3/R26 are Week-1-2026-only). This is a real, honest measurement of
the common football engine, not a claim that it reproduces the exact
current live-pricing number for every market.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.operations.grade_market_track_record_v1 import (
    american_profit,
    edge_bucket,
    model_side,
    num,
    outcome_side,
)

BOOK_ORDER = {"draftkings": 0, "fanduel": 1}


def select_one_book_row(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return props.copy()
    p = props.copy()
    p["book"] = p.book.astype(str).str.lower().str.strip()
    p["book_rank"] = p.book.map(BOOK_ORDER).fillna(99)
    p["price_count"] = num(p.over_odds).notna().astype(int) + num(p.under_odds).notna().astype(int)
    p = p.sort_values(
        ["game_id", "player_clean_key", "market", "book_rank", "price_count"],
        ascending=[True, True, True, True, False],
    )
    return p.drop_duplicates(["game_id", "player_clean_key", "market"], keep="first")


def grade(proj: pd.DataFrame, props: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_one_book_row(props)
    join_cols = ["game_id", "player_clean_key", "market"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "player"]
    selected = selected[[c for c in keep if c in selected]].copy()

    z = proj.merge(selected, on=join_cols, how="inner")
    if z.empty:
        return z, pd.DataFrame()

    z["mc_proj"] = num(z.mc_proj)
    z["actual"] = num(z.actual)
    z["line"] = num(z.line)
    z["model_pick_side"] = [model_side(p, l) for p, l in zip(z.mc_proj, z.line)]
    z = z.loc[z.model_pick_side.ne("NO_BET")].copy()
    z["chosen_odds"] = np.where(z.model_pick_side.eq("OVER"), num(z.over_odds), num(z.under_odds))
    z["actual_side"] = [outcome_side(a, l) for a, l in zip(z.actual, z.line)]
    z["bet_result"] = np.select(
        [z.actual_side.eq("PUSH"), z.model_pick_side.eq(z.actual_side)],
        ["PUSH", "WIN"], default="LOSS",
    )
    z["unit_result"] = np.where(
        z.bet_result.eq("WIN"),
        [american_profit(o) for o in z.chosen_odds],
        np.where(z.bet_result.eq("LOSS"), -1.0, 0.0),
    )
    z["model_error"] = z.mc_proj - z.actual
    z["vegas_error"] = z.line - z.actual
    z["model_closer_than_vegas"] = z.model_error.abs() < z.vegas_error.abs()
    z["abs_edge"] = (z.mc_proj - z.line).abs()
    z["edge_bucket"] = z.abs_edge.map(edge_bucket)

    summaries = []
    for market, g in z.groupby("market"):
        decided = g.loc[g.bet_result.isin(["WIN", "LOSS"]) & num(g.chosen_odds).notna()]
        summaries.append({
            "market": market,
            "matched_rows": int(len(g)),
            "decided_bets": int(len(decided)),
            "wins": int(decided.bet_result.eq("WIN").sum()),
            "losses": int(decided.bet_result.eq("LOSS").sum()),
            "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
            "units": float(decided.unit_result.sum()) if len(decided) else np.nan,
            "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
            "model_mae": float(g.model_error.abs().mean()),
            "vegas_mae": float(g.vegas_error.abs().mean()),
            "model_closer_than_vegas_rate": float(g.model_closer_than_vegas.mean()),
        })
    for season, g in z.groupby("season"):
        decided = g.loc[g.bet_result.isin(["WIN", "LOSS"]) & num(g.chosen_odds).notna()]
        summaries.append({
            "market": f"ALL_MARKETS_{int(season)}",
            "matched_rows": int(len(g)),
            "decided_bets": int(len(decided)),
            "wins": int(decided.bet_result.eq("WIN").sum()),
            "losses": int(decided.bet_result.eq("LOSS").sum()),
            "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
            "units": float(decided.unit_result.sum()) if len(decided) else np.nan,
            "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
            "model_mae": float(g.model_error.abs().mean()),
            "vegas_mae": float(g.vegas_error.abs().mean()),
            "model_closer_than_vegas_rate": float(g.model_closer_than_vegas.mean()),
        })
    return z, pd.DataFrame(summaries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p)) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)

    detail, summary = grade(proj, props)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "historical_market_vegas_benchmark_detail.csv", index=False)
    summary.to_csv(a.out_dir / "historical_market_vegas_benchmark_summary.csv", index=False)

    print("=== HISTORICAL MARKET VEGAS BENCHMARK SUMMARY ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
