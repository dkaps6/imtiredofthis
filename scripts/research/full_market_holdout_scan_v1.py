#!/usr/bin/env python3
"""Rigorous situational scan across all non-QB markets, with the holdout
discipline built in from the start (unlike situational_edge_hunt_v1.py,
which fit thresholds on pooled data and checked seasons after the fact --
a flaw caught and corrected in RECEPTIONS_UNDER_HOLDOUT_TEST_V1_RESULT.md).

Two kinds of dimensions, handled differently to avoid leakage:
- Categorical (side, home_away, week_bucket): no threshold to overfit, so
  "positive independently in both seasons" is a legitimate bar on its own.
- Quantile-based (prob_edge_q, component_sd_q): the quartile boundary is
  fit on ONE season only, frozen, and applied blind to the OTHER season,
  in both directions. A slice only counts as a candidate if BOTH directions
  are positive with adequate sample size -- this is the real bar, matching
  the corrected receptions-UNDER test.

Uses the same cohort as the receptions work (non_qb_detail_wr_r15_te_r5p_applied.csv)
so rec_yards/receptions already carry the real WR-R15/TE-R5P models where
valid; rush_yards/rush_rec_yards are identical to the base-engine grade
since those markets aren't touched by that adjustment (RB P3/R26/R22 are
Week-1-2026-only and correctly out of scope for a 2024-2025 grade).

Research only. No production change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity, home_away_from_game_id

DETAIL = Path("docs/research/overnight/non_qb_detail_wr_r15_te_r5p_applied.csv")
OUT = Path("docs/research/overnight/FULL_MARKET_HOLDOUT_SCAN_V1_RESULT.md")
MIN_N = 25
MARKETS = ["rush_yards", "rec_yards", "rush_rec_yards", "receptions"]


def home_away(row) -> str:
    return home_away_from_game_id(row.get("team"), row.get("game_id"))


def week_bucket(w: int) -> str:
    if w <= 6:
        return "EARLY_1_6"
    if w <= 12:
        return "MID_7_12"
    return "LATE_13_18"


def roi(x: pd.DataFrame) -> float:
    return float(x["unit_result"].mean()) if len(x) else float("nan")


def wr(x: pd.DataFrame) -> float:
    return float((x["bet_result"] == "WIN").mean()) if len(x) else float("nan")


def categorical_candidates(pool: pd.DataFrame, market: str, dim: str) -> list[dict]:
    out = []
    scope = pool.loc[pool.market.eq(market)]
    for val, g in scope.groupby(dim, dropna=False):
        row = {"market": market, "kind": "categorical", "dimension": dim, "value": str(val)}
        ok = True
        for season in (2024, 2025):
            sg = g.loc[g.season.eq(season)]
            row[f"n_{season}"] = len(sg)
            row[f"roi_{season}"] = roi(sg)
            row[f"wr_{season}"] = wr(sg)
            if len(sg) < MIN_N or not (row[f"roi_{season}"] > 0):
                ok = False
        row["n_pooled"] = len(g)
        row["roi_pooled"] = roi(g)
        row["candidate"] = ok
        out.append(row)
    return out


def quantile_holdout_candidates(pool: pd.DataFrame, market: str, dim_col: str, side_filter: str | None) -> list[dict]:
    scope = pool.loc[pool.market.eq(market)].copy()
    if side_filter is not None:
        scope = scope.loc[scope.side.eq(side_filter)]
    out = []
    for fit_s, test_s in [(2024, 2025), (2025, 2024)]:
        fit = scope.loc[scope.season.eq(fit_s)]
        if fit[dim_col].notna().sum() < 10:
            continue
        cutoff = float(fit[dim_col].clip(lower=0).quantile(0.75))
        test = scope.loc[scope.season.eq(test_s) & scope[dim_col].ge(cutoff)]
        out.append({
            "market": market, "kind": "quantile_holdout", "dimension": dim_col,
            "value": f"top_quartile{'_' + side_filter if side_filter else ''}",
            "fit_season": fit_s, "test_season": test_s, "cutoff": cutoff,
            "n_test": len(test), "roi_test": roi(test), "wr_test": wr(test),
        })
    return out


def main() -> int:
    df = pd.read_csv(DETAIL, low_memory=False)
    assert_benchmark_identity(
        df,
        label="full-market holdout input",
        require_team=True,
        require_opponent=("opponent" in {str(c).strip().lower() for c in df.columns}),
    )
    strong = df.loc[df.signal.eq("STRONG_EDGE")].copy()
    strong["home_away"] = strong.apply(home_away, axis=1)
    if strong["home_away"].eq("UNKNOWN").any():
        sample = strong.loc[strong["home_away"].eq("UNKNOWN"), ["team", "game_id"]].head(20).to_dict(orient="records")
        raise RuntimeError(f"home/away identity unresolved after canonicalization: {sample}")
    strong["week_bucket"] = strong["week"].astype(int).map(week_bucket)

    cat_rows: list[dict] = []
    for market in MARKETS:
        for dim in ("side", "home_away", "week_bucket"):
            cat_rows.extend(categorical_candidates(strong, market, dim))
    cat_df = pd.DataFrame(cat_rows)

    quant_rows: list[dict] = []
    for market in MARKETS:
        for dim_col in ("prob_edge", "component_sd"):
            for side_filter in (None, "OVER", "UNDER"):
                quant_rows.extend(quantile_holdout_candidates(strong, market, dim_col, side_filter))
    quant_df = pd.DataFrame(quant_rows)

    quant_candidates = []
    for (market, dim, value), g in quant_df.groupby(["market", "dimension", "value"]):
        if len(g) == 2 and (g["n_test"] >= MIN_N).all() and (g["roi_test"] > 0).all():
            quant_candidates.append({
                "market": market, "dimension": dim, "value": value,
                "n_fit2024_test2025": int(g.loc[g.fit_season.eq(2024), "n_test"].iloc[0]),
                "roi_fit2024_test2025": float(g.loc[g.fit_season.eq(2024), "roi_test"].iloc[0]),
                "n_fit2025_test2024": int(g.loc[g.fit_season.eq(2025), "n_test"].iloc[0]),
                "roi_fit2025_test2024": float(g.loc[g.fit_season.eq(2025), "roi_test"].iloc[0]),
            })
    quant_candidates_df = pd.DataFrame(quant_candidates)

    cat_out_csv = OUT.with_name("full_market_holdout_scan_categorical.csv")
    quant_out_csv = OUT.with_name("full_market_holdout_scan_quantile.csv")
    cat_df.to_csv(cat_out_csv, index=False)
    quant_df.to_csv(quant_out_csv, index=False)

    cat_candidates = cat_df.loc[cat_df.get("candidate", False) == True] if not cat_df.empty else cat_df

    print(f"Categorical slices tested: {len(cat_df)}; candidates: {len(cat_candidates)}")
    print(f"Quantile-holdout rules tested: {len(quant_df)//2}; candidates (both directions positive): {len(quant_candidates_df)}")
    if not cat_candidates.empty:
        print("\n=== Categorical candidates ===")
        print(cat_candidates.to_string(index=False))
    if not quant_candidates_df.empty:
        print("\n=== Quantile-holdout candidates ===")
        print(quant_candidates_df.to_string(index=False))

    lines = ["STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.\n",
             "# Full-Market Holdout Scan V1\n",
             "Extends the receptions-only situational search to rush_yards, rec_yards,",
             "rush_rec_yards, and receptions, with the holdout discipline built in from the",
             "start this time: categorical dimensions (side/home-away/week-bucket) checked",
             "independently in both seasons; quantile dimensions (prob_edge, component_sd)",
             "fit on one season and tested blind on the other, in both directions, requiring",
             "BOTH directions positive with n>=25 to count as a candidate.\n",
             f"Categorical slices tested: {len(cat_df)}. Candidates: {len(cat_candidates)}.",
             f"Quantile-holdout rules tested: {len(quant_df)//2}. Candidates (both directions positive): {len(quant_candidates_df)}.\n",
             "## Categorical candidates\n"]
    lines.append(cat_candidates.to_markdown(index=False) if not cat_candidates.empty else "None.")
    lines.append("\n## Quantile-holdout candidates (both fit/test directions positive)\n")
    lines.append(quant_candidates_df.to_markdown(index=False) if not quant_candidates_df.empty else "None.")
    lines.append(f"\nFull tables: `{cat_out_csv.name}`, `{quant_out_csv.name}`.\n")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())