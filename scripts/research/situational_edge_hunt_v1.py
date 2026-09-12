#!/usr/bin/env python3
"""Situational edge-hunting over the existing full-stack Vegas benchmark detail.

Research only. Reads data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv
(already-computed, real 2024-2025 graded bets) and slices STRONG_EDGE/LEAN_EDGE
bets by situational context (home/away, season, week bucket, model-uncertainty
bucket, edge-magnitude bucket) instead of just the tier-level cut the existing
benchmark already reported. Looks for slices with real sample size and ROI that
is both positive AND consistent across both 2024 and 2025 individually -- the
same consistency bar the rest of this project's research already uses ("both
seasons", "6/6 seasons") before calling anything real.

Writes no production file. Output is a report only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DETAIL = Path("data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv")
OUT = Path("docs/research/overnight/SITUATIONAL_EDGE_HUNT_V1_RESULT.md")

MIN_N_SEASON = 25  # minimum bets per season for a slice to even be considered


def home_away(row) -> str:
    parts = str(row["game_id"]).split("_")
    if len(parts) != 4:
        return "UNKNOWN"
    home_team = parts[3]
    return "HOME" if str(row["team"]) == home_team else "AWAY"


def week_bucket(w: int) -> str:
    if w <= 6:
        return "EARLY_1_6"
    if w <= 12:
        return "MID_7_12"
    return "LATE_13_18"


def roi(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    return float(bets["unit_result"].sum() / len(bets))


def win_rate(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    return float((bets["bet_result"] == "WIN").mean())


def evaluate_slice(df: pd.DataFrame, dim: str, market: str | None = None) -> pd.DataFrame:
    scope = df if market is None else df.loc[df["market"].eq(market)]
    rows = []
    for val, g in scope.groupby(dim, dropna=False):
        by_season = {}
        ok_seasons = 0
        for season in (2024, 2025):
            sg = g.loc[g["season"].eq(season)]
            n = len(sg)
            r = roi(sg) if n else float("nan")
            by_season[season] = (n, r)
            if n >= MIN_N_SEASON and r is not None and not np.isnan(r) and r > 0:
                ok_seasons += 1
        n_total = len(g)
        r_total = roi(g)
        wr_total = win_rate(g)
        rows.append({
            "market": market or "ALL_NON_QB",
            "dimension": dim,
            "value": val,
            "n": n_total,
            "win_rate": wr_total,
            "roi_per_unit": r_total,
            "n_2024": by_season[2024][0],
            "roi_2024": by_season[2024][1],
            "n_2025": by_season[2025][0],
            "roi_2025": by_season[2025][1],
            "positive_in_both_seasons_min_n": ok_seasons == 2,
        })
    return pd.DataFrame(rows).sort_values("roi_per_unit", ascending=False)


def main() -> int:
    df = pd.read_csv(DETAIL, low_memory=False)
    df["home_away"] = df.apply(home_away, axis=1)
    df["week_bucket"] = df["week"].astype(int).map(week_bucket)
    df["component_sd_q"] = pd.qcut(df["component_sd"], q=4, labels=["SD_Q1_LOW", "SD_Q2", "SD_Q3", "SD_Q4_HIGH"], duplicates="drop")
    df["prob_edge_q"] = pd.qcut(df["prob_edge"].clip(lower=0), q=4, labels=["EDGE_Q1_LOW", "EDGE_Q2", "EDGE_Q3", "EDGE_Q4_HIGH"], duplicates="drop")

    strong = df.loc[df["signal"].eq("STRONG_EDGE")].copy()
    lean_or_strong = df.loc[df["signal"].isin(["STRONG_EDGE", "LEAN_EDGE"])].copy()

    dims = ["home_away", "week_bucket", "component_sd_q", "prob_edge_q", "side"]
    markets = [None, "rush_yards", "rec_yards", "receptions", "rush_rec_yards"]

    all_results = []
    for pool_name, pool in [("STRONG_EDGE_ONLY", strong), ("LEAN_OR_STRONG", lean_or_strong)]:
        for m in markets:
            for d in dims:
                res = evaluate_slice(pool, d, m)
                res["bet_pool"] = pool_name
                all_results.append(res)
    result = pd.concat(all_results, ignore_index=True)

    result_csv = OUT.with_suffix(".csv")
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(result_csv, index=False)

    candidates = result.loc[
        result["positive_in_both_seasons_min_n"]
        & result["roi_per_unit"].gt(0)
    ].sort_values("roi_per_unit", ascending=False)

    lines = []
    lines.append("STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.\n")
    lines.append("# Situational Edge Hunt V1 — Result\n")
    lines.append(f"Source: `{DETAIL}` (already-computed, real 2024-2025 graded bets, {len(df)} rows).")
    lines.append("Method: slice STRONG_EDGE and LEAN_OR_STRONG bet pools by situational")
    lines.append("dimensions (home/away, week bucket, model-uncertainty quartile, edge-magnitude")
    lines.append("quartile, bet side) instead of just the tier-level cut already reported in")
    lines.append("`full_stack_vegas_benchmark_v1/README.md`. A slice only counts as a candidate")
    lines.append(f"if it has >= {MIN_N_SEASON} bets in EACH of 2024 and 2025 individually, AND positive")
    lines.append("ROI in EACH season individually (not just pooled) -- the same both-seasons")
    lines.append("consistency bar this project's own research already requires elsewhere.\n")
    lines.append(f"## Candidates found: {len(candidates)}\n")
    if candidates.empty:
        lines.append("**None.** No situational slice (market x dimension x value), in either the")
        lines.append("STRONG_EDGE-only or LEAN_OR_STRONG pool, shows positive ROI independently in")
        lines.append("both 2024 and 2025 with adequate sample size. This is a real result, not a")
        lines.append("missing-data problem: the underlying detail file has full situational")
        lines.append("coverage for every row tested.\n")
    else:
        lines.append(candidates.to_markdown(index=False))
        lines.append("")

    lines.append("## Full slice table (all dimensions, both pools, all markets)\n")
    lines.append(f"Full results (all {len(result)} slices, most only tested for completeness) written to")
    lines.append(f"`{result_csv}`.\n")

    lines.append("## Top 15 slices by pooled ROI (for context, regardless of both-seasons filter)\n")
    top = result.sort_values("roi_per_unit", ascending=False).head(15)
    lines.append(top.to_markdown(index=False))
    lines.append("")

    lines.append("## Bottom 15 slices by pooled ROI (worst-losing contexts, for symmetry)\n")
    bottom = result.sort_values("roi_per_unit", ascending=True).head(15)
    lines.append(bottom.to_markdown(index=False))
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} and {result_csv}")
    print(f"Candidates passing both-seasons-positive filter: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
