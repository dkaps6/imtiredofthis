#!/usr/bin/env python3
"""Betting reliability scorecard V1.

Answers a practical question the situational edge hunt (`situational_edge_hunt_v1.py`)
was not built to answer directly: on a given night, which *kind* of bet the
board surfaces is actually worth taking, versus which kind has no backtested
track record and should be skipped regardless of how confident the raw
number looks.

This is NOT new research -- it reuses the exact same already-computed,
real 2024-2025 graded-bet detail (`data/backtests/full_stack_vegas_benchmark_v1/
non_qb_detail.csv`) and the exact same "both seasons independently positive,
minimum sample size" validation bar the rest of this project's research
already uses (see `situational_edge_hunt_v1.py`, which found zero of 146
finer situational slices clear that bar). This script deliberately stays at
a coarser grain -- market x signal tier x edge-magnitude quartile x side --
specifically to avoid repeating that multiple-comparisons trap: fewer,
larger, more defensible cuts instead of hunting for one that happens to
clear the bar by chance.

Research/decision-support tool only. Writes no production file, changes no
model behavior, does not touch the live betting board.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DETAIL = Path("data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv")
OUT_DIR = Path("docs/research/overnight")
OUT_CSV = OUT_DIR / "BETTING_RELIABILITY_SCORECARD_V1.csv"
OUT_MD = OUT_DIR / "BETTING_RELIABILITY_SCORECARD_V1.md"

MIN_N_SEASON = 25
SEASONS = (2024, 2025)


def roi(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    return float(bets["unit_result"].sum() / len(bets))


def win_rate(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    return float((bets["bet_result"] == "WIN").mean())


def score_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        by_season = {}
        ok_seasons = 0
        for season in SEASONS:
            sg = g.loc[g["season"].eq(season)]
            n = len(sg)
            r = roi(sg)
            by_season[season] = (n, r)
            if n >= MIN_N_SEASON and not np.isnan(r) and r > 0:
                ok_seasons += 1
        row = dict(zip(group_cols, key))
        row.update({
            "n": len(g),
            "win_rate": win_rate(g),
            "roi_per_unit": roi(g),
            "n_2024": by_season[2024][0],
            "roi_2024": by_season[2024][1],
            "n_2025": by_season[2025][0],
            "roi_2025": by_season[2025][1],
            "validated_both_seasons_positive": ok_seasons == len(SEASONS),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roi_per_unit", ascending=False)


def main() -> int:
    df = pd.read_csv(DETAIL, low_memory=False)
    df["prob_edge_q"] = pd.qcut(
        df["prob_edge"].clip(lower=0), q=4,
        labels=["EDGE_Q1_LOW", "EDGE_Q2", "EDGE_Q3", "EDGE_Q4_HIGH"], duplicates="drop",
    )

    cuts = {
        "market_only": ["market"],
        "market_x_signal": ["market", "signal"],
        "market_x_edge_quartile": ["market", "prob_edge_q"],
        "market_x_side": ["market", "side"],
        "market_x_signal_x_side": ["market", "signal", "side"],
        "signal_only": ["signal"],
        "side_only": ["side"],
        "edge_quartile_only": ["prob_edge_q"],
    }

    all_results = []
    for cut_name, cols in cuts.items():
        res = score_group(df, cols)
        res.insert(0, "cut", cut_name)
        all_results.append(res)
    result = pd.concat(all_results, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    validated = result.loc[result["validated_both_seasons_positive"]].sort_values("roi_per_unit", ascending=False)

    lines = []
    lines.append("# Betting Reliability Scorecard V1\n")
    lines.append(
        f"Source: `{DETAIL}` ({len(df)} real graded bets, 2024-2025, non-QB markets only; "
        "QB pass_yards has no comparable row-level detail file -- see "
        "`data/backtests/full_stack_vegas_benchmark_v1/qb_*_summary.csv` separately)."
    )
    lines.append(
        f"Bar for VALIDATED: positive ROI independently in each of {SEASONS[0]} and "
        f"{SEASONS[1]}, with >= {MIN_N_SEASON} bets in each season. Same bar "
        "`situational_edge_hunt_v1.py` already uses; that script tested 146 finer "
        "situational slices and found zero pass. This scorecard stays at a "
        "coarser grain (market x signal tier x edge quartile x side) so a pass "
        "here is less likely to be a multiple-comparisons artifact.\n"
    )

    lines.append(f"## Validated cuts: {len(validated)}\n")
    if validated.empty:
        lines.append(
            "**None.** No market, signal tier, edge-magnitude quartile, side, or "
            "combination of those, shows positive ROI independently in both 2024 "
            "and 2025 with adequate sample size. Read literally: as of this data, "
            "there is no proven basis for treating any category of board output as "
            "reliably profitable. A `HAS EDGE` tag on tonight's board reflects the "
            "model's own probability estimate, not a backtested track record.\n"
        )
    else:
        lines.append(validated.to_markdown(index=False))
        lines.append("")

    lines.append("## Market-level summary (all bets, for context)\n")
    market_only = result.loc[result["cut"].eq("market_only")].drop(columns=["cut"])
    lines.append(market_only.to_markdown(index=False))
    lines.append("")

    lines.append("## Market x signal tier (STRONG_EDGE / LEAN_EDGE / NO_EDGE)\n")
    market_signal = result.loc[result["cut"].eq("market_x_signal")].drop(columns=["cut"])
    lines.append(market_signal.to_markdown(index=False))
    lines.append("")

    lines.append("## Market x edge-magnitude quartile\n")
    market_edge = result.loc[result["cut"].eq("market_x_edge_quartile")].drop(columns=["cut"])
    lines.append(market_edge.to_markdown(index=False))
    lines.append("")

    lines.append("## Market x side (OVER / UNDER)\n")
    market_side = result.loc[result["cut"].eq("market_x_side")].drop(columns=["cut"])
    lines.append(market_side.to_markdown(index=False))
    lines.append("")

    side_only = result.loc[result["cut"].eq("side_only")].drop(columns=["cut"])
    over_row = side_only.loc[side_only["side"].eq("OVER")]
    under_row = side_only.loc[side_only["side"].eq("UNDER")]
    if not over_row.empty and not under_row.empty:
        lines.append("## Side skew\n")
        lines.append(
            f"The model's chosen side across all {len(df)} bets is UNDER "
            f"{int(under_row['n'].iloc[0])} times vs OVER {int(over_row['n'].iloc[0])} times "
            f"({under_row['n'].iloc[0] / len(df):.1%} UNDER). Combined with pooled ROI of "
            f"{under_row['roi_per_unit'].iloc[0]:+.2%} on UNDER vs {over_row['roi_per_unit'].iloc[0]:+.2%} "
            "on OVER, this is consistent with a systematic overprediction bias in the "
            "underlying projections, not just noise -- worth investigating as a "
            "calibration issue separately from this scorecard's per-slice question.\n"
        )

    lines.append("## Full cut table\n")
    lines.append(f"All {len(result)} rows across every cut tested written to `{OUT_CSV}`.\n")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD} and {OUT_CSV}")
    print(f"Validated cuts: {len(validated)} / {len(result)} tested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
