#!/usr/bin/env python3
"""Grade arbitrary QB projection candidates against the archived Vegas benchmark.

Used by Migration 61 after football projections are frozen. Sportsbook lines are
never model features and never enter candidate selection gates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_qb_vegas_benchmark import (
    american_profit,
    edge_bucket,
    grade_candidate,
    model_side,
    num,
    outcome_side,
    select_props,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--projection-file", action="append", required=True)
    p.add_argument("--props", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-coverage", type=float, default=0.70)
    a = p.parse_args()

    proj = pd.concat([pd.read_csv(Path(x)) for x in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)
    candidates = sorted(c.removeprefix("mc_proj_") for c in proj.columns if c.startswith("mc_proj_"))
    if not candidates:
        raise RuntimeError("no mc_proj_* candidates")

    selected = select_props(props)
    keep = [c for c in ["game_id", "player_clean_key", "book", "line", "over_odds", "under_odds", "player"] if c in selected]
    z = proj.merge(selected[keep], on=["game_id", "player_clean_key"], how="left", validate="many_to_one")
    z["has_prop"] = num(z.line).notna()

    coverage_rows = []
    season_support = True
    for season, g in z.groupby("season"):
        cov = float(g.has_prop.mean())
        season_support &= cov >= float(a.min_coverage)
        coverage_rows.append({"season": int(season), "stable_qb_rows": len(g), "matched_prop_rows": int(g.has_prop.sum()), "coverage": cov})
    combined_cov = float(z.has_prop.mean())
    all_supported = bool(season_support and combined_cov >= float(a.min_coverage))
    coverage_rows.append({"season": "combined", "stable_qb_rows": len(z), "matched_prop_rows": int(z.has_prop.sum()), "coverage": combined_cov})
    coverage = pd.DataFrame(coverage_rows)
    coverage["min_coverage_required"] = float(a.min_coverage)
    coverage["all_required_seasons_supported"] = all_supported
    coverage["claim_status"] = np.where(
        (coverage.coverage >= float(a.min_coverage)) & all_supported,
        "benchmark_claims_supported",
        "exploratory_only_low_coverage",
    )
    coverage["market_source"] = "gcampb41/nfl_data- Action Network-derived archive"
    coverage["source_line_definition"] = "archived_latest_per_book_closing_like_not_exact_30min"
    coverage["exact_30min_snapshot"] = False

    matched = z[z.has_prop].copy()
    metrics = []
    for season, g in matched.groupby("season"):
        for c in candidates:
            metrics.append(grade_candidate(g, c, season_label=str(int(season))))
    for c in candidates:
        metrics.append(grade_candidate(matched, c, season_label="combined"))
    metrics = pd.DataFrame(metrics)

    detail_parts = []
    for c in candidates:
        d = matched.copy()
        d["candidate"] = c
        d["model_proj"] = num(d[f"mc_proj_{c}"])
        d["model_error"] = d.model_proj - num(d.actual)
        d["edge_yards"] = d.model_proj - num(d.line)
        d["abs_edge_yards"] = d.edge_yards.abs()
        d["edge_bucket"] = d.abs_edge_yards.map(edge_bucket)
        d["model_side"] = [model_side(p, l) for p, l in zip(d.model_proj, num(d.line))]
        d["actual_side"] = [outcome_side(x, l) for x, l in zip(num(d.actual), num(d.line))]
        d["market_result"] = np.select(
            [d.actual_side.eq("PUSH"), d.model_side.eq("NO_BET"), d.model_side.eq(d.actual_side)],
            ["PUSH", "NO_BET", "WIN"], default="LOSS"
        )
        d["chosen_odds"] = np.where(d.model_side.eq("OVER"), num(d.over_odds), num(d.under_odds))
        d["unit_result"] = np.nan
        win = d.market_result.eq("WIN") & num(d.chosen_odds).notna()
        loss = d.market_result.eq("LOSS") & num(d.chosen_odds).notna()
        d.loc[win, "unit_result"] = [american_profit(o) for o in d.loc[win, "chosen_odds"]]
        d.loc[loss, "unit_result"] = -1.0
        detail_parts.append(d)
    detail = pd.concat(detail_parts, ignore_index=True)

    threshold_rows = []
    for candidate, cg in detail.groupby("candidate"):
        for threshold in (0, 5, 10, 15, 20, 30):
            g = cg[num(cg.abs_edge_yards).ge(threshold) & cg.market_result.isin(["WIN", "LOSS"])].copy()
            priced = g[num(g.chosen_odds).notna() & num(g.unit_result).notna()]
            threshold_rows.append({
                "candidate": candidate, "min_abs_edge_yards": threshold, "bets": len(g),
                "wins": int(g.market_result.eq("WIN").sum()),
                "win_rate": float(g.market_result.eq("WIN").mean()) if len(g) else np.nan,
                "priced_bets": len(priced),
                "units": float(num(priced.unit_result).sum()) if len(priced) else np.nan,
                "roi_per_unit": float(num(priced.unit_result).mean()) if len(priced) else np.nan,
            })

    tail_rows = []
    for candidate, g in detail.groupby("candidate"):
        for tail, mask in {
            "all_100plus_abs_error": num(g.model_error).abs().ge(100),
            "100plus_overprojection": num(g.model_error).ge(100),
            "100plus_underprojection": num(g.model_error).le(-100),
        }.items():
            t = g[mask & g.market_result.isin(["WIN", "LOSS"])].copy()
            tail_rows.append({
                "candidate": candidate, "tail": tail, "bets": len(t),
                "wins": int(t.market_result.eq("WIN").sum()),
                "win_rate": float(t.market_result.eq("WIN").mean()) if len(t) else np.nan,
                "mean_abs_model_error": float(num(t.model_error).abs().mean()) if len(t) else np.nan,
                "mean_abs_edge_yards": float(num(t.abs_edge_yards).mean()) if len(t) else np.nan,
            })

    for df in (metrics,):
        df["market_source"] = "gcampb41/nfl_data- Action Network-derived archive"
        df["source_line_definition"] = "archived_latest_per_book_closing_like_not_exact_30min"
        df["exact_30min_snapshot"] = False

    a.out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(a.out_dir / "m61_market_coverage.csv", index=False)
    metrics.to_csv(a.out_dir / "m61_football_and_market_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(a.out_dir / "m61_market_edge_thresholds.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(a.out_dir / "m61_catastrophic_market_diagnostics.csv", index=False)
    detail.to_csv(a.out_dir / "m61_market_game_detail.csv", index=False)
    print("=== M61 VEGAS COVERAGE ==="); print(coverage.to_string(index=False))
    print("\n=== M61 FOOTBALL + VEGAS ==="); print(metrics.to_string(index=False))
    print("\n=== M61 EDGE THRESHOLDS ==="); print(pd.DataFrame(threshold_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
