#!/usr/bin/env python3
"""Grade the full projection stack against Vegas using the real PLAY/LEAN gate.

This reuses the exact decision-signal formulas already live in
scripts/master_betting_workbook_core_v2.py (implied_prob, no_vig, ev_roi, and
the STRONG EDGE / LEAN EDGE / NO EDGE thresholds: EV >= .05 and prob_edge >=
.03 for STRONG; EV > 0 for LEAN) instead of a naive "bet every disagreement"
rule. The point of this script is to test the same selectivity production
would actually apply, not a friendlier version invented for this backtest.

Fidelity note: production's fair probability comes from the full Monte Carlo
outcome distribution (empirical P(actual > line) across simulated draws).
Persisting that full distribution per historical row was out of scope here,
so this uses a Normal(mean=proj, sd=component_sd) approximation, where
component_sd is the spread across mc_proj/ml_proj/state_proj -- a real
disagreement signal, but not the literal simulated distribution production
uses. This is disclosed, not hidden: treat probability-threshold results as
directionally informative, not exact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import select_one_book_row
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side

STRONG_EV_GATE = 0.05
STRONG_PROB_EDGE_GATE = 0.03


def implied_prob(odds):
    o = num(pd.Series([odds])).iloc[0]
    if pd.isna(o) or o == 0:
        return np.nan
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def no_vig(a, b):
    if pd.isna(a):
        return np.nan
    if pd.isna(b):
        return a
    return a / (a + b) if (a + b) else np.nan


def ev_roi(prob, odds):
    if pd.isna(prob) or pd.isna(odds) or odds == 0:
        return np.nan
    payout = odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    return prob * payout - (1 - prob)


def signal(ev: float, prob_edge: float) -> str:
    if pd.notna(ev) and pd.notna(prob_edge) and ev >= STRONG_EV_GATE and prob_edge >= STRONG_PROB_EDGE_GATE:
        return "STRONG_EDGE"
    if pd.notna(ev) and ev > 0:
        return "LEAN_EDGE"
    return "NO_EDGE"


def grade(proj: pd.DataFrame, props: pd.DataFrame, *, proj_col: str = "ensemble_proj") -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_one_book_row(props)
    join_cols = ["game_id", "player_clean_key", "market"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "player"]
    selected = selected[[c for c in keep if c in selected]].copy()

    z = proj.merge(selected, on=join_cols, how="inner")
    if z.empty:
        return z, pd.DataFrame()

    z["proj"] = num(z[proj_col])
    z["actual"] = num(z.actual)
    z["line"] = num(z.line)
    comps = z[[c for c in ["mc_proj", "ml_proj", "state_proj"] if c in z.columns]].apply(num)
    z["component_sd"] = comps.std(axis=1, skipna=True).clip(lower=1e-6)

    z["p_over"] = norm.cdf((z.proj - z.line) / z.component_sd)
    z["p_under"] = 1.0 - z.p_over
    z["over_implied"] = z.over_odds.map(implied_prob)
    z["under_implied"] = z.under_odds.map(implied_prob)
    z["over_novig"] = [no_vig(a, b) for a, b in zip(z.over_implied, z.under_implied)]
    z["under_novig"] = [no_vig(a, b) for a, b in zip(z.under_implied, z.over_implied)]
    z["ev_over"] = [ev_roi(p, o) for p, o in zip(z.p_over, z.over_odds)]
    z["ev_under"] = [ev_roi(p, o) for p, o in zip(z.p_under, z.under_odds)]

    best_over = z.ev_under.isna() | (z.ev_over.fillna(-np.inf) >= z.ev_under.fillna(-np.inf))
    z["side"] = np.where(best_over, "OVER", "UNDER")
    z["best_ev"] = np.where(best_over, z.ev_over, z.ev_under)
    z["best_model_p"] = np.where(best_over, z.p_over, z.p_under)
    z["best_market_p"] = np.where(best_over, z.over_novig, z.under_novig)
    z["prob_edge"] = z.best_model_p - z.best_market_p
    z["chosen_odds"] = np.where(best_over, z.over_odds, z.under_odds)
    z["signal"] = [signal(e, q) for e, q in zip(z.best_ev, z.prob_edge)]

    z["actual_side"] = [outcome_side(a, l) for a, l in zip(z.actual, z.line)]
    z["bet_result"] = np.select(
        [z.actual_side.eq("PUSH"), z.side.eq(z.actual_side)],
        ["PUSH", "WIN"], default="LOSS",
    )
    z["unit_result"] = np.where(
        z.bet_result.eq("WIN"),
        [american_profit(o) for o in z.chosen_odds],
        np.where(z.bet_result.eq("LOSS"), -1.0, 0.0),
    )
    z["model_error"] = z.proj - z.actual
    z["vegas_error"] = z.line - z.actual

    summaries = []
    tiers = {
        "ALL_NO_FILTER": z,
        "LEAN_OR_STRONG": z.loc[z.signal.isin(["LEAN_EDGE", "STRONG_EDGE"])],
        "STRONG_ONLY_PLAY_TIER": z.loc[z.signal.eq("STRONG_EDGE")],
    }
    scope = list(z.market.unique()) + ["ALL_MARKETS"]
    for market in scope:
        for tier_name, tier_df in tiers.items():
            g = tier_df if market == "ALL_MARKETS" else tier_df.loc[tier_df.market.eq(market)]
            decided = g.loc[g.bet_result.isin(["WIN", "LOSS"]) & num(g.chosen_odds).notna()]
            summaries.append({
                "market": market, "tier": tier_name,
                "matched_rows": int(len(g)),
                "decided_bets": int(len(decided)),
                "wins": int(decided.bet_result.eq("WIN").sum()),
                "losses": int(decided.bet_result.eq("LOSS").sum()),
                "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
                "units": float(decided.unit_result.sum()) if len(decided) else np.nan,
                "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
                "model_mae": float(g.model_error.abs().mean()) if len(g) else np.nan,
                "vegas_mae": float(g.vegas_error.abs().mean()) if len(g) else np.nan,
            })
    return z, pd.DataFrame(summaries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--proj-col", default="ensemble_proj")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p)) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)

    detail, summary = grade(proj, props, proj_col=a.proj_col)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "full_stack_vegas_benchmark_detail.csv", index=False)
    summary.to_csv(a.out_dir / "full_stack_vegas_benchmark_summary.csv", index=False)

    print("=== FULL STACK VEGAS BENCHMARK SUMMARY ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
