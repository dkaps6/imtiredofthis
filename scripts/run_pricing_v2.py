#!/usr/bin/env python3
"""Production pricing: empirical probabilities from joint Monte Carlo outcomes.

Migration 3 applies the canonical empirical football rules to simulation inputs
before the joint Monte Carlo runs.  Final projections are still the empirical
mean of simulated outcomes; there is no post-hoc projection multiplier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.pricing_v2 import _fair_market_prob, _fair_odds
from scripts.runtime_context import resolve_season
from scripts.simulation_v2 import MARKET_MAP, lookup, simulate

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "props_priced_clean.csv"
RULE_INPUTS = DATA / "model_rule_simulation_inputs.csv"


def price(season: int) -> pd.DataFrame:
    metrics_path = DATA / "metrics_ready.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        raise RuntimeError("data/metrics_ready.csv missing or empty")
    df = pd.read_csv(metrics_path)
    df.columns = [str(c).lower() for c in df.columns]
    if "season" in df.columns:
        df = df.loc[pd.to_numeric(df["season"], errors="coerce").eq(int(season))].copy()
    if df.empty:
        raise RuntimeError(f"metrics_ready contains no rows for season={season}")

    # Rules alter finite opportunity / efficiency / uncertainty assumptions
    # before simulation. Keep an audit artifact so every priced projection can be
    # traced back to the exact rule-adjusted inputs used by Monte Carlo.
    df = apply_rules_to_metrics(df)
    rule_rows = int(pd.to_numeric(df.get("rules_applied", 0), errors="coerce").fillna(0).sum())
    if rule_rows == 0:
        raise RuntimeError("Canonical rule adapter matched 0 metrics rows; refusing untracked production pricing")
    RULE_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RULE_INPUTS, index=False)
    print(f"[pricing_mc] canonical rules applied rows={rule_rows}/{len(df)} -> {RULE_INPUTS}")

    sims = simulate(df)
    rows = []
    missed = []
    for idx, row in df.iterrows():
        raw_market = str(row.get("market", "") or "").lower()
        market = MARKET_MAP.get(raw_market, raw_market)
        outcomes = lookup(sims, row, raw_market)
        if outcomes is None or len(outcomes) == 0:
            missed.append((row.get("player"), raw_market))
            continue

        if market == "anytime_td":
            line = 0.5
            p_over = float(np.mean(outcomes >= 1.0))
        else:
            try:
                line = float(row.get("line"))
            except Exception:
                missed.append((row.get("player"), raw_market))
                continue
            p_over = float(np.mean(outcomes > line))
        p_under = 1.0 - p_over
        model_proj = float(np.mean(outcomes))
        model_sd = float(np.std(outcomes, ddof=1)) if len(outcomes) > 1 else 0.0
        mkt_over, mkt_under = _fair_market_prob(row.get("over_odds"), row.get("under_odds"))

        common = {
            "event_id": row.get("event_id"),
            "player": row.get("player"),
            "player_clean_key": row.get("player_clean_key"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": market,
            "source_market": raw_market,
            "vegas_line": line,
            "model_proj": model_proj,
            "model_sd": model_sd,
            "simulation_iterations": sims.iterations,
            "rules_applied": int(row.get("rules_applied", 0) or 0),
            "rules_role": row.get("rules_role"),
            "season": int(season),
            "week": row.get("week"),
            "book": row.get("book"),
            "book_title": row.get("book_title"),
            "vegas_over_odds": row.get("over_odds"),
            "vegas_under_odds": row.get("under_odds"),
        }
        for side, prob, market_prob, vegas_odds in (
            ("OVER", p_over, mkt_over, row.get("over_odds")),
            ("UNDER", p_under, mkt_under, row.get("under_odds")),
        ):
            edge = prob - market_prob if pd.notna(market_prob) else np.nan
            rec = dict(common)
            rec.update({
                "side": side,
                "fair_prob": prob,
                "market_prob": market_prob,
                "vegas_odds": vegas_odds,
                "fair_odds": _fair_odds(prob),
                "edge_pct": edge,
                "edge_abs": abs(edge) if pd.notna(edge) else np.nan,
            })
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Monte Carlo pricing produced 0 rows")
    if missed:
        debug = DATA / "_debug" / "pricing_unsimulated_props.csv"
        debug.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missed, columns=["player", "market"]).drop_duplicates().to_csv(debug, index=False)
        print(f"[pricing_mc] WARN unsimulated player/markets={len(set(missed))} -> {debug}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    out = price(season)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[pricing_mc] wrote rows={len(out)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
