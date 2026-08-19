#!/usr/bin/env python3
"""Production pricing from canonical predictive components + joint simulation.

Current production order:
1. independent leakage-safe ML v2 and Markov state v2 projections are attached for audit/future ensemble,
2. leakage-safe empirical-Bayesian player baseline,
3. empirical football/context rules,
4. joint Monte Carlo distribution,
5. sportsbook comparison.

Migration 4C deliberately does NOT blend ML/state into ``model_proj`` yet.
Those independent signals remain parallel until Migration 4D learns ensemble
weights from walk-forward evidence. No model component uses the sportsbook line
to construct a player projection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ml_v2 import apply_ml_to_metrics
from scripts.modeling.state_v2 import apply_state_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.pricing_v2 import _fair_market_prob, _fair_odds
from scripts.runtime_context import resolve_season
from scripts.simulation_v2 import MARKET_MAP, lookup, simulate

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT = OUTPUTS / "props_priced_clean.csv"
RULE_INPUTS = DATA / "model_rule_simulation_inputs.csv"
ML_DIAGNOSTICS = DATA / "model_ml_diagnostics.csv"
STATE_DIAGNOSTICS = DATA / "model_state_diagnostics.csv"


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

    if not ML_DIAGNOSTICS.exists() or ML_DIAGNOSTICS.stat().st_size == 0:
        raise RuntimeError("data/model_ml_diagnostics.csv missing; ML v2 must train before production pricing")
    df = apply_ml_to_metrics(df, pd.read_csv(ML_DIAGNOSTICS))
    ml_rows = int(pd.to_numeric(df.get("ml_applied", 0), errors="coerce").fillna(0).sum())
    if ml_rows == 0:
        raise RuntimeError("ML v2 matched 0 supported pricing rows; refusing silent placeholder behavior")

    if not STATE_DIAGNOSTICS.exists() or STATE_DIAGNOSTICS.stat().st_size == 0:
        raise RuntimeError("data/model_state_diagnostics.csv missing; state v2 must train before production pricing")
    df = apply_state_to_metrics(df, pd.read_csv(STATE_DIAGNOSTICS))
    state_rows = int(pd.to_numeric(df.get("state_applied", 0), errors="coerce").fillna(0).sum())
    if state_rows == 0:
        raise RuntimeError("State v2 matched 0 supported pricing rows; refusing legacy 0.5 fallback behavior")

    df = apply_bayesian_to_metrics(df)
    bayes_rows = int(pd.to_numeric(df.get("bayes_applied", 0), errors="coerce").fillna(0).sum())
    if bayes_rows == 0:
        raise RuntimeError("Bayesian adapter matched 0 metrics rows; refusing baseline-only production pricing")

    df = apply_rules_to_metrics(df)
    rule_rows = int(pd.to_numeric(df.get("rules_applied", 0), errors="coerce").fillna(0).sum())
    if rule_rows == 0:
        raise RuntimeError("Canonical rule adapter matched 0 metrics rows; refusing untracked production pricing")
    RULE_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RULE_INPUTS, index=False)
    print(f"[pricing_mc] ML v2 supported rows={ml_rows}/{len(df)} (parallel signal; not blended yet)")
    print(f"[pricing_mc] state v2 supported rows={state_rows}/{len(df)} (parallel signal; not blended yet)")
    print(f"[pricing_mc] bayesian baseline rows={bayes_rows}/{len(df)}")
    print(f"[pricing_mc] canonical rules applied rows={rule_rows}/{len(df)} -> {RULE_INPUTS}")

    sims = simulate(df)
    rows, missed = [], []
    for _, row in df.iterrows():
        raw_market = str(row.get("market", "") or "").lower()
        market = MARKET_MAP.get(raw_market, raw_market)
        outcomes = lookup(sims, row, raw_market)
        if outcomes is None or len(outcomes) == 0:
            missed.append((row.get("player"), raw_market)); continue
        if market == "anytime_td":
            line = 0.5
            p_over = float(np.mean(outcomes >= 1.0))
        else:
            try:
                line = float(row.get("line"))
            except Exception:
                missed.append((row.get("player"), raw_market)); continue
            p_over = float(np.mean(outcomes > line))
        p_under = 1.0 - p_over
        model_proj = float(np.mean(outcomes))
        model_sd = float(np.std(outcomes, ddof=1)) if len(outcomes) > 1 else 0.0
        mkt_over, mkt_under = _fair_market_prob(row.get("over_odds"), row.get("under_odds"))
        common = {
            "event_id": row.get("event_id"), "player": row.get("player"), "player_clean_key": row.get("player_clean_key"),
            "team": row.get("team"), "opponent": row.get("opponent"), "market": market, "source_market": raw_market,
            "vegas_line": line, "model_proj": model_proj, "model_sd": model_sd, "simulation_iterations": sims.iterations,
            "ml_proj": row.get("ml_proj"), "ml_applied": int(row.get("ml_applied", 0) or 0), "ml_method": row.get("ml_method"), "ml_training_cutoff": row.get("ml_training_cutoff"),
            "state_proj": row.get("state_proj"), "state_applied": int(row.get("state_applied", 0) or 0), "state_method": row.get("state_method"), "state_training_cutoff": row.get("state_training_cutoff"),
            "bayes_applied": int(row.get("bayes_applied", 0) or 0), "bayes_evidence_state": row.get("bayes_evidence_state"),
            "rules_applied": int(row.get("rules_applied", 0) or 0), "rules_role": row.get("rules_role"),
            "season": int(season), "week": row.get("week"), "book": row.get("book"), "book_title": row.get("book_title"),
            "vegas_over_odds": row.get("over_odds"), "vegas_under_odds": row.get("under_odds"),
        }
        for side, prob, market_prob, vegas_odds in (("OVER", p_over, mkt_over, row.get("over_odds")), ("UNDER", p_under, mkt_under, row.get("under_odds"))):
            edge = prob - market_prob if pd.notna(market_prob) else np.nan
            rec = dict(common)
            rec.update({"side": side, "fair_prob": prob, "market_prob": market_prob, "vegas_odds": vegas_odds, "fair_odds": _fair_odds(prob), "edge_pct": edge, "edge_abs": abs(edge) if pd.notna(edge) else np.nan})
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
    parser = argparse.ArgumentParser(); parser.add_argument("--season", type=int, default=None); args = parser.parse_args()
    season = int(args.season if args.season is not None else resolve_season())
    out = price(season); OUTPUTS.mkdir(parents=True, exist_ok=True); out.to_csv(OUT, index=False)
    print(f"[pricing_mc] wrote rows={len(out)} -> {OUT}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
