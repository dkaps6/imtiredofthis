#!/usr/bin/env python3
"""Diagnose STRONG gate mechanics on an empirical-MC fair-probability cohort.

Research/diagnosis only. This script does not fit thresholds, alter model
probabilities, or use outcomes to propose a production decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRONG_EV_GATE = 0.05
STRONG_PROB_EDGE_GATE = 0.03


def num(x):
    return pd.to_numeric(x, errors="coerce")


def implied_prob(odds: pd.Series) -> pd.Series:
    o = num(odds)
    out = pd.Series(np.nan, index=o.index, dtype=float)
    pos = o > 0
    neg = o < 0
    out.loc[pos] = 100.0 / (o.loc[pos] + 100.0)
    out.loc[neg] = (-o.loc[neg]) / ((-o.loc[neg]) + 100.0)
    return out


def auc_rank(y: pd.Series, score: pd.Series) -> float:
    z = pd.DataFrame({"y": num(y), "score": num(score)}).dropna()
    if z.empty or z.y.nunique() < 2:
        return np.nan
    n1 = int((z.y == 1).sum())
    n0 = int((z.y == 0).sum())
    ranks = z.score.rank(method="average")
    return float((ranks[z.y.eq(1)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize_scope(g: pd.DataFrame, market: str) -> dict:
    decided = g.loc[g.bet_result.isin(["WIN", "LOSS"])].copy()
    ev_pass = g.best_ev.ge(STRONG_EV_GATE)
    edge_pass = g.prob_edge.ge(STRONG_PROB_EDGE_GATE)
    joint = ev_pass & edge_pass
    strong_decided = decided.loc[
        decided.best_ev.ge(STRONG_EV_GATE)
        & decided.prob_edge.ge(STRONG_PROB_EDGE_GATE)
    ]
    return {
        "market": market,
        "rows": int(len(g)),
        "ev5_rows": int(ev_pass.sum()),
        "ev5_coverage": float(ev_pass.mean()) if len(g) else np.nan,
        "edge3_rows": int(edge_pass.sum()),
        "edge3_coverage": float(edge_pass.mean()) if len(g) else np.nan,
        "joint_rows": int(joint.sum()),
        "joint_coverage": float(joint.mean()) if len(g) else np.nan,
        "ev5_but_not_edge3_rows": int((ev_pass & ~edge_pass).sum()),
        "edge3_but_not_ev5_rows": int((edge_pass & ~ev_pass).sum()),
        "ev_implies_edge_on_observed_rows": bool((~ev_pass | edge_pass).all()),
        "ev_required_p_stricter_share": float(g.ev_required_p.ge(g.edge_required_p - 1e-12).mean()),
        "median_model_p": float(g.best_model_p.median()),
        "median_market_novig_p": float(g.best_market_p.median()),
        "median_raw_implied_p": float(g.chosen_raw_implied.median()),
        "median_ev5_required_p": float(g.ev_required_p.median()),
        "median_edge3_required_p": float(g.edge_required_p.median()),
        "strong_mean_model_p": float(strong_decided.best_model_p.mean()) if len(strong_decided) else np.nan,
        "strong_mean_predicted_ev": float(strong_decided.best_ev.mean()) if len(strong_decided) else np.nan,
        "strong_win_rate": float(strong_decided.bet_result.eq("WIN").mean()) if len(strong_decided) else np.nan,
        "strong_realized_roi": float(num(strong_decided.unit_result).mean()) if len(strong_decided) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(a.detail, low_memory=False)
    required = {
        "market", "best_ev", "prob_edge", "best_model_p", "best_market_p",
        "chosen_odds", "bet_result", "unit_result", "signal",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    for c in ["best_ev", "prob_edge", "best_model_p", "best_market_p", "chosen_odds", "unit_result"]:
        x[c] = num(x[c])
    x["chosen_raw_implied"] = implied_prob(x.chosen_odds)
    # EV = p / raw_implied - 1 for valid American odds.
    x["ev_required_p"] = (1.0 + STRONG_EV_GATE) * x.chosen_raw_implied
    x["edge_required_p"] = x.best_market_p + STRONG_PROB_EDGE_GATE

    scopes = [summarize_scope(g, str(m)) for m, g in x.groupby("market", sort=True)]
    scopes.append(summarize_scope(x, "ALL_MARKETS"))
    pd.DataFrame(scopes).to_csv(
        a.out_dir / "empirical_strong_gate_overlap_summary.csv", index=False
    )

    decided = x.loc[x.bet_result.isin(["WIN", "LOSS"])].copy()
    decided["actual_win"] = decided.bet_result.eq("WIN").astype(int)
    edges = [-1e-9, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.000001]
    decided["model_p_bin"] = pd.cut(
        decided.best_model_p, bins=edges, include_lowest=True
    )
    bins = (
        decided.groupby("model_p_bin", observed=True)
        .agg(
            rows=("best_model_p", "size"),
            mean_model_p=("best_model_p", "mean"),
            mean_market_novig_p=("best_market_p", "mean"),
            realized_win_rate=("actual_win", "mean"),
            mean_predicted_ev=("best_ev", "mean"),
            realized_roi=("unit_result", "mean"),
        )
        .reset_index()
    )
    bins["model_p_bin"] = bins.model_p_bin.astype(str)
    bins.to_csv(a.out_dir / "empirical_model_p_calibration_bins.csv", index=False)

    strong = decided.loc[
        decided.best_ev.ge(STRONG_EV_GATE)
        & decided.prob_edge.ge(STRONG_PROB_EDGE_GATE)
    ].copy()
    diagnostics = {
        "rows": int(len(x)),
        "decided_rows": int(len(decided)),
        "strong_rows": int(len(strong)),
        "strong_coverage": float(len(strong) / len(x)),
        "strong_mean_model_p": float(strong.best_model_p.mean()),
        "strong_median_model_p": float(strong.best_model_p.median()),
        "strong_mean_market_novig_p": float(strong.best_market_p.mean()),
        "strong_mean_raw_implied_p": float(strong.chosen_raw_implied.mean()),
        "strong_mean_predicted_ev": float(strong.best_ev.mean()),
        "strong_median_predicted_ev": float(strong.best_ev.median()),
        "strong_realized_win_rate": float(strong.actual_win.mean()),
        "strong_realized_roi": float(strong.unit_result.mean()),
        "best_model_p_auc_for_realized_win": auc_rank(decided.actual_win, decided.best_model_p),
        "ev_gate_implies_edge_gate_on_all_observed_ev_pass_rows": bool(
            (~x.best_ev.ge(STRONG_EV_GATE) | x.prob_edge.ge(STRONG_PROB_EDGE_GATE)).all()
        ),
        "ev_required_p_stricter_share": float(
            x.ev_required_p.ge(x.edge_required_p - 1e-12).mean()
        ),
    }
    (a.out_dir / "empirical_strong_gate_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
