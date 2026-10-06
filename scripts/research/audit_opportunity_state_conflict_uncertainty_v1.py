#!/usr/bin/env python3
"""Opportunity State Conflict Uncertainty V1.

Frozen in docs/research/OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_V1_PLAN.md.

Tests one football-only uncertainty signal:
    abs(PlayerForm opportunity state - Bayesian opportunity state)

against the absolute error of the existing Bayesian opportunity authority.
No mean replacement, parameter fitting, sportsbook data, 2026 outcome data,
threshold search or production mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.audit_bayesian_current_state_transmission_v1 import (
    PRIMARY,
    _load_normalized_logs,
    _load_rosters,
    build_panel,
)

VERSION = "OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_V1"
REPS = 10_000
SEED = 20261006
MIN_ROWS = 100
MIN_PLAYERS = 20


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    q = pd.DataFrame({"a": _finite(a), "b": _finite(b)}).dropna()
    if len(q) < 3 or q["a"].nunique() < 2 or q["b"].nunique() < 2:
        return float("nan")
    return float(q["a"].corr(q["b"], method="spearman"))


def _support(q: pd.DataFrame) -> tuple[bool, str]:
    if len(q) < MIN_ROWS:
        return False, f"rows<{MIN_ROWS}"
    if q["player_identity_key"].nunique() < MIN_PLAYERS:
        return False, f"players<{MIN_PLAYERS}"
    if q["state_conflict"].nunique() < 2:
        return False, "state_conflict_has_no_variance"
    return True, "PASS"


def _quartile_masks(q: pd.DataFrame) -> tuple[pd.Series, pd.Series, float, float]:
    x = _finite(q["state_conflict"])
    q25 = float(x.quantile(0.25))
    q75 = float(x.quantile(0.75))
    low = x.le(q25)
    high = x.ge(q75)
    return low, high, q25, q75


def _cluster_bootstrap(q: pd.DataFrame, q25: float, q75: float) -> dict:
    players = sorted(q["player_identity_key"].astype(str).unique().tolist())
    if len(players) < 2:
        return {"reps": 0, "valid_reps": 0, "ci_low": None, "ci_high": None}

    # Preserve each player's complete row cluster, but summarize its fixed Q1/Q4
    # contributions once so 10k bootstrap replicates do not repeatedly concat
    # thousands of rows. This is mathematically identical to resampling player
    # clusters with replacement under the frozen original Q1/Q4 thresholds.
    stats = []
    for p in players:
        z = q.loc[q["player_identity_key"].astype(str).eq(p)]
        low = z["state_conflict"].le(q25)
        high = z["state_conflict"].ge(q75)
        stats.append([
            float(z.loc[high, "bayes_abs_error"].sum()),
            float(high.sum()),
            float(z.loc[low, "bayes_abs_error"].sum()),
            float(low.sum()),
        ])
    a = np.asarray(stats, dtype=float)

    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(
        len(players),
        [1.0 / len(players)] * len(players),
        size=REPS,
    ).astype(float)

    high_sum = counts @ a[:, 0]
    high_n = counts @ a[:, 1]
    low_sum = counts @ a[:, 2]
    low_n = counts @ a[:, 3]
    valid = (high_n > 0) & (low_n > 0)
    if not valid.any():
        return {"reps": REPS, "valid_reps": 0, "ci_low": None, "ci_high": None}

    arr = (high_sum[valid] / high_n[valid]) - (low_sum[valid] / low_n[valid])
    return {
        "reps": REPS,
        "valid_reps": int(valid.sum()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_effect_gt_0": float((arr > 0).mean()),
    }


def score_cell(q: pd.DataFrame, season: int, position: str, metric: str) -> dict:
    z = q.copy()
    z["playerform_value"] = _finite(z["playerform_value"])
    z["bayes_value"] = _finite(z["bayes_value"])
    z["actual_value"] = _finite(z["actual_value"])
    z = z.dropna(subset=["playerform_value", "bayes_value", "actual_value"])
    z["state_conflict"] = (z["playerform_value"] - z["bayes_value"]).abs()
    z["bayes_abs_error"] = (z["bayes_value"] - z["actual_value"]).abs()

    ok, support_reason = _support(z)
    rec = {
        "season": int(season),
        "position": position,
        "metric": metric,
        "rows": int(len(z)),
        "unique_players": int(z["player_identity_key"].nunique()),
        "support": support_reason,
        "state_conflict_mean": float(z["state_conflict"].mean()) if len(z) else None,
        "state_conflict_median": float(z["state_conflict"].median()) if len(z) else None,
        "bayes_abs_error_mean": float(z["bayes_abs_error"].mean()) if len(z) else None,
    }
    if not ok:
        rec.update({
            "spearman_conflict_vs_bayes_ae": None,
            "q25_conflict": None,
            "q75_conflict": None,
            "q1_rows": 0,
            "q4_rows": 0,
            "q1_bayes_ae": None,
            "q4_bayes_ae": None,
            "q4_minus_q1_bayes_ae": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "cell_pass": False,
        })
        return rec

    low, high, q25, q75 = _quartile_masks(z)
    if not low.any() or not high.any():
        rec["support"] = "quartile_support_empty"
        rec["cell_pass"] = False
        return rec

    rho = _spearman(z["state_conflict"], z["bayes_abs_error"])
    q1 = float(z.loc[low, "bayes_abs_error"].mean())
    q4 = float(z.loc[high, "bayes_abs_error"].mean())
    delta = q4 - q1
    boot = _cluster_bootstrap(z, q25, q75)

    ci_low = boot.get("ci_low")
    passed = bool(
        np.isfinite(rho)
        and rho > 0
        and delta > 0
        and ci_low is not None
        and np.isfinite(ci_low)
        and ci_low > 0
    )
    rec.update({
        "spearman_conflict_vs_bayes_ae": float(rho),
        "q25_conflict": q25,
        "q75_conflict": q75,
        "q1_rows": int(low.sum()),
        "q4_rows": int(high.sum()),
        "q1_bayes_ae": q1,
        "q4_bayes_ae": q4,
        "q4_minus_q1_bayes_ae": float(delta),
        "bootstrap_reps": boot.get("reps"),
        "bootstrap_valid_reps": boot.get("valid_reps"),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": boot.get("ci_high"),
        "bootstrap_p_effect_gt_0": boot.get("p_effect_gt_0"),
        "cell_pass": passed,
    })
    return rec


def classify(cells: pd.DataFrame) -> dict:
    per_metric = []
    for position, metric, _actual_col in PRIMARY:
        q = cells.loc[cells["position"].eq(position) & cells["metric"].eq(metric)].copy()
        by = {int(r.season): bool(r.cell_pass) for r in q.itertuples()}
        replicated = by.get(2024, False) and by.get(2025, False)
        per_metric.append({
            "position": position,
            "metric": metric,
            "season_2024_pass": bool(by.get(2024, False)),
            "season_2025_pass": bool(by.get(2025, False)),
            "disposition": (
                "STATE_CONFLICT_UNCERTAINTY_REPLICATED"
                if replicated
                else "STATE_CONFLICT_UNCERTAINTY_NOT_REPLICATED"
            ),
        })
    any_rep = any(x["disposition"] == "STATE_CONFLICT_UNCERTAINTY_REPLICATED" for x in per_metric)
    return {
        "disposition": (
            "OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_SIGNAL_CONFIRMED"
            if any_rep
            else "OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_NULL"
        ),
        "per_metric": per_metric,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/research/opportunity_state_conflict_uncertainty_v1"),
    )
    args = ap.parse_args()

    logs = _load_normalized_logs({2023, 2024, 2025})
    rosters = {season: _load_rosters(season) for season in (2024, 2025)}
    panel, universe_audit = build_panel(logs, rosters)

    rows = []
    for season in (2024, 2025):
        for position, metric, _actual_col in PRIMARY:
            q = panel.loc[
                panel["season"].eq(season)
                & panel["position"].eq(position)
                & panel["metric"].eq(metric)
            ].copy()
            rows.append(score_cell(q, season, position, metric))

    cells = pd.DataFrame(rows)
    result = classify(cells)
    result.update({
        "version": VERSION,
        "plan": "docs/research/OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_V1_PLAN.md",
        "sportsbook_inputs_used": 0,
        "outcomes_2026_used": 0,
        "parameters_fit": 0,
        "candidate_mean_changes": 0,
        "bootstrap_reps": REPS,
        "bootstrap_seed": SEED,
        "source_panel_rows": int(len(panel)),
        "universe_audit_rows": int(len(universe_audit)),
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "opportunity_state_conflict_uncertainty_cells.csv", index=False)
    (args.out_dir / "opportunity_state_conflict_uncertainty_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(cells.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
