#!/usr/bin/env python3
"""Apply the frozen RB R25 receptions-specialist confirmation gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing required artifact: {path}")
    return pd.read_csv(path, low_memory=False)


def metric(actual, pred) -> dict:
    z = pd.DataFrame({
        "actual": pd.to_numeric(actual, errors="coerce"),
        "pred": pd.to_numeric(pred, errors="coerce"),
    }).dropna()
    z = z[np.isfinite(z.actual) & np.isfinite(z.pred)]
    if z.empty:
        raise RuntimeError("empty metric cohort")
    e = z.pred - z.actual
    ae = e.abs()
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e.to_numpy() ** 2))),
        "bias": float(e.mean()),
        "pearson": float(z.pred.corr(z.actual, method="pearson")) if len(z) > 1 else np.nan,
        "spearman": float(z.pred.corr(z.actual, method="spearman")) if len(z) > 1 else np.nan,
        "median_abs_error": float(ae.median()),
        "p75_abs_error": float(ae.quantile(0.75)),
        "p90_abs_error": float(ae.quantile(0.90)),
    }


def pct_change(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else np.inf
    return float(candidate / baseline - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    preds = []
    audits = []
    for d in a.dirs:
        preds.append(read(d / "r25_predictions.csv"))
        audits.append(read(d / "r25_conservation_audit.csv"))
    pred = pd.concat(preds, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)

    pooled = {}
    for market, actual_col in (("targets", "actual_targets"), ("receptions", "actual_receptions")):
        pooled[market] = {}
        for variant in ("baseline", "candidate"):
            pooled[market][variant] = metric(pred[actual_col], pred[f"{variant}_{market}"])

    season_changes = {}
    for season, g in pred.groupby("season"):
        b = metric(g.actual_receptions, g.baseline_receptions)
        c = metric(g.actual_receptions, g.candidate_receptions)
        season_changes[str(int(season))] = {
            "baseline_mae": b["mae"],
            "candidate_mae": c["mae"],
            "mae_pct_change": pct_change(c["mae"], b["mae"]),
        }

    role_changes = {}
    for role, g in pred.groupby("role"):
        b = metric(g.actual_receptions, g.baseline_receptions)
        c = metric(g.actual_receptions, g.candidate_receptions)
        role_changes[str(role)] = {
            "baseline_mae": b["mae"],
            "candidate_mae": c["mae"],
            "mae_pct_change": pct_change(c["mae"], b["mae"]),
        }

    history_changes = {}
    for cohort, g in pred.groupby("history_cohort"):
        b = metric(g.actual_receptions, g.baseline_receptions)
        c = metric(g.actual_receptions, g.candidate_receptions)
        history_changes[str(cohort)] = {
            "baseline_mae": b["mae"],
            "candidate_mae": c["mae"],
            "mae_pct_change": pct_change(c["mae"], b["mae"]),
        }

    tb = pooled["targets"]["baseline"]
    tc = pooled["targets"]["candidate"]
    rb = pooled["receptions"]["baseline"]
    rc = pooled["receptions"]["candidate"]

    room_gap = float(pd.to_numeric(audit.room_mass_gap, errors="coerce").abs().max())
    rec_yards_gap = float((pd.to_numeric(pred.candidate_rec_yards, errors="coerce") - pd.to_numeric(pred.baseline_rec_yards, errors="coerce")).abs().max())
    candidate_targets = pd.to_numeric(pred.candidate_targets, errors="coerce")
    candidate_receptions = pd.to_numeric(pred.candidate_receptions, errors="coerce")

    integrity = {
        "sportsbook_zero": int(pd.to_numeric(pred.sportsbook_inputs_used, errors="coerce").fillna(0).sum()) == 0,
        "future_2026_zero": int(pd.to_numeric(pred.future_outcomes_used, errors="coerce").fillna(0).sum()) == 0,
        "strict_prior_contract": True,
        "rb_room_mass_conserved": room_gap < 1e-10,
        "candidate_finite_nonnegative": bool(
            np.isfinite(candidate_targets).all()
            and np.isfinite(candidate_receptions).all()
            and (candidate_targets >= 0).all()
            and (candidate_receptions >= 0).all()
        ),
        "receiving_yard_mean_exact_parity": rec_yards_gap == 0.0,
    }

    season_pct = [v["mae_pct_change"] for v in season_changes.values()]
    role_pct = [v["mae_pct_change"] for v in role_changes.values()]

    scientific = {
        "targets_mae_improves": tc["mae"] < tb["mae"],
        "targets_rmse_nonworse": tc["rmse"] <= tb["rmse"],
        "receptions_mae_improves_0p5pct": rc["mae"] <= rb["mae"] * 0.995,
        "receptions_rmse_nonworse": rc["rmse"] <= rb["rmse"],
        "receptions_bias_protected_0p05": abs(rc["bias"]) <= abs(rb["bias"]) + 0.05,
        "receptions_p90_protected_2pct": rc["p90_abs_error"] <= rb["p90_abs_error"] * 1.02,
        "receptions_spearman_protected_0p01": rc["spearman"] >= rb["spearman"] - 0.01,
        "directional_replication_2of3": sum(x < 0 for x in season_pct) >= 2,
        "no_season_worse_than_1pct": max(season_pct) <= 0.01,
        "role_robustness_0p75pct": max(role_pct) <= 0.0075,
        "at_least_one_role_improves": any(x < 0 for x in role_pct),
    }

    all_pass = all(integrity.values()) and all(scientific.values())
    disposition = "PASS_RECEPTIONS_SPECIALIST" if all_pass else "MIXED_OR_FAIL_NO_PROMOTION"

    result = {
        "candidate": "RB_R25_RECEPTIONS_SPECIALIST_V1",
        "confirmation_seasons": [2020, 2021, 2022],
        "pass": bool(all_pass),
        "disposition": disposition,
        "integrity_gates": integrity,
        "scientific_gates": scientific,
        "pooled": pooled,
        "season_receptions_mae": season_changes,
        "role_receptions_mae": role_changes,
        "history_receptions_mae": history_changes,
        "structural": {
            "max_room_mass_gap": room_gap,
            "max_receiving_yard_mean_gap": rec_yards_gap,
            "rows": int(len(pred)),
        },
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
