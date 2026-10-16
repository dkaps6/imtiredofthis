#!/usr/bin/env python3
"""Frozen pass-catcher efficiency-difficulty trust/selectivity V1.

Research-only reliability diagnostic around unchanged final WR/TE receiving-yard
projections. No mean or distribution value is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_V1"
BOOT_REPS = 5000
TOL = 1e-9

CONFIG = {
    "WR": {
        "seasons": (2023, 2024),
        "expected_rows": 3224,
        "expected_players": 211,
        "dev": 2023,
        "confirm": 2024,
        "min_dev_rows": 1200,
        "min_dev_players": 140,
        "min_confirm_rows": 1600,
        "min_confirm_players": 160,
        "min_pool_rows": 3000,
        "min_pool_players": 190,
        "player_seeds": (2026101531, 2026101532),
        "team_seeds": (2026101533, 2026101534),
        "qualify": "WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED",
        "closed": "NO_ACTIONABLE_WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY",
    },
    "TE": {
        "seasons": (2024, 2025),
        "expected_rows": 2005,
        "expected_players": 121,
        "dev": 2024,
        "confirm": 2025,
        "min_dev_rows": 800,
        "min_dev_players": 80,
        "min_confirm_rows": 800,
        "min_confirm_players": 80,
        "min_pool_rows": 1800,
        "min_pool_players": 110,
        "player_seeds": (2026101541, 2026101542),
        "team_seeds": (2026101543, 2026101544),
        "qualify": "TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED",
        "closed": "NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY",
    },
}

REQUIRED = {
    "season",
    "week",
    "team",
    "player_clean_key",
    "prior_games",
    "prior8_abs_opportunity_error_mean",
    "prior8_abs_efficiency_error_mean",
    "current_opportunity_error",
    "current_abs_opportunity_error",
    "current_efficiency_error",
    "current_abs_efficiency_error",
    "current_total_error",
    "current_abs_total_error",
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def rank_pct(s):
    return num(s).rank(method="average", pct=True)


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def partial_corr(x, y, z):
    rxy = corr(x, y)
    rxz = corr(x, z)
    ryz = corr(y, z)
    if not all(np.isfinite([rxy, rxz, ryz])):
        return float("nan")
    den = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)) ** 0.5
    if den <= 1e-12:
        return float("nan")
    return float((rxy - rxz * ryz) / den)


def read_parent(path: Path, position: str):
    cfg = CONFIG[position]
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]

    missing = REQUIRED - set(x.columns)
    if missing:
        raise RuntimeError(f"{position} parent missing {sorted(missing)}")

    for c in REQUIRED - {"team", "player_clean_key"}:
        x[c] = num(x[c])

    if len(x) != cfg["expected_rows"]:
        raise RuntimeError(
            f"{position} row drift {len(x)} != {cfg['expected_rows']}"
        )
    if int(x["player_clean_key"].nunique()) != cfg["expected_players"]:
        raise RuntimeError(
            f"{position} player drift {x['player_clean_key'].nunique()} "
            f"!= {cfg['expected_players']}"
        )

    seasons = set(x["season"].dropna().astype(int).unique())
    if seasons != set(cfg["seasons"]):
        raise RuntimeError(f"{position} season drift {sorted(seasons)}")

    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        raise RuntimeError(f"{position} duplicate identities")
    if not x["prior_games"].ge(4).all():
        raise RuntimeError(f"{position} prior support drift")

    numeric = [
        "prior8_abs_opportunity_error_mean",
        "prior8_abs_efficiency_error_mean",
        "current_opportunity_error",
        "current_abs_opportunity_error",
        "current_efficiency_error",
        "current_abs_efficiency_error",
        "current_total_error",
        "current_abs_total_error",
    ]
    if not np.isfinite(x[numeric].to_numpy(float)).all():
        raise RuntimeError(f"{position} nonfinite source values")

    nonnegative = [
        "prior8_abs_opportunity_error_mean",
        "prior8_abs_efficiency_error_mean",
        "current_abs_opportunity_error",
        "current_abs_efficiency_error",
        "current_abs_total_error",
    ]
    if not (x[nonnegative] >= 0).all().all():
        raise RuntimeError(f"{position} negative abs/difficulty values")

    decomp = (
        x["current_opportunity_error"]
        + x["current_efficiency_error"]
        - x["current_total_error"]
    ).abs().max()
    abs_total = (
        x["current_total_error"].abs() - x["current_abs_total_error"]
    ).abs().max()
    abs_opp = (
        x["current_opportunity_error"].abs()
        - x["current_abs_opportunity_error"]
    ).abs().max()
    abs_eff = (
        x["current_efficiency_error"].abs()
        - x["current_abs_efficiency_error"]
    ).abs().max()

    if max(decomp, abs_total, abs_opp, abs_eff) > TOL:
        raise RuntimeError(f"{position} decomposition/absolute identity failed")

    g = x.groupby(["season", "week"], sort=False)
    x["efficiency_fragility_rank"] = g[
        "prior8_abs_efficiency_error_mean"
    ].transform(rank_pct)
    x["opportunity_fragility_rank"] = g[
        "prior8_abs_opportunity_error_mean"
    ].transform(rank_pct)
    x["total_error_badness_rank"] = g["current_abs_total_error"].transform(
        rank_pct
    )
    x["efficiency_error_badness_rank"] = g[
        "current_abs_efficiency_error"
    ].transform(rank_pct)

    x["efficiency_trust_bucket"] = "MIDDLE"
    x.loc[
        x["efficiency_fragility_rank"].le(0.25),
        "efficiency_trust_bucket",
    ] = "HIGH_EFFICIENCY_TRUST"
    x.loc[
        x["efficiency_fragility_rank"].ge(0.75),
        "efficiency_trust_bucket",
    ] = "LOW_EFFICIENCY_TRUST"

    x["team_week_cluster"] = (
        x["season"].astype(int).astype(str)
        + "-"
        + x["week"].astype(int).astype(str)
        + "-"
        + x["team"].astype(str)
    )

    integrity = {
        "max_decomposition_gap": float(decomp),
        "max_abs_total_gap": float(abs_total),
        "max_abs_opportunity_gap": float(abs_opp),
        "max_abs_efficiency_gap": float(abs_eff),
    }
    return x.sort_values(keys).reset_index(drop=True), integrity


def metrics(q: pd.DataFrame):
    hi = q.loc[q["efficiency_trust_bucket"].eq("HIGH_EFFICIENCY_TRUST")]
    lo = q.loc[q["efficiency_trust_bucket"].eq("LOW_EFFICIENCY_TRUST")]
    return {
        "rows": int(len(q)),
        "players": int(q["player_clean_key"].nunique()),
        "primary_partial_rank_association": partial_corr(
            q["efficiency_fragility_rank"],
            q["total_error_badness_rank"],
            q["opportunity_fragility_rank"],
        ),
        "secondary_efficiency_partial_rank_association": partial_corr(
            q["efficiency_fragility_rank"],
            q["efficiency_error_badness_rank"],
            q["opportunity_fragility_rank"],
        ),
        "raw_final_abs_error": float(q["current_abs_total_error"].mean()),
        "high_efficiency_trust_raw_final_abs_error": (
            float(hi["current_abs_total_error"].mean()) if len(hi) else np.nan
        ),
        "low_efficiency_trust_raw_final_abs_error": (
            float(lo["current_abs_total_error"].mean()) if len(lo) else np.nan
        ),
        "high_rows": int(len(hi)),
        "low_rows": int(len(lo)),
    }


def bootstrap(q: pd.DataFrame, cluster_col: str, seed: int):
    cols = [
        "efficiency_fragility_rank",
        "total_error_badness_rank",
        "opportunity_fragility_rank",
        cluster_col,
    ]
    z = q[cols].dropna().reset_index(drop=True)
    labels = z[cluster_col].astype(str).to_numpy()
    unique = np.unique(labels)
    groups = {k: np.flatnonzero(labels == k) for k in unique}

    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[k] for k in sampled])
        b = z.iloc[idx]
        vals[i] = partial_corr(
            b["efficiency_fragility_rank"],
            b["total_error_badness_rank"],
            b["opportunity_fragility_rank"],
        )

    vals = vals[np.isfinite(vals)]
    return {
        "reps": int(len(vals)),
        "clusters": int(len(unique)),
        "observed": partial_corr(
            z["efficiency_fragility_rank"],
            z["total_error_badness_rank"],
            z["opportunity_fragility_rank"],
        ),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "p_positive": float(np.mean(vals > 0)),
    }


def evaluate(x: pd.DataFrame, position: str, integrity: dict):
    cfg = CONFIG[position]
    dev = x.loc[x["season"].eq(cfg["dev"])].copy()
    confirm = x.loc[x["season"].eq(cfg["confirm"])].copy()

    md = metrics(dev)
    mc = metrics(confirm)
    mp = metrics(x)

    pooled_player = bootstrap(
        x, "player_clean_key", cfg["player_seeds"][0]
    )
    confirm_player = bootstrap(
        confirm, "player_clean_key", cfg["player_seeds"][1]
    )
    pooled_team = bootstrap(
        x, "team_week_cluster", cfg["team_seeds"][0]
    )
    confirm_team = bootstrap(
        confirm, "team_week_cluster", cfg["team_seeds"][1]
    )

    support = (
        md["rows"] >= cfg["min_dev_rows"]
        and md["players"] >= cfg["min_dev_players"]
        and mc["rows"] >= cfg["min_confirm_rows"]
        and mc["players"] >= cfg["min_confirm_players"]
        and mp["rows"] >= cfg["min_pool_rows"]
        and mp["players"] >= cfg["min_pool_players"]
    )

    gates = {
        "source_integrity": True,
        "support": bool(support),
        "development_primary_positive": bool(
            md["primary_partial_rank_association"] > 0
        ),
        "confirmation_primary_positive": bool(
            mc["primary_partial_rank_association"] > 0
        ),
        "pooled_player_cluster_ci_low_positive": bool(
            pooled_player["ci_low"] > 0
        ),
        "pooled_team_week_cluster_ci_low_positive": bool(
            pooled_team["ci_low"] > 0
        ),
        "confirmation_player_cluster_ci_low_positive": bool(
            confirm_player["ci_low"] > 0
        ),
        "confirmation_team_week_cluster_ci_low_positive": bool(
            confirm_team["ci_low"] > 0
        ),
        "development_secondary_mechanism_positive": bool(
            md["secondary_efficiency_partial_rank_association"] > 0
        ),
        "confirmation_secondary_mechanism_positive": bool(
            mc["secondary_efficiency_partial_rank_association"] > 0
        ),
        "development_high_trust_better_than_low_trust": bool(
            md["high_efficiency_trust_raw_final_abs_error"]
            < md["low_efficiency_trust_raw_final_abs_error"]
        ),
        "confirmation_high_trust_better_than_low_trust": bool(
            mc["high_efficiency_trust_raw_final_abs_error"]
            < mc["low_efficiency_trust_raw_final_abs_error"]
        ),
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "outcomes_2026_read_zero": True,
        "predictive_models_fit_zero": True,
        "projection_distribution_values_changed_zero": True,
        "production_unchanged": True,
    }

    disposition = cfg["qualify"] if all(gates.values()) else cfg["closed"]

    return {
        "position": position,
        "disposition": disposition,
        "integrity": integrity,
        "development_season": cfg["dev"],
        "confirmation_season": cfg["confirm"],
        "development_metrics": md,
        "confirmation_metrics": mc,
        "pooled_metrics": mp,
        "pooled_player_cluster": pooled_player,
        "pooled_team_week_cluster": pooled_team,
        "confirmation_player_cluster": confirm_player,
        "confirmation_team_week_cluster": confirm_team,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-rows", type=Path, required=True)
    ap.add_argument("--te-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        wr, wr_integrity = read_parent(args.wr_rows, "WR")
        te, te_integrity = read_parent(args.te_rows, "TE")

        wr_result = evaluate(wr, "WR", wr_integrity)
        te_result = evaluate(te, "TE", te_integrity)

        qualified = [
            wr_result["disposition"].endswith("_QUALIFIED"),
            te_result["disposition"].endswith("_QUALIFIED"),
        ]
        if all(qualified):
            overall = (
                "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_"
                "BOTH_QUALIFIED"
            )
        elif any(qualified):
            overall = (
                "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_PARTIAL"
            )
        else:
            overall = (
                "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_CLOSED"
            )

        result = {
            "version": VERSION,
            "disposition": overall,
            "wr": wr_result,
            "te": te_result,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "predictive_models_fit": 0,
            "projection_distribution_values_changed": 0,
            "production_changed": False,
            "bootstrap_reps": BOOT_REPS,
        }

        wr.assign(position="WR").to_csv(
            args.out_dir / "wr_efficiency_difficulty_trust_rows.csv",
            index=False,
        )
        te.assign(position="TE").to_csv(
            args.out_dir / "te_efficiency_difficulty_trust_rows.csv",
            index=False,
        )
    except Exception as exc:
        result = {
            "version": VERSION,
            "disposition": (
                "PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_"
                "INTEGRITY_FAILURE"
            ),
            "error": str(exc),
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "predictive_models_fit": 0,
            "projection_distribution_values_changed": 0,
            "production_changed": False,
            "bootstrap_reps": BOOT_REPS,
        }
        (
            args.out_dir
            / "pass_catcher_efficiency_difficulty_trust_selectivity_result.json"
        ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (
        args.out_dir
        / "pass_catcher_efficiency_difficulty_trust_selectivity_result.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
