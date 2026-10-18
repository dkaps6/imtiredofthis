#!/usr/bin/env python3
"""Score frozen PRISM WR/TE player signed component-bias V1.

Predicts future signed opportunity/efficiency component error from strictly-prior
same-player signed component history shrunk toward a target-player-excluded
position prior. Research-only: no component or projection mean changes.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VERSION = "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL"
FREEZE_COMMIT = "dcf70236709d0f586c72bcaa4e3550dbefe31efb"

PARENT_RUN = 39061606501
PARENT_ARTIFACT = 12111278777
PARENT_DIGEST = "sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9"

SOURCE_RUN = 39100442923
SOURCE_ARTIFACT = 12126685270
SOURCE_DIGEST = "sha256:d29de06947a0b3aead1e20f05b66d5cdfacfa393d8b1c535808533ed3b0a0d7a"

PSEUDO_COUNT = 4.0
MIN_PRIOR = 4
MAX_PRIOR = 8
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-10

CELLS = (
    ("WR", "OPPORTUNITY", 2023, 2024),
    ("WR", "EFFICIENCY", 2023, 2024),
    ("TE", "OPPORTUNITY", 2024, 2025),
    ("TE", "EFFICIENCY", 2024, 2025),
)

SUPPORT = {
    ("WR", 2023): {"rows": 700, "players": 120, "teams": 28},
    ("WR", 2024): {"rows": 1100, "players": 135, "teams": 28},
    ("TE", 2024): {"rows": 575, "players": 70, "teams": 28},
    ("TE", 2025): {"rows": 600, "players": 75, "teams": 28},
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true": True, "false": False})
        .fillna(False).astype(bool)
    )


def load_parent(path: Path, position: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {position} parent rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "player_clean_key",
        "prior_component_rows_available",
        "prior_component_min_week", "prior_component_max_week",
        "instability_state_source_eligible", "identity_gap",
        "prior8_opportunity_error_mean", "prior8_efficiency_error_mean",
        "current_opportunity_error", "current_efficiency_error",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"{position} parent missing columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in {position}: {forbidden}")

    for c in (
        "season", "week", "prior_component_rows_available",
        "prior_component_min_week", "prior_component_max_week", "identity_gap",
        "prior8_opportunity_error_mean", "prior8_efficiency_error_mean",
        "current_opportunity_error", "current_efficiency_error",
    ):
        x[c] = num(x[c])
    x["season"] = x["season"].astype("Int64")
    x["week"] = x["week"].astype("Int64")
    x["prior_component_rows_available"] = x["prior_component_rows_available"].astype("Int64")
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    x["source_eligible"] = bool_series(x["instability_state_source_eligible"])
    x["position"] = position

    expected = {2023, 2024} if position == "WR" else {2024, 2025}
    x = x.loc[x["season"].isin(expected)].copy()
    if x.empty:
        raise RuntimeError(f"zero {position} rows after season filter")
    if x["team"].eq("").any() or x["player_clean_key"].eq("").any():
        raise RuntimeError(f"blank identity in {position} rows")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError(f"duplicate {position} player/week/team identity")
    if not np.isfinite(x["identity_gap"]).all() or float(x["identity_gap"].abs().max()) > TOL:
        raise RuntimeError(f"{position} decomposition identity failure")

    eligible = x.loc[x["source_eligible"]].copy()
    if (
        eligible["prior_component_rows_available"].lt(MIN_PRIOR)
        | eligible["prior_component_rows_available"].gt(MAX_PRIOR)
        | eligible["prior_component_min_week"].ge(eligible["week"])
        | eligible["prior_component_max_week"].ge(eligible["week"])
    ).any():
        raise RuntimeError(f"{position} source eligibility chronology/support drift")

    required_numeric = (
        "prior8_opportunity_error_mean", "prior8_efficiency_error_mean",
        "current_opportunity_error", "current_efficiency_error",
    )
    if not all(np.isfinite(eligible[c]).all() for c in required_numeric):
        raise RuntimeError(f"{position} nonfinite signed component field")
    return x


def load_source_result(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing source result: {path}")
    p = json.loads(path.read_text())
    if p.get("version") != "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SOURCE_AUDIT":
        raise RuntimeError("wrong source result version")
    if p.get("disposition") != "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY_ALL":
        raise RuntimeError(f"source not ready: {p.get('disposition')}")
    if int(p.get("predictive_relationships_scored", -1)) != 0:
        raise RuntimeError("source audit unexpectedly scored predictive relationship")
    return p


def add_position_priors(x: pd.DataFrame, component: str) -> pd.DataFrame:
    current_col = (
        "current_opportunity_error" if component == "OPPORTUNITY"
        else "current_efficiency_error"
    )
    player_prior_col = (
        "prior8_opportunity_error_mean" if component == "OPPORTUNITY"
        else "prior8_efficiency_error_mean"
    )

    z = x.sort_values(
        ["season", "week", "team", "player_clean_key"], kind="mergesort"
    ).reset_index(drop=True).copy()

    total_sum: dict[int, float] = defaultdict(float)
    total_n: dict[int, int] = defaultdict(int)
    player_sum: dict[tuple[int, str], float] = defaultdict(float)
    player_n: dict[tuple[int, str], int] = defaultdict(int)

    position_prior = np.zeros(len(z), dtype=float)
    position_prior_n = np.zeros(len(z), dtype=int)

    for (season, week), idxs in z.groupby(["season", "week"], sort=True).groups.items():
        idxs = list(idxs)
        season_i = int(season)

        # Capture all target states before this week's outcomes update priors.
        for i in idxs:
            pid = str(z.at[i, "player_clean_key"])
            key = (season_i, pid)
            n_other = int(total_n[season_i] - player_n[key])
            sum_other = float(total_sum[season_i] - player_sum[key])
            position_prior_n[i] = n_other
            position_prior[i] = sum_other / n_other if n_other > 0 else 0.0

        # Only now may this week's outcomes enter future-week priors.
        for i in idxs:
            pid = str(z.at[i, "player_clean_key"])
            val = float(z.at[i, current_col])
            if not math.isfinite(val):
                raise RuntimeError(f"nonfinite {component} current error")
            key = (season_i, pid)
            total_sum[season_i] += val
            total_n[season_i] += 1
            player_sum[key] += val
            player_n[key] += 1

    z["position_prior_mean"] = position_prior
    z["position_prior_other_player_rows"] = position_prior_n
    n = num(z["prior_component_rows_available"]).to_numpy(float)
    pp = num(z[player_prior_col]).to_numpy(float)
    fp = num(z["position_prior_mean"]).to_numpy(float)
    z["hierarchical_signed_bias"] = (n * pp + PSEUDO_COUNT * fp) / (n + PSEUDO_COUNT)
    z["target_component_error"] = num(z[current_col])
    z["player_prior8_signed_bias"] = num(z[player_prior_col])
    z["component"] = component
    z["team_week_cluster"] = (
        z["season"].astype("string") + "-" + z["week"].astype("string")
        + "-" + z["team"].astype("string")
    )
    return z


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna().copy()
    grouped = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(grouped) < 2:
        return {"clusters": int(len(grouped)), "valid_reps": 0,
                "ci95_low": math.nan, "ci95_high": math.nan}
    a = grouped[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for j in range(BOOT_REPS):
        idx = rng.integers(0, len(a), size=len(a))
        total = a[idx].sum(axis=0)
        vals[j] = total[0] / total[1] if total[1] > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    return {
        "clusters": int(len(a)),
        "valid_reps": int(len(vals)),
        "ci95_low": float(np.quantile(vals, 0.025)) if len(vals) else math.nan,
        "ci95_high": float(np.quantile(vals, 0.975)) if len(vals) else math.nan,
    }


def equal_player_frame(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, p in g.groupby("player_clean_key", sort=True):
        if len(p) < 5:
            continue
        rows.append({
            "player_clean_key": str(pid),
            "rows": int(len(p)),
            "mean_delta_meta_ae": float(p["delta_meta_ae"].mean()),
            "baseline_meta_ae": float(p["baseline_meta_ae"].mean()),
            "candidate_meta_ae": float(p["candidate_meta_ae"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict:
    vals = num(p["mean_delta_meta_ae"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {"players": int(len(vals)), "valid_reps": 0,
                "ci95_low": math.nan, "ci95_high": math.nan}
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for j in range(BOOT_REPS):
        out[j] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "players": int(len(vals)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(out, 0.025)),
        "ci95_high": float(np.quantile(out, 0.975)),
    }


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    aa = num(a).to_numpy(float)
    bb = num(b).to_numpy(float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3 or np.unique(aa[mask]).size < 2 or np.unique(bb[mask]).size < 2:
        return math.nan
    return float(spearmanr(aa[mask], bb[mask]).statistic)


def sign_accuracy(pred: pd.Series, target: pd.Series) -> float:
    p = num(pred).to_numpy(float)
    y = num(target).to_numpy(float)
    mask = np.isfinite(p) & np.isfinite(y) & (np.abs(y) > 1e-12)
    if not mask.any():
        return math.nan
    return float((np.sign(p[mask]) == np.sign(y[mask])).mean())


def score_slice(state: pd.DataFrame, position: str, component: str, season: int) -> tuple[dict, pd.DataFrame]:
    g = state.loc[
        state["position"].eq(position)
        & state["component"].eq(component)
        & state["season"].eq(season)
        & state["source_eligible"].eq(True)
    ].copy()

    req = SUPPORT[(position, season)]
    support = {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_clean_key"].nunique()) >= req["players"],
        "teams": int(g["team"].nunique()) >= req["teams"],
        "chronology": bool(
            (
                g["prior_component_rows_available"].between(MIN_PRIOR, MAX_PRIOR)
                & g["prior_component_min_week"].lt(g["week"])
                & g["prior_component_max_week"].lt(g["week"])
            ).all()
        ),
        "decomposition_identity": bool(
            len(g) > 0 and float(g["identity_gap"].abs().max()) <= TOL
        ),
    }

    target = num(g["target_component_error"]).to_numpy(float)
    base = num(g["position_prior_mean"]).to_numpy(float)
    cand = num(g["hierarchical_signed_bias"]).to_numpy(float)
    g["baseline_meta_ae"] = np.abs(target - base)
    g["candidate_meta_ae"] = np.abs(target - cand)
    g["delta_meta_ae"] = g["baseline_meta_ae"] - g["candidate_meta_ae"]
    g["baseline_meta_sqerr"] = np.square(target - base)
    g["candidate_meta_sqerr"] = np.square(target - cand)
    g["baseline_signed_residual"] = target - base
    g["candidate_signed_residual"] = target - cand

    players = equal_player_frame(g)
    pvals = (
        num(players["mean_delta_meta_ae"]).dropna()
        if len(players) else pd.Series(dtype=float)
    )
    base_rmse = float(np.sqrt(g["baseline_meta_sqerr"].mean())) if len(g) else math.nan
    cand_rmse = float(np.sqrt(g["candidate_meta_sqerr"].mean())) if len(g) else math.nan
    base_spear = safe_spearman(g["position_prior_mean"], g["target_component_error"]) if len(g) else math.nan
    cand_spear = safe_spearman(g["hierarchical_signed_bias"], g["target_component_error"]) if len(g) else math.nan

    cell_i = [c[:2] for c in CELLS].index((position, component))
    rec = {
        "position": position,
        "component": component,
        "season": season,
        "rows": int(len(g)),
        "players": int(g["player_clean_key"].nunique()),
        "teams": int(g["team"].nunique()),
        "support_gates": support,
        "mean_delta_meta_ae": float(g["delta_meta_ae"].mean()) if len(g) else math.nan,
        "baseline_meta_ae": float(g["baseline_meta_ae"].mean()) if len(g) else math.nan,
        "candidate_meta_ae": float(g["candidate_meta_ae"].mean()) if len(g) else math.nan,
        "baseline_meta_rmse": base_rmse,
        "candidate_meta_rmse": cand_rmse,
        "baseline_spearman": base_spear,
        "candidate_spearman": cand_spear,
        "baseline_sign_accuracy": sign_accuracy(g["position_prior_mean"], g["target_component_error"]) if len(g) else math.nan,
        "candidate_sign_accuracy": sign_accuracy(g["hierarchical_signed_bias"], g["target_component_error"]) if len(g) else math.nan,
        "baseline_mean_signed_residual": float(g["baseline_signed_residual"].mean()) if len(g) else math.nan,
        "candidate_mean_signed_residual": float(g["candidate_signed_residual"].mean()) if len(g) else math.nan,
        "evaluable_players_ge5": int(len(players)),
        "equal_player_mean_improvement": float(pvals.mean()) if len(pvals) else math.nan,
        "equal_player_median_improvement": float(pvals.median()) if len(pvals) else math.nan,
        "equal_player_improved_share": float((pvals > 0).mean()) if len(pvals) else math.nan,
        "equal_player_p10_improvement": float(pvals.quantile(0.10)) if len(pvals) else math.nan,
        "equal_player_p25_improvement": float(pvals.quantile(0.25)) if len(pvals) else math.nan,
        "equal_player_p50_improvement": float(pvals.quantile(0.50)) if len(pvals) else math.nan,
        "equal_player_p75_improvement": float(pvals.quantile(0.75)) if len(pvals) else math.nan,
        "equal_player_p90_improvement": float(pvals.quantile(0.90)) if len(pvals) else math.nan,
        "worst_named_player_mean_improvement": float(pvals.min()) if len(pvals) else math.nan,
        "baseline_equal_player_meta_ae": float(players["baseline_meta_ae"].mean()) if len(players) else math.nan,
        "player_cluster_bootstrap": cluster_bootstrap(
            g, "delta_meta_ae", "player_clean_key", BOOT_SEED + season + cell_i * 100
        ) if len(g) else {},
        "team_week_cluster_bootstrap": cluster_bootstrap(
            g, "delta_meta_ae", "team_week_cluster", BOOT_SEED + 10000 + season + cell_i * 100
        ) if len(g) else {},
        "equal_player_bootstrap": equal_player_bootstrap(
            players, BOOT_SEED + 20000 + season + cell_i * 100
        ) if len(players) else {},
    }
    return rec, players


def qualify(dev: dict, conf: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(conf["baseline_equal_player_meta_ae"])
        if np.isfinite(conf["baseline_equal_player_meta_ae"]) else math.nan
    )
    gates = {
        "development_support": bool(all(dev["support_gates"].values())),
        "development_mean_meta_ae_improves": bool(dev["mean_delta_meta_ae"] > 0),
        "development_rmse_improves": bool(dev["candidate_meta_rmse"] < dev["baseline_meta_rmse"]),
        "development_spearman_improves": bool(dev["candidate_spearman"] > dev["baseline_spearman"]),
        "confirmation_support": bool(all(conf["support_gates"].values())),
        "confirmation_mean_meta_ae_improves": bool(conf["mean_delta_meta_ae"] > 0),
        "confirmation_player_cluster_ci_low_gt_zero": bool(
            conf["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_team_week_cluster_ci_low_gt_zero": bool(
            conf["team_week_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_equal_player_mean_improves": bool(conf["equal_player_mean_improvement"] > 0),
        "confirmation_equal_player_ci_low_gt_zero": bool(
            conf["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_median_player_nonworse": bool(conf["equal_player_median_improvement"] >= 0),
        "confirmation_player_improved_share_majority": bool(conf["equal_player_improved_share"] > 0.50),
        "confirmation_rmse_improves": bool(conf["candidate_meta_rmse"] < conf["baseline_meta_rmse"]),
        "confirmation_spearman_improves": bool(conf["candidate_spearman"] > conf["baseline_spearman"]),
        "confirmation_tail_harm_guard": bool(
            np.isfinite(tail_floor) and conf["equal_player_p10_improvement"] >= tail_floor
        ),
    }
    gates["tail_floor"] = tail_floor
    ok = all(v for k, v in gates.items() if k != "tail_floor")
    return gates, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-parent-rows", required=True, type=Path)
    ap.add_argument("--te-parent-rows", required=True, type=Path)
    ap.add_argument("--source-result", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        load_source_result(args.source_result)
        parents = {
            "WR": load_parent(args.wr_parent_rows, "WR"),
            "TE": load_parent(args.te_parent_rows, "TE"),
        }

        states = []
        for position, x in parents.items():
            states.append(add_position_priors(x, "OPPORTUNITY"))
            states.append(add_position_priors(x, "EFFICIENCY"))
        state = pd.concat(states, ignore_index=True, sort=False)

        metrics: dict[str, dict] = {}
        gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}
        player_parts = []

        for position, component, dev_season, conf_season in CELLS:
            key = f"{position}_{component}"
            dev, pdev = score_slice(state, position, component, dev_season)
            conf, pconf = score_slice(state, position, component, conf_season)
            metrics[key] = {str(dev_season): dev, str(conf_season): conf}
            pdev["position"] = position
            pdev["component"] = component
            pdev["season"] = dev_season
            pconf["position"] = position
            pconf["component"] = component
            pconf["season"] = conf_season
            player_parts.extend([pdev, pconf])
            cgates, ok = qualify(dev, conf)
            gates[key] = cgates
            dispositions[key] = (
                f"{key}_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED"
                if ok else f"{key}_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_CLOSED"
            )

        n_qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        overall = (
            "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_ALL"
            if n_qualified == len(CELLS)
            else "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_QUALIFIED_PARTIAL"
            if n_qualified
            else "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_CLOSED_ALL"
        )

        keep = [
            "position", "component", "season", "week", "team", "player_clean_key",
            "source_eligible", "prior_component_rows_available",
            "prior_component_min_week", "prior_component_max_week",
            "player_prior8_signed_bias", "position_prior_mean",
            "position_prior_other_player_rows", "hierarchical_signed_bias",
            "target_component_error", "identity_gap", "team_week_cluster",
        ]
        state.loc[state["source_eligible"], keep].to_csv(
            args.out_dir / "wr_te_player_signed_component_bias_signal_rows.csv",
            index=False,
        )
        pd.concat(player_parts, ignore_index=True).to_csv(
            args.out_dir / "wr_te_player_signed_component_bias_player_diagnostics.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "source_run": SOURCE_RUN,
            "source_artifact": SOURCE_ARTIFACT,
            "source_digest": SOURCE_DIGEST,
            "pseudo_count": PSEUDO_COUNT,
            "minimum_player_prior": MIN_PRIOR,
            "maximum_player_prior": MAX_PRIOR,
            "position_prior_excludes_target_player": True,
            "same_week_history_allowed": False,
            "cells": [f"{p}_{c}" for p, c, _, _ in CELLS],
            "metrics": metrics,
            "qualification_gates": gates,
            "cell_dispositions": dispositions,
            "disposition": overall,
            "component_means_changed": 0,
            "projection_means_changed": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SIGNAL_INTEGRITY_FAILURE",
            "error": str(exc),
            "component_means_changed": 0,
            "projection_means_changed": 0,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "wr_te_player_signed_component_bias_signal_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "wr_te_player_signed_component_bias_signal_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
