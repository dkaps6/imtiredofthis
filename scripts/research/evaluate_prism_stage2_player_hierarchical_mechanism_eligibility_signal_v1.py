#!/usr/bin/env python3
"""Score the frozen PRISM Stage-2 player hierarchical mechanism-eligibility signal V1.

This is a meta-signal about whether an already-qualified Stage-2 mechanism tends
to help a named player. It does not change the underlying Stage-2 projection.
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

VERSION = "PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1"
FREEZE_COMMIT = "8a94c8cae1448520acf283e52c641d3bd631de3a"
PARENT_RUN = 38957575645
PARENT_HEAD = "eba5787c0b06aa5f975008965df7d023b3525ba7"
PARENT_ARTIFACT = 12070774378
PARENT_DIGEST = "sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
SOURCE_RUN = 39098772787
SOURCE_HEAD = "744b2e920592803937e8a8c94e7a0c89d6ebd4eb"
SOURCE_ARTIFACT = 12126215015
SOURCE_DIGEST = "sha256:5249f41f72a248443c8a596de2a3cf4211928369b8867b3f3a4d16e70601c78d"

CELLS = ("RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")
SEASONS = (2024, 2025)
PSEUDO_COUNT = 3.0
MIN_PLAYER_PRIOR = 3
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-12

SUPPORT = {
    "RB_RUSHING": {
        2024: {"rows": 850, "players": 115, "teams": 28},
        2025: {"rows": 1100, "players": 130, "teams": 28},
    },
    "RB_RECEIVING": {
        2024: {"rows": 900, "players": 120, "teams": 28},
        2025: {"rows": 1200, "players": 135, "teams": 28},
    },
    "TE_RECEIVING": {
        2024: {"rows": 900, "players": 115, "teams": 28},
        2025: {"rows": 1100, "players": 120, "teams": 28},
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_parent(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing parent Stage-2 rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "game_key", "team", "cell", "family", "player_id",
        "player_name_key", "authority_q", "baseline_crps", "candidate_crps", "delta_crps",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"parent rows missing columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns present: {forbidden}")

    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["authority_q"] = num(x["authority_q"])
    x["baseline_crps"] = num(x["baseline_crps"])
    x["candidate_crps"] = num(x["candidate_crps"])
    x["delta_crps"] = num(x["delta_crps"])
    for c in ("game_key", "team", "cell", "family", "player_id", "player_name_key"):
        x[c] = x[c].astype("string").fillna("").str.strip()

    x = x.loc[x["season"].isin(SEASONS) & x["cell"].isin(CELLS)].copy()
    if x.empty:
        raise RuntimeError("zero eligible parent rows")
    if x["player_id"].eq("").any() or x["team"].eq("").any() or x["game_key"].eq("").any():
        raise RuntimeError("blank identity in parent rows")
    if x.duplicated(["season", "week", "team", "cell", "player_id"]).any():
        raise RuntimeError("duplicate player/cell/week parent identity")
    if x[["authority_q", "baseline_crps", "candidate_crps", "delta_crps"]].isna().any(axis=None):
        raise RuntimeError("non-finite required parent value")
    gap = (x["baseline_crps"] - x["candidate_crps"] - x["delta_crps"]).abs()
    if float(gap.max()) > 1e-10:
        raise RuntimeError(f"delta_crps identity drift max={float(gap.max())}")
    return x


def add_rank_state(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["authority_rank"] = (
        z.groupby(["season", "week", "team", "cell"], sort=False)["authority_q"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    z["authority_rank_state"] = np.select(
        [
            z["authority_q"].le(TOL),
            z["authority_rank"].eq(1),
            z["authority_rank"].eq(2),
        ],
        ["ZERO", "RANK1", "RANK2"],
        default="RANK3_PLUS",
    )
    return z


def build_strict_prior_meta_state(parent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build current-row-free role and player benefit histories.

    Both role and player histories use canonical-positive parent rows only.
    Role prior excludes the target player's own prior rows.
    Same target week is excluded by assigning the whole week before updating.
    """
    z = add_rank_state(parent)
    z = z.sort_values(
        ["season", "week", "cell", "team", "game_key", "player_id"]
    ).reset_index(drop=True)

    role_sum: dict[tuple[str, str], float] = defaultdict(float)
    role_n: dict[tuple[str, str], int] = defaultdict(int)
    role_player_sum: dict[tuple[str, str, str], float] = defaultdict(float)
    role_player_n: dict[tuple[str, str, str], int] = defaultdict(int)

    cell_sum: dict[str, float] = defaultdict(float)
    cell_n: dict[str, int] = defaultdict(int)
    cell_player_sum: dict[tuple[str, str], float] = defaultdict(float)
    cell_player_n: dict[tuple[str, str], int] = defaultdict(int)

    player_sum: dict[tuple[str, str], float] = defaultdict(float)
    player_n: dict[tuple[str, str], int] = defaultdict(int)

    role_prior = np.zeros(len(z), dtype=float)
    role_prior_n = np.zeros(len(z), dtype=int)
    player_prior = np.full(len(z), np.nan, dtype=float)
    player_prior_n = np.zeros(len(z), dtype=int)
    chronology_violations = 0

    last_seen: dict[tuple[str, str], tuple[int, int]] = {}

    for (season, week), idxs in z.groupby(["season", "week"], sort=True).groups.items():
        idxs = list(idxs)

        # Read state for the entire week before any row from this week updates history.
        for i in idxs:
            cell = str(z.at[i, "cell"])
            rank = str(z.at[i, "authority_rank_state"])
            pid = str(z.at[i, "player_id"])
            key = (cell, pid)
            chronological = (int(season), int(week))
            prev = last_seen.get(key)
            if prev is not None and chronological <= prev:
                chronology_violations += 1

            rkey = (cell, rank)
            rpkey = (cell, rank, pid)
            other_role_n = int(role_n[rkey] - role_player_n[rpkey])
            other_role_sum = float(role_sum[rkey] - role_player_sum[rpkey])

            ckey = cell
            cpkey = (cell, pid)
            other_cell_n = int(cell_n[ckey] - cell_player_n[cpkey])
            other_cell_sum = float(cell_sum[ckey] - cell_player_sum[cpkey])

            if other_role_n > 0:
                role_prior[i] = other_role_sum / other_role_n
                role_prior_n[i] = other_role_n
            elif other_cell_n > 0:
                role_prior[i] = other_cell_sum / other_cell_n
                role_prior_n[i] = other_cell_n
            else:
                role_prior[i] = 0.0
                role_prior_n[i] = 0

            n = int(player_n[key])
            player_prior_n[i] = n
            if n > 0:
                player_prior[i] = float(player_sum[key] / n)

        # Only canonical-positive rows become evidence after the week is fully read.
        for i in idxs:
            cell = str(z.at[i, "cell"])
            rank = str(z.at[i, "authority_rank_state"])
            pid = str(z.at[i, "player_id"])
            key = (cell, pid)
            last_seen[key] = (int(season), int(week))
            if float(z.at[i, "authority_q"]) <= TOL:
                continue
            d = float(z.at[i, "delta_crps"])
            rkey = (cell, rank)
            rpkey = (cell, rank, pid)
            role_sum[rkey] += d
            role_n[rkey] += 1
            role_player_sum[rpkey] += d
            role_player_n[rpkey] += 1
            cell_sum[cell] += d
            cell_n[cell] += 1
            cell_player_sum[(cell, pid)] += d
            cell_player_n[(cell, pid)] += 1
            player_sum[key] += d
            player_n[key] += 1

    z["role_prior_mean"] = role_prior
    z["role_prior_other_player_rows"] = role_prior_n
    z["player_prior_mean"] = player_prior
    z["player_prior_positive_q_rows"] = player_prior_n

    n = z["player_prior_positive_q_rows"].to_numpy(dtype=float)
    pmean = num(z["player_prior_mean"]).to_numpy(dtype=float)
    rmean = num(z["role_prior_mean"]).to_numpy(dtype=float)
    eligible = n >= MIN_PLAYER_PRIOR
    score = rmean.copy()
    score[eligible] = (
        n[eligible] * pmean[eligible] + PSEUDO_COUNT * rmean[eligible]
    ) / (n[eligible] + PSEUDO_COUNT)
    z["hierarchical_expected_benefit"] = score
    z["individual_deviation_eligible"] = eligible
    return z, chronology_violations


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna().copy()
    grouped = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(grouped) < 2:
        return {"clusters": int(len(grouped)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    a = grouped[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for j in range(BOOT_REPS):
        idx = rng.integers(0, len(a), size=len(a))
        totals = a[idx].sum(axis=0)
        vals[j] = totals[0] / totals[1] if totals[1] > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    return {
        "clusters": int(len(a)),
        "valid_reps": int(len(vals)),
        "ci95_low": float(np.quantile(vals, 0.025)) if len(vals) else math.nan,
        "ci95_high": float(np.quantile(vals, 0.975)) if len(vals) else math.nan,
    }


def equal_player_frame(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, x in g.groupby("player_id", sort=True):
        if len(x) < 5:
            continue
        rows.append({
            "player_id": str(pid),
            "rows": int(len(x)),
            "mean_delta_meta_ae": float(x["delta_meta_ae"].mean()),
            "baseline_meta_ae": float(x["baseline_meta_ae"].mean()),
            "candidate_meta_ae": float(x["candidate_meta_ae"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict:
    vals = num(p["mean_delta_meta_ae"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {"players": int(len(vals)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
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


def score_slice(state: pd.DataFrame, cell: str, season: int) -> tuple[dict, pd.DataFrame]:
    g = state.loc[
        state["cell"].eq(cell)
        & state["season"].eq(season)
        & state["authority_q"].gt(TOL)
        & state["player_prior_positive_q_rows"].ge(MIN_PLAYER_PRIOR)
    ].copy()

    req = SUPPORT[cell][season]
    support = {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_id"].nunique()) >= req["players"],
        "teams": int(g["team"].nunique()) >= req["teams"],
    }

    if len(g):
        target = num(g["delta_crps"]).to_numpy(float)
        base = num(g["role_prior_mean"]).to_numpy(float)
        cand = num(g["hierarchical_expected_benefit"]).to_numpy(float)
        g["baseline_meta_ae"] = np.abs(target - base)
        g["candidate_meta_ae"] = np.abs(target - cand)
        g["delta_meta_ae"] = g["baseline_meta_ae"] - g["candidate_meta_ae"]
        g["baseline_meta_sqerr"] = np.square(target - base)
        g["candidate_meta_sqerr"] = np.square(target - cand)
        g["predicted_eligible"] = cand > 0
    else:
        for c in (
            "baseline_meta_ae", "candidate_meta_ae", "delta_meta_ae",
            "baseline_meta_sqerr", "candidate_meta_sqerr", "predicted_eligible",
        ):
            g[c] = []

    players = equal_player_frame(g)
    baseline_rmse = float(np.sqrt(g["baseline_meta_sqerr"].mean())) if len(g) else math.nan
    candidate_rmse = float(np.sqrt(g["candidate_meta_sqerr"].mean())) if len(g) else math.nan
    baseline_spearman = safe_spearman(g["role_prior_mean"], g["delta_crps"]) if len(g) else math.nan
    candidate_spearman = safe_spearman(g["hierarchical_expected_benefit"], g["delta_crps"]) if len(g) else math.nan

    pvals = num(players["mean_delta_meta_ae"]).dropna() if len(players) else pd.Series(dtype=float)
    baseline_equal_player_meta_ae = (
        float(players["baseline_meta_ae"].mean()) if len(players) else math.nan
    )

    eligible_rows = g.loc[g["predicted_eligible"].eq(True)]
    ineligible_rows = g.loc[g["predicted_eligible"].eq(False)]
    nonzero = g.loc[num(g["delta_crps"]).abs().gt(TOL)].copy()
    if len(nonzero):
        nonzero["actual_positive"] = num(nonzero["delta_crps"]).gt(0)
        nonzero["predicted_positive"] = num(nonzero["hierarchical_expected_benefit"]).gt(0)
        sign_accuracy = float((nonzero["actual_positive"] == nonzero["predicted_positive"]).mean())
    else:
        sign_accuracy = math.nan

    rec = {
        "cell": cell,
        "season": season,
        "rows": int(len(g)),
        "players": int(g["player_id"].nunique()),
        "teams": int(g["team"].nunique()),
        "support_gates": support,
        "mean_delta_meta_ae": float(g["delta_meta_ae"].mean()) if len(g) else math.nan,
        "baseline_meta_ae": float(g["baseline_meta_ae"].mean()) if len(g) else math.nan,
        "candidate_meta_ae": float(g["candidate_meta_ae"].mean()) if len(g) else math.nan,
        "baseline_meta_rmse": baseline_rmse,
        "candidate_meta_rmse": candidate_rmse,
        "baseline_spearman": baseline_spearman,
        "candidate_spearman": candidate_spearman,
        "sign_accuracy_nonzero": sign_accuracy,
        "predicted_eligible_rows": int(len(eligible_rows)),
        "predicted_ineligible_rows": int(len(ineligible_rows)),
        "eligible_mean_target_benefit": float(eligible_rows["delta_crps"].mean()) if len(eligible_rows) else math.nan,
        "ineligible_mean_target_benefit": float(ineligible_rows["delta_crps"].mean()) if len(ineligible_rows) else math.nan,
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
        "baseline_equal_player_meta_ae": baseline_equal_player_meta_ae,
    }

    if len(g):
        rec["player_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_meta_ae", "player_id", BOOT_SEED + season + CELLS.index(cell) * 100
        )
        rec["game_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_meta_ae", "game_key", BOOT_SEED + 10000 + season + CELLS.index(cell) * 100
        )
    else:
        rec["player_cluster_bootstrap"] = {}
        rec["game_cluster_bootstrap"] = {}

    rec["equal_player_bootstrap"] = (
        equal_player_bootstrap(players, BOOT_SEED + 20000 + season + CELLS.index(cell) * 100)
        if len(players) else {}
    )
    return rec, players


def qualify(dev: dict, confirm: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(confirm["baseline_equal_player_meta_ae"])
        if np.isfinite(confirm["baseline_equal_player_meta_ae"])
        else math.nan
    )
    gates = {
        "development_support": bool(all(dev["support_gates"].values())),
        "development_mean_meta_ae_improves": bool(dev["mean_delta_meta_ae"] > 0),
        "development_rmse_improves": bool(dev["candidate_meta_rmse"] < dev["baseline_meta_rmse"]),
        "development_spearman_improves": bool(dev["candidate_spearman"] > dev["baseline_spearman"]),
        "confirmation_support": bool(all(confirm["support_gates"].values())),
        "confirmation_mean_meta_ae_improves": bool(confirm["mean_delta_meta_ae"] > 0),
        "confirmation_player_cluster_ci_low_gt_zero": bool(
            confirm["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_game_cluster_ci_low_gt_zero": bool(
            confirm["game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_equal_player_mean_improves": bool(confirm["equal_player_mean_improvement"] > 0),
        "confirmation_equal_player_ci_low_gt_zero": bool(
            confirm["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_median_player_nonworse": bool(confirm["equal_player_median_improvement"] >= 0),
        "confirmation_player_improved_share_majority": bool(confirm["equal_player_improved_share"] > 0.50),
        "confirmation_rmse_improves": bool(confirm["candidate_meta_rmse"] < confirm["baseline_meta_rmse"]),
        "confirmation_spearman_improves": bool(confirm["candidate_spearman"] > confirm["baseline_spearman"]),
        "confirmation_tail_harm_guard": bool(
            np.isfinite(tail_floor)
            and confirm["equal_player_p10_improvement"] >= tail_floor
        ),
    }
    gates["tail_floor"] = tail_floor
    return gates, all(v for k, v in gates.items() if k != "tail_floor")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parent = load_parent(args.parent_rows)
        state, chronology_violations = build_strict_prior_meta_state(parent)
        if chronology_violations:
            raise RuntimeError(f"chronology violations={chronology_violations}")
        dupes = int(state.duplicated(["season", "week", "team", "cell", "player_id"]).sum())
        if dupes:
            raise RuntimeError(f"duplicate identities={dupes}")

        metrics: dict[str, dict] = {}
        player_parts = []
        gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}

        for cell in CELLS:
            metrics[cell] = {}
            dev, p24 = score_slice(state, cell, 2024)
            confirm, p25 = score_slice(state, cell, 2025)
            metrics[cell]["2024"] = dev
            metrics[cell]["2025"] = confirm
            p24["cell"] = cell
            p24["season"] = 2024
            p25["cell"] = cell
            p25["season"] = 2025
            player_parts.extend([p24, p25])
            cgates, ok = qualify(dev, confirm)
            gates[cell] = cgates
            dispositions[cell] = (
                f"{cell}_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED"
                if ok else
                f"{cell}_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED"
            )

        qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        overall = (
            "PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_ALL"
            if qualified == len(CELLS)
            else "PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL"
            if qualified
            else "PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED_ALL"
        )

        state_out = state[
            [
                "season", "week", "game_key", "team", "cell", "family",
                "player_id", "player_name_key", "authority_q", "authority_rank",
                "authority_rank_state", "role_prior_mean", "role_prior_other_player_rows",
                "player_prior_mean", "player_prior_positive_q_rows",
                "hierarchical_expected_benefit", "individual_deviation_eligible",
                "delta_crps",
            ]
        ].copy()
        state_out.to_csv(args.out_dir / "stage2_player_hierarchical_eligibility_signal_rows.csv", index=False)
        pd.concat(player_parts, ignore_index=True).to_csv(
            args.out_dir / "stage2_player_hierarchical_eligibility_player_diagnostics.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_head": PARENT_HEAD,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "source_run": SOURCE_RUN,
            "source_head": SOURCE_HEAD,
            "source_artifact": SOURCE_ARTIFACT,
            "source_digest": SOURCE_DIGEST,
            "cells": list(CELLS),
            "pseudo_count": PSEUDO_COUNT,
            "minimum_player_prior": MIN_PLAYER_PRIOR,
            "primary_population": "CANONICAL_POSITIVE_AUTHORITY_Q_GT_1E_12",
            "same_week_history_allowed": False,
            "role_prior_excludes_target_player": True,
            "structural_zero_rows_in_history": False,
            "metrics": metrics,
            "qualification_gates": gates,
            "cell_dispositions": dispositions,
            "disposition": overall,
            "chronology_violations": 0,
            "duplicate_identities": 0,
            "projection_values_changed": 0,
            "stage2_formula_changed": False,
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
            "disposition": "PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_INTEGRITY_FAILURE",
            "error": str(exc),
            "projection_values_changed": 0,
            "stage2_formula_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "stage2_player_hierarchical_eligibility_signal_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "stage2_player_hierarchical_eligibility_signal_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
