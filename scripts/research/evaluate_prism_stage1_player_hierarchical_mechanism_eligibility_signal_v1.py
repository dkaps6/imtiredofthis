#!/usr/bin/env python3
"""Score the frozen PRISM Stage-1 player hierarchical mechanism-eligibility signal V1.

This is a meta-signal about the future utility of the already-qualified Stage-1
player-identity participation mechanism. It does not change participation
probabilities or production projections.
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

VERSION = "PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1"
FREEZE_COMMIT = "9ec910700f7febf9e5b1552c0b2d41c837051287"
PARENT_RUN = 38907459996
PARENT_ARTIFACT = 12049120604
PARENT_DIGEST = "sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b"
SOURCE_RUN = 39099745136
SOURCE_ARTIFACT = 12125863397
SOURCE_DIGEST = "sha256:1b81c160ac1f8421d5eed83bb6a6389b5d6872f5a39f468df01b3a98a53d0d38"

FAMILIES = ("QB", "RB", "WR", "TE")
SEASONS = (2024, 2025)
MIN_PLAYER_PRIOR = 3
PSEUDO_COUNT = 3.0
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-12

SUPPORT = {
    "QB": {
        2024: {"rows": 550, "players": 65, "teams": 28},
        2025: {"rows": 700, "players": 70, "teams": 28},
    },
    "RB": {
        2024: {"rows": 900, "players": 105, "teams": 28},
        2025: {"rows": 1100, "players": 120, "teams": 28},
    },
    "WR": {
        2024: {"rows": 1300, "players": 175, "teams": 28},
        2025: {"rows": 1700, "players": 190, "teams": 28},
    },
    "TE": {
        2024: {"rows": 850, "players": 105, "teams": 28},
        2025: {"rows": 1100, "players": 110, "teams": 28},
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_parent(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing Stage-1 parent rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "opponent", "game_key", "family", "player_id",
        "player_name_key", "y", "p_baseline", "p_candidate",
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
        raise RuntimeError(f"forbidden sportsbook columns: {forbidden}")

    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    for c in ("y", "p_baseline", "p_candidate"):
        x[c] = num(x[c])
    for c in ("team", "opponent", "game_key", "family", "player_id", "player_name_key"):
        x[c] = x[c].astype("string").fillna("").str.strip()

    x = x.loc[x["season"].isin(SEASONS) & x["family"].isin(FAMILIES)].copy()
    if x.empty:
        raise RuntimeError("zero Stage-1 rows after eligible filter")
    if x["player_id"].eq("").any() or x["team"].eq("").any() or x["game_key"].eq("").any():
        raise RuntimeError("blank identity in Stage-1 parent rows")
    if x.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate family/player/week/team identity")
    if not x["y"].isin([0, 1]).all():
        raise RuntimeError("nonbinary Stage-1 outcome")
    for c in ("p_baseline", "p_candidate"):
        if not ((x[c] > 0) & (x[c] < 1) & np.isfinite(x[c])).all():
            raise RuntimeError(f"{c} outside open interval")
    y = num(x["y"]).to_numpy(float)
    pb = num(x["p_baseline"]).to_numpy(float)
    pc = num(x["p_candidate"]).to_numpy(float)
    x["delta_brier"] = np.square(y - pb) - np.square(y - pc)
    return x


def build_strict_prior_meta_state(parent: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build same-week-safe, target-player-excluded family prior and player prior."""
    z = parent.sort_values(
        ["season", "week", "family", "team", "game_key", "player_id"]
    ).reset_index(drop=True)

    family_sum: dict[str, float] = defaultdict(float)
    family_n: dict[str, int] = defaultdict(int)
    family_player_sum: dict[tuple[str, str], float] = defaultdict(float)
    family_player_n: dict[tuple[str, str], int] = defaultdict(int)
    player_sum: dict[tuple[str, str], float] = defaultdict(float)
    player_n: dict[tuple[str, str], int] = defaultdict(int)

    family_prior = np.zeros(len(z), dtype=float)
    family_prior_n = np.zeros(len(z), dtype=int)
    player_prior = np.full(len(z), np.nan)
    player_prior_n = np.zeros(len(z), dtype=int)
    chronology_violations = 0
    last_seen: dict[tuple[str, str], tuple[int, int]] = {}

    for (season, week), idxs in z.groupby(["season", "week"], sort=True).groups.items():
        idxs = list(idxs)

        # Read all pregame states for this week before any target-week outcome
        # updates the historical benefit accumulators.
        for i in idxs:
            family = str(z.at[i, "family"])
            pid = str(z.at[i, "player_id"])
            key = (family, pid)
            current = (int(season), int(week))
            prev = last_seen.get(key)
            if prev is not None and current <= prev:
                chronology_violations += 1

            other_n = int(family_n[family] - family_player_n[key])
            other_sum = float(family_sum[family] - family_player_sum[key])
            if other_n > 0:
                family_prior[i] = other_sum / other_n
                family_prior_n[i] = other_n
            else:
                family_prior[i] = 0.0
                family_prior_n[i] = 0

            n = int(player_n[key])
            player_prior_n[i] = n
            if n > 0:
                player_prior[i] = float(player_sum[key] / n)

        # Update only after all rows from the week have captured prior state.
        for i in idxs:
            family = str(z.at[i, "family"])
            pid = str(z.at[i, "player_id"])
            key = (family, pid)
            d = float(z.at[i, "delta_brier"])
            family_sum[family] += d
            family_n[family] += 1
            family_player_sum[key] += d
            family_player_n[key] += 1
            player_sum[key] += d
            player_n[key] += 1
            last_seen[key] = (int(season), int(week))

    z["family_prior_mean"] = family_prior
    z["family_prior_other_player_rows"] = family_prior_n
    z["player_prior_mean"] = player_prior
    z["player_prior_mechanism_rows"] = player_prior_n

    n = z["player_prior_mechanism_rows"].to_numpy(dtype=float)
    pmean = num(z["player_prior_mean"]).to_numpy(dtype=float)
    fmean = num(z["family_prior_mean"]).to_numpy(dtype=float)
    eligible = n >= MIN_PLAYER_PRIOR
    score = fmean.copy()
    score[eligible] = (
        n[eligible] * pmean[eligible] + PSEUDO_COUNT * fmean[eligible]
    ) / (n[eligible] + PSEUDO_COUNT)
    z["hierarchical_expected_benefit"] = score
    z["individual_deviation_eligible"] = eligible
    return z, chronology_violations


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna().copy()
    grouped = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(grouped) < 2:
        return {
            "clusters": int(len(grouped)), "valid_reps": 0,
            "ci95_low": math.nan, "ci95_high": math.nan,
        }
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
        return {
            "players": int(len(vals)), "valid_reps": 0,
            "ci95_low": math.nan, "ci95_high": math.nan,
        }
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


def score_slice(state: pd.DataFrame, family: str, season: int) -> tuple[dict, pd.DataFrame]:
    g = state.loc[
        state["family"].eq(family)
        & state["season"].eq(season)
        & state["player_prior_mechanism_rows"].ge(MIN_PLAYER_PRIOR)
    ].copy()

    req = SUPPORT[family][season]
    support = {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_id"].nunique()) >= req["players"],
        "teams": int(g["team"].nunique()) >= req["teams"],
    }

    if len(g):
        target = num(g["delta_brier"]).to_numpy(float)
        base = num(g["family_prior_mean"]).to_numpy(float)
        cand = num(g["hierarchical_expected_benefit"]).to_numpy(float)
        g["baseline_meta_ae"] = np.abs(target - base)
        g["candidate_meta_ae"] = np.abs(target - cand)
        g["delta_meta_ae"] = g["baseline_meta_ae"] - g["candidate_meta_ae"]
        g["baseline_meta_sqerr"] = np.square(target - base)
        g["candidate_meta_sqerr"] = np.square(target - cand)
        g["predicted_positive_utility"] = cand > 0
    else:
        for c in (
            "baseline_meta_ae", "candidate_meta_ae", "delta_meta_ae",
            "baseline_meta_sqerr", "candidate_meta_sqerr", "predicted_positive_utility",
        ):
            g[c] = []

    players = equal_player_frame(g)
    pvals = (
        num(players["mean_delta_meta_ae"]).dropna()
        if len(players) else pd.Series(dtype=float)
    )

    baseline_rmse = (
        float(np.sqrt(g["baseline_meta_sqerr"].mean())) if len(g) else math.nan
    )
    candidate_rmse = (
        float(np.sqrt(g["candidate_meta_sqerr"].mean())) if len(g) else math.nan
    )
    baseline_spearman = (
        safe_spearman(g["family_prior_mean"], g["delta_brier"]) if len(g) else math.nan
    )
    candidate_spearman = (
        safe_spearman(g["hierarchical_expected_benefit"], g["delta_brier"])
        if len(g) else math.nan
    )

    nonzero = g.loc[num(g["delta_brier"]).abs().gt(TOL)].copy()
    if len(nonzero):
        actual = num(nonzero["delta_brier"]).gt(0)
        predicted = num(nonzero["hierarchical_expected_benefit"]).gt(0)
        sign_accuracy = float((actual == predicted).mean())
    else:
        sign_accuracy = math.nan

    positive_pred = g.loc[g["predicted_positive_utility"].eq(True)]
    nonpositive_pred = g.loc[g["predicted_positive_utility"].eq(False)]

    rec = {
        "family": family,
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
        "predicted_positive_rows": int(len(positive_pred)),
        "predicted_nonpositive_rows": int(len(nonpositive_pred)),
        "positive_pred_mean_target_benefit": (
            float(positive_pred["delta_brier"].mean()) if len(positive_pred) else math.nan
        ),
        "nonpositive_pred_mean_target_benefit": (
            float(nonpositive_pred["delta_brier"].mean()) if len(nonpositive_pred) else math.nan
        ),
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
        "family_prior_equal_player_meta_ae": (
            float(players["baseline_meta_ae"].mean()) if len(players) else math.nan
        ),
    }

    if len(g):
        fi = FAMILIES.index(family)
        rec["player_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_meta_ae", "player_id", BOOT_SEED + season + fi * 100
        )
        rec["game_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_meta_ae", "game_key", BOOT_SEED + 10000 + season + fi * 100
        )
    else:
        rec["player_cluster_bootstrap"] = {}
        rec["game_cluster_bootstrap"] = {}
    rec["equal_player_bootstrap"] = (
        equal_player_bootstrap(
            players, BOOT_SEED + 20000 + season + FAMILIES.index(family) * 100
        )
        if len(players) else {}
    )
    return rec, players


def qualify(dev: dict, confirm: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(confirm["family_prior_equal_player_meta_ae"])
        if np.isfinite(confirm["family_prior_equal_player_meta_ae"])
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
        "confirmation_equal_player_mean_improves": bool(
            confirm["equal_player_mean_improvement"] > 0
        ),
        "confirmation_equal_player_ci_low_gt_zero": bool(
            confirm["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_median_player_nonworse": bool(
            confirm["equal_player_median_improvement"] >= 0
        ),
        "confirmation_player_improved_share_majority": bool(
            confirm["equal_player_improved_share"] > 0.50
        ),
        "confirmation_rmse_improves": bool(
            confirm["candidate_meta_rmse"] < confirm["baseline_meta_rmse"]
        ),
        "confirmation_spearman_improves": bool(
            confirm["candidate_spearman"] > confirm["baseline_spearman"]
        ),
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
        dupes = int(
            state.duplicated(["season", "week", "team", "family", "player_id"]).sum()
        )
        if dupes:
            raise RuntimeError(f"duplicate identities={dupes}")

        metrics: dict[str, dict] = {}
        player_parts = []
        gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}

        for family in FAMILIES:
            metrics[family] = {}
            dev, p24 = score_slice(state, family, 2024)
            confirm, p25 = score_slice(state, family, 2025)
            metrics[family]["2024"] = dev
            metrics[family]["2025"] = confirm
            p24["family"] = family
            p24["season"] = 2024
            p25["family"] = family
            p25["season"] = 2025
            player_parts.extend([p24, p25])
            fgates, ok = qualify(dev, confirm)
            gates[family] = fgates
            dispositions[family] = (
                f"{family}_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED"
                if ok else
                f"{family}_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED"
            )

        qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        overall = (
            "PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_ALL"
            if qualified == len(FAMILIES)
            else "PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL"
            if qualified
            else "PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED_ALL"
        )

        state[
            [
                "season", "week", "game_key", "team", "opponent", "family",
                "player_id", "player_name_key", "y", "p_baseline", "p_candidate",
                "delta_brier", "family_prior_mean", "family_prior_other_player_rows",
                "player_prior_mean", "player_prior_mechanism_rows",
                "hierarchical_expected_benefit", "individual_deviation_eligible",
            ]
        ].to_csv(
            args.out_dir / "stage1_player_hierarchical_eligibility_signal_rows.csv",
            index=False,
        )
        pd.concat(player_parts, ignore_index=True).to_csv(
            args.out_dir / "stage1_player_hierarchical_eligibility_player_diagnostics.csv",
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
            "families": list(FAMILIES),
            "minimum_player_prior": MIN_PLAYER_PRIOR,
            "pseudo_count": PSEUDO_COUNT,
            "same_week_history_allowed": False,
            "family_prior_excludes_target_player": True,
            "metrics": metrics,
            "qualification_gates": gates,
            "family_dispositions": dispositions,
            "disposition": overall,
            "chronology_violations": 0,
            "duplicate_identities": 0,
            "participation_projection_values_changed": 0,
            "stage1_formula_changed": False,
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
            "disposition": "PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_INTEGRITY_FAILURE",
            "error": str(exc),
            "participation_projection_values_changed": 0,
            "stage1_formula_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "production_changed": False,
        }
        (args.out_dir / "stage1_player_hierarchical_eligibility_signal_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "stage1_player_hierarchical_eligibility_signal_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
