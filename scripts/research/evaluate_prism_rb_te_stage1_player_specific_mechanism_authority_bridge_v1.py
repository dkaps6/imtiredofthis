#!/usr/bin/env python3
"""Score the frozen RB/TE player-specific Stage-1 mechanism-authority bridge V1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1"
FREEZE_COMMIT = "ba405d3e67495af8a812362b3401250476fff5f1"
PARENT_RUN = 38907459996
PARENT_ARTIFACT = 12049120604
PARENT_DIGEST = "sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b"
SIGNAL_RUN = 39099930673
SIGNAL_ARTIFACT = 12126251820
SIGNAL_DIGEST = "sha256:9ef8c1c7e07fd4be381a745b5eb96d5ed976809d88aff478e623a5d74794f519"

FAMILIES = ("RB", "TE")
SEASONS = (2024, 2025)
MIN_MECHANISM_HISTORY = 3
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-12

SUPPORT = {
    "RB": {
        2024: {"rows": 800, "players": 80, "positives": 500, "zeros": 75, "evaluable_players": 55, "teams": 28},
        2025: {"rows": 800, "players": 80, "positives": 500, "zeros": 75, "evaluable_players": 55, "teams": 28},
    },
    "TE": {
        2024: {"rows": 700, "players": 75, "positives": 550, "zeros": 25, "evaluable_players": 50, "teams": 28},
        2025: {"rows": 700, "players": 75, "positives": 550, "zeros": 25, "evaluable_players": 50, "teams": 28},
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def binary_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    if not np.isfinite(pp).all() or (pp <= 0).any() or (pp >= 1).any():
        raise RuntimeError("probability outside open interval")
    return -(yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp))


def personalize_probability(
    p_baseline: np.ndarray,
    p_candidate: np.ndarray,
    prior_rows: np.ndarray,
    expected_benefit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen authority rule; does not inspect target y."""
    pb = np.asarray(p_baseline, dtype=float)
    pc = np.asarray(p_candidate, dtype=float)
    n = np.asarray(prior_rows, dtype=int)
    h = np.asarray(expected_benefit, dtype=float)
    if not (len(pb) == len(pc) == len(n) == len(h)):
        raise RuntimeError("authority input length mismatch")
    fallback = (n >= MIN_MECHANISM_HISTORY) & (h <= 0)
    out = pc.copy()
    out[fallback] = pb[fallback]
    return out, fallback


def read_parent(path: Path) -> pd.DataFrame:
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
        raise RuntimeError(f"parent missing columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in parent: {forbidden}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["y"] = num(x["y"])
    x["p_baseline"] = num(x["p_baseline"])
    x["p_candidate"] = num(x["p_candidate"])
    for c in ("team", "opponent", "game_key", "family", "player_id", "player_name_key"):
        x[c] = x[c].astype("string").fillna("").str.strip()
    x = x.loc[x["season"].isin(SEASONS) & x["family"].isin(FAMILIES)].copy()
    if x.empty:
        raise RuntimeError("zero eligible parent rows")
    if x.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate parent identity")
    if not x["y"].isin([0, 1]).all():
        raise RuntimeError("nonbinary parent target")
    for c in ("p_baseline", "p_candidate"):
        if not ((x[c] > 0) & (x[c] < 1) & np.isfinite(x[c])).all():
            raise RuntimeError(f"{c} outside open interval")
    return x


def read_signal(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing Stage-1 signal rows: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "game_key", "team", "family", "player_id",
        "player_prior_mechanism_rows", "hierarchical_expected_benefit",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"signal missing columns: {sorted(missing)}")
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in signal: {forbidden}")
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["player_prior_mechanism_rows"] = num(x["player_prior_mechanism_rows"]).astype("Int64")
    x["hierarchical_expected_benefit"] = num(x["hierarchical_expected_benefit"])
    for c in ("game_key", "team", "family", "player_id"):
        x[c] = x[c].astype("string").fillna("").str.strip()
    x = x.loc[x["season"].isin(SEASONS) & x["family"].isin(FAMILIES)].copy()
    if x.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate signal identity")
    if x[["player_prior_mechanism_rows", "hierarchical_expected_benefit"]].isna().any(axis=None):
        raise RuntimeError("non-finite signal authority state")
    return x


def prepare(parent: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "game_key", "team", "family", "player_id"]
    s = signal[keys + ["player_prior_mechanism_rows", "hierarchical_expected_benefit"]].copy()
    z = parent.merge(s, on=keys, how="left", validate="one_to_one")
    if z[["player_prior_mechanism_rows", "hierarchical_expected_benefit"]].isna().any(axis=None):
        raise RuntimeError("signal attach miss")
    pp, fallback = personalize_probability(
        z["p_baseline"].to_numpy(float),
        z["p_candidate"].to_numpy(float),
        z["player_prior_mechanism_rows"].to_numpy(int),
        z["hierarchical_expected_benefit"].to_numpy(float),
    )
    z["p_personalized"] = pp
    z["authority_fallback_to_family_prior"] = fallback
    z["meta_low_support"] = z["player_prior_mechanism_rows"].lt(MIN_MECHANISM_HISTORY)

    y = z["y"].to_numpy(float)
    pu = z["p_candidate"].to_numpy(float)
    ppers = z["p_personalized"].to_numpy(float)
    z["universal_stage1_brier"] = np.square(y - pu)
    z["personalized_stage1_brier"] = np.square(y - ppers)
    z["delta_personalized_brier"] = (
        z["universal_stage1_brier"] - z["personalized_stage1_brier"]
    )
    z["universal_stage1_logloss"] = binary_logloss(y, pu)
    z["personalized_stage1_logloss"] = binary_logloss(y, ppers)
    z["delta_personalized_logloss"] = (
        z["universal_stage1_logloss"] - z["personalized_stage1_logloss"]
    )
    return z


def cluster_bootstrap(
    df: pd.DataFrame, value_col: str, cluster_col: str, seed: int
) -> dict:
    z = df[[value_col, cluster_col]].dropna()
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
        vals[j] = total[0] / total[1]
    return {
        "clusters": int(len(a)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


def equal_player_frame(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, x in g.groupby("player_id", sort=True):
        if len(x) < 5:
            continue
        rows.append({
            "player_id": str(pid),
            "rows": int(len(x)),
            "mean_delta_brier": float(x["delta_personalized_brier"].mean()),
            "universal_stage1_brier": float(x["universal_stage1_brier"].mean()),
            "fallback_rows": int(x["authority_fallback_to_family_prior"].sum()),
            "low_meta_support_rows": int(x["meta_low_support"].sum()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict:
    vals = num(p["mean_delta_brier"]).dropna().to_numpy(float)
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


def support_gates(g: pd.DataFrame, players: pd.DataFrame, family: str, season: int) -> dict:
    req = SUPPORT[family][season]
    return {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_id"].nunique()) >= req["players"],
        "positives": int(g["y"].sum()) >= req["positives"],
        "zeros": int((g["y"] == 0).sum()) >= req["zeros"],
        "evaluable_players": int(len(players)) >= req["evaluable_players"],
        "teams": int(g["team"].nunique()) >= req["teams"],
    }


def score_slice(
    z: pd.DataFrame, family: str, season: int
) -> tuple[dict, pd.DataFrame]:
    g = z.loc[z["family"].eq(family) & z["season"].eq(season)].copy()
    p = equal_player_frame(g)
    support = support_gates(g, p, family, season)
    vals = (
        num(p["mean_delta_brier"]).dropna()
        if len(p) else pd.Series(dtype=float)
    )
    low = g.loc[g["meta_low_support"]]
    changed = g.loc[g["authority_fallback_to_family_prior"]]

    rec = {
        "family": family,
        "season": season,
        "rows": int(len(g)),
        "players": int(g["player_id"].nunique()),
        "teams": int(g["team"].nunique()),
        "positives": int(g["y"].sum()),
        "zeros": int((g["y"] == 0).sum()),
        "support_gates": support,
        "mean_delta_brier": float(g["delta_personalized_brier"].mean()),
        "mean_delta_logloss": float(g["delta_personalized_logloss"].mean()),
        "observed_participation_rate": float(g["y"].mean()),
        "mean_universal_probability": float(g["p_candidate"].mean()),
        "mean_personalized_probability": float(g["p_personalized"].mean()),
        "authority_fallback_rows": int(len(changed)),
        "authority_fallback_players": int(changed["player_id"].nunique()),
        "low_meta_support_rows": int(len(low)),
        "low_meta_support_mean_delta_brier": (
            float(low["delta_personalized_brier"].mean()) if len(low) else math.nan
        ),
        "evaluable_players_ge5": int(len(p)),
        "equal_player_mean_improvement": float(vals.mean()) if len(vals) else math.nan,
        "equal_player_median_improvement": float(vals.median()) if len(vals) else math.nan,
        "equal_player_improved_share": float((vals > TOL).mean()) if len(vals) else math.nan,
        "equal_player_tied_share": float((vals.abs() <= TOL).mean()) if len(vals) else math.nan,
        "equal_player_worsened_share": float((vals < -TOL).mean()) if len(vals) else math.nan,
        "equal_player_p10_improvement": float(vals.quantile(0.10)) if len(vals) else math.nan,
        "equal_player_p25_improvement": float(vals.quantile(0.25)) if len(vals) else math.nan,
        "equal_player_p50_improvement": float(vals.quantile(0.50)) if len(vals) else math.nan,
        "equal_player_p75_improvement": float(vals.quantile(0.75)) if len(vals) else math.nan,
        "equal_player_p90_improvement": float(vals.quantile(0.90)) if len(vals) else math.nan,
        "worst_named_player_mean_improvement": float(vals.min()) if len(vals) else math.nan,
        "universal_stage1_equal_player_brier": (
            float(p["universal_stage1_brier"].mean()) if len(p) else math.nan
        ),
    }
    fi = FAMILIES.index(family)
    rec["player_cluster_bootstrap"] = cluster_bootstrap(
        g, "delta_personalized_brier", "player_id",
        BOOT_SEED + season + fi * 100,
    )
    rec["game_cluster_bootstrap"] = cluster_bootstrap(
        g, "delta_personalized_brier", "game_key",
        BOOT_SEED + 10000 + season + fi * 100,
    )
    rec["equal_player_bootstrap"] = equal_player_bootstrap(
        p, BOOT_SEED + 20000 + season + fi * 100
    )
    return rec, p


def pooled_score(z: pd.DataFrame, family: str) -> dict:
    g = z.loc[z["family"].eq(family)].copy()
    # Match original Stage-1 pooled contract: season-qualified clusters.
    g["pooled_player_cluster"] = (
        g["season"].astype(str) + ":" + g["player_id"].astype(str)
    )
    g["pooled_game_cluster"] = (
        g["season"].astype(str) + ":" + g["game_key"].astype(str)
    )
    fi = FAMILIES.index(family)
    return {
        "rows": int(len(g)),
        "mean_delta_brier": float(g["delta_personalized_brier"].mean()),
        "mean_delta_logloss": float(g["delta_personalized_logloss"].mean()),
        "player_cluster_bootstrap": cluster_bootstrap(
            g, "delta_personalized_brier", "pooled_player_cluster",
            BOOT_SEED + 30000 + fi * 100,
        ),
        "game_cluster_bootstrap": cluster_bootstrap(
            g, "delta_personalized_brier", "pooled_game_cluster",
            BOOT_SEED + 40000 + fi * 100,
        ),
    }


def qualify(dev: dict, confirm: dict, pooled: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(confirm["universal_stage1_equal_player_brier"])
        if np.isfinite(confirm["universal_stage1_equal_player_brier"])
        else math.nan
    )
    gates = {
        "development_support": bool(all(dev["support_gates"].values())),
        "development_brier_improves": bool(dev["mean_delta_brier"] > 0),
        "development_logloss_nonworse": bool(dev["mean_delta_logloss"] >= 0),
        "confirmation_support": bool(all(confirm["support_gates"].values())),
        "confirmation_brier_improves": bool(confirm["mean_delta_brier"] > 0),
        "confirmation_logloss_nonworse": bool(confirm["mean_delta_logloss"] >= 0),
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
        "confirmation_improved_share_gt_worsened_share": bool(
            confirm["equal_player_improved_share"] > confirm["equal_player_worsened_share"]
        ),
        "confirmation_low_meta_support_nonworse": bool(
            confirm["low_meta_support_mean_delta_brier"] >= 0
        ),
        "confirmation_tail_harm_guard": bool(
            np.isfinite(tail_floor)
            and confirm["equal_player_p10_improvement"] >= tail_floor
        ),
        "pooled_brier_improves": bool(pooled["mean_delta_brier"] > 0),
        "pooled_player_cluster_ci_low_gt_zero": bool(
            pooled["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "pooled_game_cluster_ci_low_gt_zero": bool(
            pooled["game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
    }
    gates["tail_floor"] = tail_floor
    return gates, all(v for k, v in gates.items() if k != "tail_floor")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rows", required=True, type=Path)
    ap.add_argument("--signal-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parent = read_parent(args.parent_rows)
        signal = read_signal(args.signal_rows)
        z = prepare(parent, signal)
        if z.duplicated(["season", "week", "team", "family", "player_id"]).any():
            raise RuntimeError("duplicate merged identity")

        metrics: dict[str, dict] = {}
        player_parts = []
        gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}

        for family in FAMILIES:
            dev, p24 = score_slice(z, family, 2024)
            confirm, p25 = score_slice(z, family, 2025)
            pooled = pooled_score(z, family)
            metrics[family] = {"2024": dev, "2025": confirm, "pooled": pooled}
            p24["family"] = family
            p24["season"] = 2024
            p25["family"] = family
            p25["season"] = 2025
            player_parts.extend([p24, p25])
            fgates, ok = qualify(dev, confirm, pooled)
            gates[family] = fgates
            dispositions[family] = (
                f"{family}_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED"
                if ok else
                f"{family}_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED"
            )

        qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        overall = (
            "PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_ALL"
            if qualified == len(FAMILIES)
            else "PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL"
            if qualified
            else "PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED_ALL"
        )

        z[
            [
                "season", "week", "game_key", "team", "opponent", "family",
                "player_id", "player_name_key", "y", "p_baseline", "p_candidate",
                "player_prior_mechanism_rows", "hierarchical_expected_benefit",
                "meta_low_support", "authority_fallback_to_family_prior",
                "p_personalized", "universal_stage1_brier",
                "personalized_stage1_brier", "delta_personalized_brier",
                "universal_stage1_logloss", "personalized_stage1_logloss",
                "delta_personalized_logloss",
            ]
        ].to_csv(
            args.out_dir / "rb_te_stage1_player_specific_authority_bridge_rows.csv",
            index=False,
        )
        pd.concat(player_parts, ignore_index=True).to_csv(
            args.out_dir / "rb_te_stage1_player_specific_authority_bridge_player_diagnostics.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "signal_run": SIGNAL_RUN,
            "signal_artifact": SIGNAL_ARTIFACT,
            "signal_digest": SIGNAL_DIGEST,
            "families": list(FAMILIES),
            "minimum_mechanism_history": MIN_MECHANISM_HISTORY,
            "negative_or_zero_signal_action": "REVERT_TO_FAMILY_P_BASELINE",
            "positive_signal_action": "PRESERVE_STAGE1_P_CANDIDATE",
            "low_meta_support_action": "PRESERVE_STAGE1_P_CANDIDATE",
            "metrics": metrics,
            "qualification_gates": gates,
            "family_dispositions": dispositions,
            "disposition": overall,
            "production_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_INTEGRITY_FAILURE",
            "error": str(exc),
            "production_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
        }
        (args.out_dir / "rb_te_stage1_player_specific_authority_bridge_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "rb_te_stage1_player_specific_authority_bridge_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
