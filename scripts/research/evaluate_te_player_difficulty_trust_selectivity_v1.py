#!/usr/bin/env python3
"""Frozen TE player trust/selectivity V1 evaluator.

Research-only reliability layer. It labels trust in the existing canonical
TE-R5P distribution without changing any projection or distribution value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_V1"
QUALIFIED = "TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED"
CLOSED = "NO_ACTIONABLE_TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY"
PARENT_DISPOSITION = "NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS"
BOOT_REPS = 5000
POOL_PLAYER_SEED = 2026101511
POOL_GAME_SEED = 2026101512
Y25_PLAYER_SEED = 2026101513
Y25_GAME_SEED = 2026101514


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty input: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _corr(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(method="average", pct=True)


def _prepare(rows_path: Path, parent_result_path: Path) -> pd.DataFrame:
    parent = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if parent.get("disposition") != PARENT_DISPOSITION:
        raise RuntimeError(
            f"parent disposition drift {parent.get('disposition')} != {PARENT_DISPOSITION}"
        )
    if int(parent.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("parent sportsbook leakage")
    if int(parent.get("oddsapi_calls", 1)) != 0:
        raise RuntimeError("parent OddsAPI leakage")
    if int(parent.get("outcomes_2026_read", 1)) != 0:
        raise RuntimeError("parent 2026 outcome leakage")
    if bool(parent.get("production_changed", True)):
        raise RuntimeError("parent production mutation")

    x = _read(rows_path)
    need = {
        "season", "week", "team", "player", "player_clean_key",
        "player_identity_key", "game_id", "player_mae", "week_pool_mae",
        "baseline_mean", "baseline_sd", "baseline_crps", "actual_rec_yards",
    }
    miss = sorted(need - set(x.columns))
    if miss:
        raise RuntimeError(f"trust parent rows missing columns: {miss}")

    for c in (
        "season", "week", "player_mae", "week_pool_mae", "baseline_mean",
        "baseline_sd", "baseline_crps", "actual_rec_yards",
    ):
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.loc[x["season"].isin([2024, 2025])].copy()
    x["abs_mean_error"] = (x["actual_rec_yards"] - x["baseline_mean"]).abs()
    x["standardized_crps"] = x["baseline_crps"] / x["baseline_sd"]
    x["standardized_abs_error"] = x["abs_mean_error"] / x["baseline_sd"]
    x["difficulty_ratio_vs_pool"] = x["player_mae"] / x["week_pool_mae"]

    good = (
        x["season"].notna()
        & x["week"].notna()
        & x["player_mae"].gt(0)
        & x["week_pool_mae"].gt(0)
        & x["baseline_sd"].gt(0)
        & x["baseline_crps"].ge(0)
        & x["actual_rec_yards"].ge(0)
        & np.isfinite(x["standardized_crps"])
        & np.isfinite(x["standardized_abs_error"])
        & x["player_identity_key"].fillna("").astype(str).str.strip().ne("")
        & x["game_id"].fillna("").astype(str).str.strip().ne("")
    )
    x = x.loc[good].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)

    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        sample = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate trust identities: {sample}")

    g = x.groupby(["season", "week"], sort=False)
    x["trust_fragility_score"] = g["player_mae"].transform(_rank_pct)
    x["crps_badness_rank"] = g["standardized_crps"].transform(_rank_pct)
    x["mean_error_badness_rank"] = g["standardized_abs_error"].transform(_rank_pct)

    x["trust_bucket"] = "MIDDLE"
    x.loc[x["trust_fragility_score"].le(0.25), "trust_bucket"] = "HIGH_TRUST"
    x.loc[x["trust_fragility_score"].ge(0.75), "trust_bucket"] = "LOW_TRUST"

    if not x["trust_fragility_score"].between(0, 1).all():
        raise RuntimeError("trust score range drift")
    return x.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def _metrics(q: pd.DataFrame) -> dict:
    if q.empty:
        return {"rows": 0, "players": 0, "games": 0}
    hi = q.loc[q["trust_bucket"].eq("HIGH_TRUST")]
    lo = q.loc[q["trust_bucket"].eq("LOW_TRUST")]
    return {
        "rows": int(len(q)),
        "players": int(q["player_identity_key"].nunique()),
        "games": int(q["game_id"].nunique()),
        "weeks": int(q[["season", "week"]].drop_duplicates().shape[0]),
        "primary_rank_association": _corr(
            q["trust_fragility_score"], q["crps_badness_rank"]
        ),
        "secondary_rank_association": _corr(
            q["trust_fragility_score"], q["mean_error_badness_rank"]
        ),
        "overall_standardized_crps": float(q["standardized_crps"].mean()),
        "high_trust_standardized_crps": float(hi["standardized_crps"].mean()) if len(hi) else np.nan,
        "low_trust_standardized_crps": float(lo["standardized_crps"].mean()) if len(lo) else np.nan,
        "overall_standardized_abs_error": float(q["standardized_abs_error"].mean()),
        "high_trust_standardized_abs_error": float(hi["standardized_abs_error"].mean()) if len(hi) else np.nan,
        "low_trust_standardized_abs_error": float(lo["standardized_abs_error"].mean()) if len(lo) else np.nan,
        "overall_raw_crps": float(q["baseline_crps"].mean()),
        "high_trust_raw_crps": float(hi["baseline_crps"].mean()) if len(hi) else np.nan,
        "low_trust_raw_crps": float(lo["baseline_crps"].mean()) if len(lo) else np.nan,
        "overall_abs_mean_error": float(q["abs_mean_error"].mean()),
        "high_trust_abs_mean_error": float(hi["abs_mean_error"].mean()) if len(hi) else np.nan,
        "low_trust_abs_mean_error": float(lo["abs_mean_error"].mean()) if len(lo) else np.nan,
        "high_trust_rows": int(len(hi)),
        "low_trust_rows": int(len(lo)),
        "median_difficulty_ratio_vs_pool": float(q["difficulty_ratio_vs_pool"].median()),
        "high_trust_median_ratio_vs_pool": float(hi["difficulty_ratio_vs_pool"].median()) if len(hi) else np.nan,
        "low_trust_median_ratio_vs_pool": float(lo["difficulty_ratio_vs_pool"].median()) if len(lo) else np.nan,
    }


def _cluster_boot(
    q: pd.DataFrame,
    cluster_col: str,
    *,
    seed: int,
) -> dict:
    cols = ["trust_fragility_score", "crps_badness_rank", cluster_col]
    z = q[cols].dropna().copy()
    groups = {str(k): g for k, g in z.groupby(cluster_col, sort=True)}
    keys = np.asarray(sorted(groups), dtype=object)
    observed = _corr(z["trust_fragility_score"], z["crps_badness_rank"])
    if len(keys) < 2 or not np.isfinite(observed):
        return {
            "reps": 0,
            "observed": observed,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_positive": np.nan,
            "clusters": int(len(keys)),
        }
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        parts = [groups[str(k)] for k in sampled]
        b = pd.concat(parts, ignore_index=True)
        vals[i] = _corr(b["trust_fragility_score"], b["crps_badness_rank"])
    vals = vals[np.isfinite(vals)]
    return {
        "reps": int(len(vals)),
        "observed": observed,
        "ci_low": float(np.quantile(vals, .025, method="linear")),
        "ci_high": float(np.quantile(vals, .975, method="linear")),
        "p_positive": float(np.mean(vals > 0)),
        "clusters": int(len(keys)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--parent-result", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    x = _prepare(a.rows, a.parent_result)
    y24 = x.loc[x["season"].eq(2024)].copy()
    y25 = x.loc[x["season"].eq(2025)].copy()

    m24 = _metrics(y24)
    m25 = _metrics(y25)
    mp = _metrics(x)

    pooled_player = _cluster_boot(x, "player_identity_key", seed=POOL_PLAYER_SEED)
    pooled_game = _cluster_boot(x, "game_id", seed=POOL_GAME_SEED)
    y25_player = _cluster_boot(y25, "player_identity_key", seed=Y25_PLAYER_SEED)
    y25_game = _cluster_boot(y25, "game_id", seed=Y25_GAME_SEED)

    support = (
        m24["rows"] >= 600 and m24["players"] >= 60
        and m25["rows"] >= 600 and m25["players"] >= 60
        and mp["rows"] >= 1300 and mp["players"] >= 90
        and m24["high_trust_rows"] >= 120 and m24["low_trust_rows"] >= 120
        and m25["high_trust_rows"] >= 120 and m25["low_trust_rows"] >= 120
    )

    gates = {
        "support": bool(support),
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "outcomes_2026_read_zero": True,
        "production_unchanged": True,
        "projection_distribution_values_changed_zero": True,
        "2024_primary_rank_association_positive": bool(m24["primary_rank_association"] > 0),
        "2025_primary_rank_association_positive": bool(m25["primary_rank_association"] > 0),
        "pooled_player_cluster_ci_low_positive": bool(pooled_player["ci_low"] > 0),
        "pooled_game_cluster_ci_low_positive": bool(pooled_game["ci_low"] > 0),
        "2025_player_cluster_ci_low_positive": bool(y25_player["ci_low"] > 0),
        "2025_game_cluster_ci_low_positive": bool(y25_game["ci_low"] > 0),
        "2024_low_trust_crps_worse_than_high_trust": bool(
            m24["low_trust_standardized_crps"] > m24["high_trust_standardized_crps"]
        ),
        "2025_low_trust_crps_worse_than_high_trust": bool(
            m25["low_trust_standardized_crps"] > m25["high_trust_standardized_crps"]
        ),
        "2024_high_trust_crps_better_than_overall": bool(
            m24["high_trust_standardized_crps"] < m24["overall_standardized_crps"]
        ),
        "2025_high_trust_crps_better_than_overall": bool(
            m25["high_trust_standardized_crps"] < m25["overall_standardized_crps"]
        ),
        "2024_low_trust_crps_worse_than_overall": bool(
            m24["low_trust_standardized_crps"] > m24["overall_standardized_crps"]
        ),
        "2025_low_trust_crps_worse_than_overall": bool(
            m25["low_trust_standardized_crps"] > m25["overall_standardized_crps"]
        ),
        "2024_secondary_rank_association_positive": bool(m24["secondary_rank_association"] > 0),
        "2025_secondary_rank_association_positive": bool(m25["secondary_rank_association"] > 0),
        "2024_low_trust_abs_error_worse_than_high_trust": bool(
            m24["low_trust_standardized_abs_error"] > m24["high_trust_standardized_abs_error"]
        ),
        "2025_low_trust_abs_error_worse_than_high_trust": bool(
            m25["low_trust_standardized_abs_error"] > m25["high_trust_standardized_abs_error"]
        ),
    }

    disposition = QUALIFIED if all(gates.values()) else CLOSED
    result = {
        "version": VERSION,
        "disposition": disposition,
        "source_parent_disposition": PARENT_DISPOSITION,
        "season_metrics": {"2024": m24, "2025": m25},
        "pooled_metrics": mp,
        "pooled_player_cluster": pooled_player,
        "pooled_game_cluster": pooled_game,
        "2025_player_cluster": y25_player,
        "2025_game_cluster": y25_game,
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "trust_score_formula": "within-season-week average percentile rank of strictly-prior player_mae",
        "primary_outcome": "baseline_crps / baseline_sd",
        "secondary_outcome": "abs(actual_rec_yards - baseline_mean) / baseline_sd",
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "projection_distribution_values_changed": 0,
        "production_changed": False,
        "models_fit": 0,
        "bootstrap_reps": BOOT_REPS,
    }

    x.to_csv(a.out_dir / "te_player_difficulty_trust_selectivity_rows.csv", index=False)
    (a.out_dir / "te_player_difficulty_trust_selectivity_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
