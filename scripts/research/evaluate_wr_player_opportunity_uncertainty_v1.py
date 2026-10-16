#!/usr/bin/env python3
"""WR Player Opportunity Uncertainty V1.

Research-only player-first uncertainty diagnostic. Uses only the already-frozen
strictly-prior WR opportunity-difficulty feature from the canonical
WR_PLAYER_PERSISTENCE_MIXED_MECHANISM artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

VERSION = "WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1"
TARGET_SEASONS = (2023, 2024)
EXPECTED_ROWS = 3224
EXPECTED_PLAYERS = 211
MIN_SEASON_ROWS = 700
MIN_SEASON_PLAYERS = 100
MIN_POOLED_ROWS = 2000
MIN_POOLED_PLAYERS = 150
BOOT_REPS = 5000
BOOT_SEED = 20261014
EPS = 1e-6
SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
Z80 = float(norm.ppf(0.90))
Z90 = float(norm.ppf(0.95))

REQUIRED = {
    "season",
    "week",
    "team",
    "player_clean_key",
    "wr_rank",
    "prior_games",
    "prior8_abs_opportunity_error_mean",
    "current_opportunity_error",
}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def read_source(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    missing = REQUIRED - set(x.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")

    for c in [
        "season",
        "week",
        "wr_rank",
        "prior_games",
        "prior8_abs_opportunity_error_mean",
        "current_opportunity_error",
    ]:
        x[c] = num(x[c])
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x["team"] = x["team"].fillna("").astype(str).str.strip()

    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"parent row drift: {len(x)} != {EXPECTED_ROWS}")
    if int(x["player_clean_key"].nunique()) != EXPECTED_PLAYERS:
        raise RuntimeError(
            f"parent player drift: {int(x['player_clean_key'].nunique())} != {EXPECTED_PLAYERS}"
        )
    if x["player_clean_key"].eq("").any():
        raise RuntimeError("blank player identity")
    if x.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("duplicate player-week identity")

    bad_season = ~x["season"].isin(TARGET_SEASONS)
    if bad_season.any():
        raise RuntimeError(
            f"unexpected seasons: {sorted(x.loc[bad_season, 'season'].dropna().astype(int).unique())}"
        )

    x = x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["week"].notna()
        & x["prior_games"].ge(4)
        & x["prior8_abs_opportunity_error_mean"].notna()
        & x["prior8_abs_opportunity_error_mean"].ge(0)
        & x["current_opportunity_error"].notna()
    ].copy()

    if len(x) != EXPECTED_ROWS:
        raise RuntimeError(f"eligible source drift: {len(x)} != {EXPECTED_ROWS}")

    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    return x.sort_values(
        ["season", "week", "team", "player_clean_key"]
    ).reset_index(drop=True)


def normal_crps_zero(y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), EPS)
    z = y / sigma
    return sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def prepare(x: pd.DataFrame) -> pd.DataFrame:
    q = x.copy()
    q["player_mae"] = q["prior8_abs_opportunity_error_mean"].astype(float)

    pool = (
        q.groupby(["season", "week"], as_index=False)["player_mae"]
        .median()
        .rename(columns={"player_mae": "week_pool_mae"})
    )
    q = q.merge(pool, on=["season", "week"], how="left", validate="many_to_one")
    if q["week_pool_mae"].isna().any():
        raise RuntimeError("missing week pooled baseline")

    q["candidate_sigma"] = np.maximum(EPS, q["player_mae"] * SQRT_PI_OVER_2)
    q["baseline_sigma"] = np.maximum(EPS, q["week_pool_mae"] * SQRT_PI_OVER_2)

    y = q["current_opportunity_error"].to_numpy(float)
    q["candidate_crps"] = normal_crps_zero(y, q["candidate_sigma"].to_numpy(float))
    q["baseline_crps"] = normal_crps_zero(y, q["baseline_sigma"].to_numpy(float))
    q["crps_gain"] = q["baseline_crps"] - q["candidate_crps"]

    ay = np.abs(y)
    q["candidate_cover80"] = ay <= Z80 * q["candidate_sigma"].to_numpy(float)
    q["baseline_cover80"] = ay <= Z80 * q["baseline_sigma"].to_numpy(float)
    q["candidate_cover90"] = ay <= Z90 * q["candidate_sigma"].to_numpy(float)
    q["baseline_cover90"] = ay <= Z90 * q["baseline_sigma"].to_numpy(float)

    q75 = float(np.quantile(q["player_mae"].to_numpy(float), 0.75, method="linear"))
    q25 = float(np.quantile(q["player_mae"].to_numpy(float), 0.25, method="linear"))
    q["difficulty_q4"] = q["player_mae"].ge(q75)
    q["difficulty_q1"] = q["player_mae"].le(q25)
    q["role"] = np.where(q["wr_rank"].eq(1), "WR1", "WR2+")
    q.attrs["difficulty_q75"] = q75
    q.attrs["difficulty_q25"] = q25
    return q


def coverage_gap(x: pd.Series, nominal: float) -> float:
    return abs(float(x.mean()) - nominal)


def metrics(q: pd.DataFrame) -> dict:
    if q.empty:
        return {
            "rows": 0,
            "players": 0,
            "baseline_crps": None,
            "candidate_crps": None,
            "crps_gain": None,
        }
    return {
        "rows": int(len(q)),
        "players": int(q["player_clean_key"].nunique()),
        "baseline_crps": float(q["baseline_crps"].mean()),
        "candidate_crps": float(q["candidate_crps"].mean()),
        "crps_gain": float(q["crps_gain"].mean()),
        "baseline_80_coverage": float(q["baseline_cover80"].mean()),
        "candidate_80_coverage": float(q["candidate_cover80"].mean()),
        "baseline_80_gap": coverage_gap(q["baseline_cover80"], 0.80),
        "candidate_80_gap": coverage_gap(q["candidate_cover80"], 0.80),
        "baseline_90_coverage": float(q["baseline_cover90"].mean()),
        "candidate_90_coverage": float(q["candidate_cover90"].mean()),
        "baseline_90_gap": coverage_gap(q["baseline_cover90"], 0.90),
        "candidate_90_gap": coverage_gap(q["candidate_cover90"], 0.90),
        "mean_player_mae": float(q["player_mae"].mean()),
        "median_player_mae": float(q["player_mae"].median()),
        "mean_week_pool_mae": float(q["week_pool_mae"].mean()),
    }


def bootstrap_player_crps_gain(q: pd.DataFrame) -> dict:
    players = sorted(q["player_clean_key"].astype(str).unique())
    groups = {
        p: q.loc[q["player_clean_key"].astype(str).eq(p), "crps_gain"].to_numpy(float)
        for p in players
    }
    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sampled = rng.choice(players, size=len(players), replace=True)
        vals[i] = float(np.concatenate([groups[p] for p in sampled]).mean())
    return {
        "reps": BOOT_REPS,
        "seed": BOOT_SEED,
        "observed_gain": float(q["crps_gain"].mean()),
        "p_positive": float((vals > 0).mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = read_source(args.rows)
    q = prepare(source)

    season_metrics = {
        str(s): metrics(q.loc[q["season"].eq(s)].copy()) for s in TARGET_SEASONS
    }
    pooled = metrics(q)
    q4 = metrics(q.loc[q["difficulty_q4"]].copy())
    q1 = metrics(q.loc[q["difficulty_q1"]].copy())
    wr1 = metrics(q.loc[q["role"].eq("WR1")].copy())
    wr2p = metrics(q.loc[q["role"].eq("WR2+")].copy())
    boot = bootstrap_player_crps_gain(q)

    support = (
        all(
            season_metrics[str(s)]["rows"] >= MIN_SEASON_ROWS
            and season_metrics[str(s)]["players"] >= MIN_SEASON_PLAYERS
            for s in TARGET_SEASONS
        )
        and pooled["rows"] >= MIN_POOLED_ROWS
        and pooled["players"] >= MIN_POOLED_PLAYERS
    )

    gates = {
        "support": bool(support),
        "2023_crps_improves": bool(
            season_metrics["2023"]["candidate_crps"] < season_metrics["2023"]["baseline_crps"]
        ),
        "2024_crps_improves": bool(
            season_metrics["2024"]["candidate_crps"] < season_metrics["2024"]["baseline_crps"]
        ),
        "pooled_crps_gain_positive": bool(pooled["crps_gain"] > 0),
        "player_cluster_ci_low_positive": bool(boot["ci_low"] > 0),
        "pooled_80_gap_nonworse": bool(
            pooled["candidate_80_gap"] <= pooled["baseline_80_gap"]
        ),
        "pooled_90_gap_nonworse": bool(
            pooled["candidate_90_gap"] <= pooled["baseline_90_gap"]
        ),
        "high_difficulty_q4_crps_improves": bool(
            q4["candidate_crps"] < q4["baseline_crps"]
        ),
        "source_rows_exact": bool(len(source) == EXPECTED_ROWS),
        "source_players_exact": bool(
            source["player_clean_key"].nunique() == EXPECTED_PLAYERS
        ),
    }

    integrity = (
        gates["source_rows_exact"]
        and gates["source_players_exact"]
        and np.isfinite(q["candidate_sigma"]).all()
        and np.isfinite(q["baseline_sigma"]).all()
        and (q["candidate_sigma"] > 0).all()
        and (q["baseline_sigma"] > 0).all()
    )

    scientific_keys = [
        "support",
        "2023_crps_improves",
        "2024_crps_improves",
        "pooled_crps_gain_positive",
        "player_cluster_ci_low_positive",
        "pooled_80_gap_nonworse",
        "pooled_90_gap_nonworse",
        "high_difficulty_q4_crps_improves",
    ]

    if not integrity:
        disposition = "WR_PLAYER_OPPORTUNITY_UNCERTAINTY_INTEGRITY_FAILURE"
    elif not support:
        disposition = "WR_PLAYER_OPPORTUNITY_UNCERTAINTY_SOURCE_LIMITED"
    elif all(gates[k] for k in scientific_keys):
        disposition = "WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED"
    else:
        disposition = "NO_ACTIONABLE_WR_PLAYER_OPPORTUNITY_UNCERTAINTY"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "source_run": 37634044900,
        "source_artifact": 11487359185,
        "source_artifact_digest": "sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3",
        "source_parent_disposition": "WR_PLAYER_PERSISTENCE_MIXED_MECHANISM",
        "target_seasons": list(TARGET_SEASONS),
        "rows": int(len(q)),
        "players": int(q["player_clean_key"].nunique()),
        "difficulty_q25": float(q.attrs["difficulty_q25"]),
        "difficulty_q75": float(q.attrs["difficulty_q75"]),
        "season_metrics": season_metrics,
        "pooled_metrics": pooled,
        "high_difficulty_q4": q4,
        "low_difficulty_q1": q1,
        "wr1_diagnostic": wr1,
        "wr2plus_diagnostic": wr2p,
        "player_cluster_bootstrap": boot,
        "gates": gates,
        "failed_scientific_gates": [k for k in scientific_keys if not gates[k]],
        "candidate_models_fit": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_read": 0,
        "projection_mean_changes": 0,
        "production_changed": False,
    }

    q.to_csv(args.out_dir / "wr_player_opportunity_uncertainty_rows.csv", index=False)
    (args.out_dir / "wr_player_opportunity_uncertainty_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
