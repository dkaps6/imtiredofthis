#!/usr/bin/env python3
"""PRISM pass-catcher x QB-regime target-depth signal V1.

Frozen one-candidate target-depth scoring test. Target entitlement and all
market-specific output translation are intentionally out of scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.audit_prism_pass_catcher_qb_regime_target_depth_state_v1_source import (
    load_target_events,
)

VERSION = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1"
SEASONS = (2024, 2025)
FAMILIES = ("WR", "TE")
BOOTSTRAPS = 5000
SEED = 20261017


def num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def bools(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)
    return x.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def target_outcomes() -> pd.DataFrame:
    events, _ = load_target_events()
    x = events.loc[
        events["season"].isin(SEASONS) & events["air_yards"].notna()
    ].copy()
    if x.empty:
        raise RuntimeError("zero target-depth outcome events")
    x["air_yards"] = num(x["air_yards"])
    out = (
        x.groupby(["season", "week", "team", "receiver_id"], as_index=False)
        .agg(
            actual_target_depth=("air_yards", "mean"),
            actual_finite_air_targets=("air_yards", "count"),
        )
    )
    return out.rename(columns={"receiver_id": "player_id"})


def bootstrap_cluster_ci(
    x: pd.DataFrame,
    cluster_col: str,
    value_col: str = "error_delta",
    reps: int = BOOTSTRAPS,
    seed: int = SEED,
) -> tuple[float, float]:
    groups = {
        str(k): num(g[value_col]).dropna().to_numpy(float)
        for k, g in x.groupby(cluster_col, sort=True)
    }
    keys = [k for k, v in groups.items() if len(v)]
    if len(keys) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals[i] = float(np.concatenate([groups[k] for k in sampled]).mean())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def metrics(x: pd.DataFrame) -> dict[str, float | int]:
    b = num(x["baseline_abs_error"])
    c = num(x["candidate_abs_error"])
    actual = num(x["actual_target_depth"])
    bp = num(x["baseline_target_depth"])
    cp = num(x["candidate_target_depth"])
    player_delta = x.groupby("player_id", sort=True)["error_delta"].mean()
    return {
        "rows": int(len(x)),
        "players": int(x["player_id"].nunique()),
        "teamweeks": int(x["teamweek"].nunique()),
        "baseline_mae": float(b.mean()),
        "candidate_mae": float(c.mean()),
        "mae_delta": float(c.mean() - b.mean()),
        "baseline_rmse": float(np.sqrt(np.mean(np.square(bp - actual)))),
        "candidate_rmse": float(np.sqrt(np.mean(np.square(cp - actual)))),
        "baseline_p90_ae": float(b.quantile(0.90)),
        "candidate_p90_ae": float(c.quantile(0.90)),
        "candidate_closer_rate": float((c < b).mean()),
        "ties_rate": float((c == b).mean()),
        "baseline_bias": float((bp - actual).mean()),
        "candidate_bias": float((cp - actual).mean()),
        "median_equal_player_mae_delta": float(player_delta.median()),
        "mean_actual_finite_air_targets": float(num(x["actual_finite_air_targets"]).mean()),
    }


def build(state: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    need = {
        "season", "week", "team", "player_id", "family", "mechanism_source_ready",
        "chronology_violation", "prior8_mean_air_yards", "same_qb_mean_air_yards",
        "prior8_receiver_games", "prior8_finite_air_targets", "same_qb_prior_games",
        "same_qb_finite_air_targets", "selected_qb_id",
    }
    missing = need - set(state.columns)
    if missing:
        raise RuntimeError(f"source state missing {sorted(missing)}")

    s = state.copy()
    s["season"] = num(s["season"]).astype("Int64")
    s["week"] = num(s["week"]).astype("Int64")
    s["family"] = s["family"].astype(str).str.upper().str.strip()
    s["mechanism_source_ready"] = bools(s["mechanism_source_ready"])
    s["chronology_violation"] = bools(s["chronology_violation"])
    s = s.loc[s["season"].isin(SEASONS) & s["family"].isin(FAMILIES)].copy()

    if s.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate source state identity")
    if bool(s["chronology_violation"].any()):
        raise RuntimeError("source state chronology violation")

    s["baseline_target_depth"] = num(s["prior8_mean_air_yards"])
    s["candidate_target_depth"] = num(s["same_qb_mean_air_yards"])

    eligible = s.loc[
        s["mechanism_source_ready"]
        & s["baseline_target_depth"].notna()
        & s["candidate_target_depth"].notna()
    ].copy()
    if eligible.empty:
        raise RuntimeError("zero frozen-mechanism eligible rows")

    outcome = target_outcomes()
    if outcome.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate target-depth outcome identity")

    x = eligible.merge(
        outcome,
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="one_to_one",
    )
    x["actual_target_depth"] = num(x["actual_target_depth"])
    x["actual_finite_air_targets"] = num(x["actual_finite_air_targets"])
    scored = x.loc[
        x["actual_target_depth"].notna() & x["actual_finite_air_targets"].gt(0)
    ].copy()
    if scored.empty:
        raise RuntimeError("zero scoreable target-depth player-games")

    scored["teamweek"] = (
        scored["season"].astype(str)
        + "_"
        + scored["week"].astype(str)
        + "_"
        + scored["team"].astype(str)
    )
    scored["baseline_abs_error"] = (
        scored["baseline_target_depth"] - scored["actual_target_depth"]
    ).abs()
    scored["candidate_abs_error"] = (
        scored["candidate_target_depth"] - scored["actual_target_depth"]
    ).abs()
    scored["error_delta"] = (
        scored["candidate_abs_error"] - scored["baseline_abs_error"]
    )

    metric_rows = []
    family_results: dict[str, dict[str, Any]] = {f: {} for f in FAMILIES}
    for family in FAMILIES:
        for season in SEASONS:
            z = scored.loc[
                scored["family"].eq(family) & scored["season"].eq(season)
            ].copy()
            m = metrics(z) if len(z) else {"rows": 0, "players": 0}
            if season == 2025 and len(z):
                p_lo, p_hi = bootstrap_cluster_ci(
                    z, "player_id", seed=SEED + (11 if family == "WR" else 12)
                )
                t_lo, t_hi = bootstrap_cluster_ci(
                    z, "teamweek", seed=SEED + (21 if family == "WR" else 22)
                )
                m["player_bootstrap_ci_low"] = p_lo
                m["player_bootstrap_ci_high"] = p_hi
                m["teamweek_bootstrap_ci_low"] = t_lo
                m["teamweek_bootstrap_ci_high"] = t_hi
            family_results[family][str(season)] = m
            metric_rows.append({"family": family, "season": season, **m})

    gates: dict[str, dict[str, bool]] = {}
    qualified: dict[str, bool] = {}
    for family in FAMILIES:
        d = family_results[family]["2024"]
        c = family_results[family]["2025"]
        min_rows = 750 if family == "WR" else 350
        min_players = 100 if family == "WR" else 50
        g = {
            "dev_support": int(d.get("rows", 0)) >= min_rows
            and int(d.get("players", 0)) >= min_players,
            "dev_mae_improves": d.get("candidate_mae", np.inf)
            < d.get("baseline_mae", -np.inf),
            "dev_rmse_nonworse": d.get("candidate_rmse", np.inf)
            <= d.get("baseline_rmse", -np.inf),
            "dev_p90_nonworse": d.get("candidate_p90_ae", np.inf)
            <= d.get("baseline_p90_ae", -np.inf),
            "dev_closer_gt_50": d.get("candidate_closer_rate", 0.0) > 0.50,
            "dev_median_equal_player_delta_lt_0": d.get(
                "median_equal_player_mae_delta", np.inf
            )
            < 0.0,
            "confirm_support": int(c.get("rows", 0)) >= min_rows
            and int(c.get("players", 0)) >= min_players,
            "confirm_mae_improves": c.get("candidate_mae", np.inf)
            < c.get("baseline_mae", -np.inf),
            "confirm_rmse_nonworse": c.get("candidate_rmse", np.inf)
            <= c.get("baseline_rmse", -np.inf),
            "confirm_p90_nonworse": c.get("candidate_p90_ae", np.inf)
            <= c.get("baseline_p90_ae", -np.inf),
            "confirm_closer_gt_50": c.get("candidate_closer_rate", 0.0) > 0.50,
            "confirm_median_equal_player_delta_lt_0": c.get(
                "median_equal_player_mae_delta", np.inf
            )
            < 0.0,
            "confirm_player_ci_high_lt_0": c.get(
                "player_bootstrap_ci_high", np.inf
            )
            < 0.0,
            "confirm_teamweek_ci_high_lt_0": c.get(
                "teamweek_bootstrap_ci_high", np.inf
            )
            < 0.0,
            "chronology_zero": int(
                scored.loc[scored["family"].eq(family), "chronology_violation"].sum()
            )
            == 0,
        }
        gates[family] = g
        qualified[family] = bool(all(g.values()))

    if all(qualified.values()):
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_QUALIFIED_ALL"
    elif any(qualified.values()):
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_QUALIFIED_PARTIAL"
    else:
        disposition = "PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_CLOSED"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "candidate_variants": 1,
        "development_season": 2024,
        "confirmation_season": 2025,
        "families": list(FAMILIES),
        "family_qualified": qualified,
        "gates": gates,
        "metrics": metric_rows,
        "integrity": {
            "chronology_violations": int(scored["chronology_violation"].sum()),
            "target_entitlement_loaded": False,
            "target_entitlement_changed": False,
            "receiving_yards_scored": False,
            "receptions_scored": False,
            "touchdowns_scored": False,
            "market_translation_scored": False,
            "coefficients_fit": 0,
            "thresholds_searched": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        },
        "boundary": (
            "A pass qualifies only QB-conditioned target-depth state. Any receiving-"
            "yard/reception/TD translation requires a separately frozen test."
        ),
    }
    return scored, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows, result = build(read_csv(a.state, "QB-regime target-depth source state"))
    rows.to_csv(
        a.out_dir / "qb_regime_target_depth_signal_rows.csv",
        index=False,
        float_format="%.12f",
    )
    pd.DataFrame(result["metrics"]).to_csv(
        a.out_dir / "qb_regime_target_depth_signal_metrics.csv",
        index=False,
        float_format="%.12f",
    )
    (a.out_dir / "qb_regime_target_depth_signal_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
