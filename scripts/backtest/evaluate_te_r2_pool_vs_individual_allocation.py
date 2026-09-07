#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
COMPONENTS = ["TEAM_TE_POOL", "INDIVIDUAL_ALLOCATION"]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, got {len(hits)}")
    return hits[0]


def metric(actual: pd.Series, pred: pd.Series, thresholds: tuple[float, ...]) -> dict:
    z = pd.DataFrame({"actual": pd.to_numeric(actual, errors="coerce"), "pred": pd.to_numeric(pred, errors="coerce")}).dropna()
    err = z["pred"] - z["actual"]
    ae = err.abs()
    out = {
        "n": int(len(z)),
        "mae": float(ae.mean()) if len(z) else np.nan,
        "rmse": float(np.sqrt(np.mean(err * err))) if len(z) else np.nan,
        "bias": float(err.mean()) if len(z) else np.nan,
        "correlation": float(z["pred"].corr(z["actual"])) if len(z) > 2 else np.nan,
        "median_abs": float(ae.median()) if len(z) else np.nan,
        "p75_abs": float(ae.quantile(.75)) if len(z) else np.nan,
        "p90_abs": float(ae.quantile(.90)) if len(z) else np.nan,
    }
    for t in thresholds:
        out[f"miss{int(t)}"] = float(ae.ge(t).mean()) if len(z) else np.nan
    return out


def same_direction(component: pd.Series, residual: pd.Series) -> float:
    c = pd.to_numeric(component, errors="coerce")
    r = pd.to_numeric(residual, errors="coerce")
    mask = c.notna() & r.notna() & r.ne(0)
    if not mask.any():
        return np.nan
    return float((np.sign(c.loc[mask]) == np.sign(r.loc[mask])).mean())


def route(pool_share: float, alloc_share: float) -> str:
    if pool_share >= .55:
        return "TE_TARGET_POOL_FIRST"
    if alloc_share >= .55:
        return "TE_INDIVIDUAL_ALLOCATION_FIRST"
    return "TE_JOINT_POOL_ALLOCATION_REQUIRED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    src = pd.read_csv(one(args.joint_root, "joint_v1_paired_player_casebook.csv"), low_memory=False)
    src.columns = [str(c).strip().lower() for c in src.columns]
    req = {
        "season", "week", "event_id", "team", "player", "player_clean_key",
        "position_group", "b0_expected_targets", "targets"
    }
    missing = req - set(src.columns)
    if missing:
        raise RuntimeError(f"missing columns: {sorted(missing)}")

    x = src.loc[
        src["position_group"].eq("TE")
        & pd.to_numeric(src["season"], errors="coerce").isin(SEASONS)
    ].copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["b0_expected_targets"] = pd.to_numeric(x["b0_expected_targets"], errors="coerce")
    x["targets"] = pd.to_numeric(x["targets"], errors="coerce")
    x = x.dropna(subset=["season", "week", "event_id", "team", "player_clean_key", "b0_expected_targets", "targets"]).copy()
    x["b0_expected_targets"] = x["b0_expected_targets"].clip(lower=0)
    x["targets"] = x["targets"].clip(lower=0)

    team_keys = ["season", "week", "event_id", "team"]
    x["pred_te_pool"] = x.groupby(team_keys)["b0_expected_targets"].transform("sum")
    x["actual_te_pool"] = x.groupby(team_keys)["targets"].transform("sum")
    x["pred_te_share"] = np.where(x["pred_te_pool"] > 0, x["b0_expected_targets"] / x["pred_te_pool"], 0.0)
    x["actual_te_share"] = np.where(x["actual_te_pool"] > 0, x["targets"] / x["actual_te_pool"], 0.0)

    x["pred_targets_recon"] = x["pred_te_pool"] * x["pred_te_share"]
    x["actual_targets_recon"] = x["actual_te_pool"] * x["actual_te_share"]
    x["pred_identity_gap"] = (x["pred_targets_recon"] - x["b0_expected_targets"]).abs()
    x["actual_identity_gap"] = (x["actual_targets_recon"] - x["targets"]).abs()

    x["team_te_pool_component"] = (
        (x["actual_te_pool"] - x["pred_te_pool"])
        * (x["pred_te_share"] + x["actual_te_share"]) / 2.0
    )
    x["individual_allocation_component"] = (
        (x["actual_te_share"] - x["pred_te_share"])
        * (x["pred_te_pool"] + x["actual_te_pool"]) / 2.0
    )
    x["target_residual_actual_minus_pred"] = x["targets"] - x["b0_expected_targets"]
    x["shapley_sum"] = x["team_te_pool_component"] + x["individual_allocation_component"]
    x["shapley_gap"] = (x["shapley_sum"] - x["target_residual_actual_minus_pred"]).abs()
    x["target_abs_error"] = x["target_residual_actual_minus_pred"].abs()

    max_identity = float(max(x["pred_identity_gap"].max(), x["actual_identity_gap"].max())) if len(x) else np.inf
    max_shapley = float(x["shapley_gap"].max()) if len(x) else np.inf

    abs_pool = float(x["team_te_pool_component"].abs().sum())
    abs_alloc = float(x["individual_allocation_component"].abs().sum())
    total_abs = abs_pool + abs_alloc
    pooled_pool_share = abs_pool / total_abs if total_abs else np.nan
    pooled_alloc_share = abs_alloc / total_abs if total_abs else np.nan

    metric_rows = []
    mechanism_rows = []
    for label, g in [("POOLED", x)] + [(str(s), x.loc[x["season"].eq(s)]) for s in SEASONS]:
        metric_rows.append({"season": label, "level": "PLAYER", **metric(g["targets"], g["b0_expected_targets"], (2, 4, 6))})
        apool = float(g["team_te_pool_component"].abs().sum())
        aalloc = float(g["individual_allocation_component"].abs().sum())
        denom = apool + aalloc
        mechanism_rows.append({
            "season": label,
            "player_games": int(len(g)),
            "mean_abs_team_te_pool_component": float(g["team_te_pool_component"].abs().mean()) if len(g) else np.nan,
            "mean_abs_individual_allocation_component": float(g["individual_allocation_component"].abs().mean()) if len(g) else np.nan,
            "team_te_pool_abs_mass_share": apool / denom if denom else np.nan,
            "individual_allocation_abs_mass_share": aalloc / denom if denom else np.nan,
            "team_te_pool_same_direction_rate": same_direction(g["team_te_pool_component"], g["target_residual_actual_minus_pred"]),
            "individual_allocation_same_direction_rate": same_direction(g["individual_allocation_component"], g["target_residual_actual_minus_pred"]),
            "route": route(apool / denom, aalloc / denom) if denom else "NO_MASS",
        })

    team = x[team_keys + ["pred_te_pool", "actual_te_pool"]].drop_duplicates(team_keys).copy()
    team_metric_rows = []
    for label, g in [("POOLED", team)] + [(str(s), team.loc[team["season"].eq(s)]) for s in SEASONS]:
        m = metric(g["actual_te_pool"], g["pred_te_pool"], (3, 5))
        team_metric_rows.append({"season": label, "level": "TEAM_TE_POOL", **m})

    q75 = float(x["target_abs_error"].quantile(.75)) if len(x) else np.nan
    high = x.loc[x["target_abs_error"].ge(q75)].copy()
    high["game_dominant"] = np.where(
        high["team_te_pool_component"].abs() >= high["individual_allocation_component"].abs(),
        "TEAM_TE_POOL", "INDIVIDUAL_ALLOCATION"
    )
    high_pool_abs = float(high["team_te_pool_component"].abs().sum())
    high_alloc_abs = float(high["individual_allocation_component"].abs().sum())
    high_total = high_pool_abs + high_alloc_abs
    high_pool_share = high_pool_abs / high_total if high_total else np.nan
    high_alloc_share = high_alloc_abs / high_total if high_total else np.nan

    try:
        x["b0_target_tier"] = pd.qcut(x["b0_expected_targets"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        x["b0_target_tier"] = "UNAVAILABLE"
    tier_rows = []
    for tier, g in x.groupby("b0_target_tier", observed=True):
        apool = float(g["team_te_pool_component"].abs().sum())
        aalloc = float(g["individual_allocation_component"].abs().sum())
        denom = apool + aalloc
        m = metric(g["targets"], g["b0_expected_targets"], (2, 4, 6))
        tier_rows.append({
            "b0_target_tier": str(tier), "n": int(len(g)),
            "target_mae": m["mae"], "target_p90_abs": m["p90_abs"],
            "miss4": m["miss4"], "miss6": m["miss6"],
            "team_te_pool_abs_mass_share": apool / denom if denom else np.nan,
            "individual_allocation_abs_mass_share": aalloc / denom if denom else np.nan,
        })

    profiles = []
    for key, g in x.groupby("player_clean_key"):
        if len(g) < 20:
            continue
        apool = float(g["team_te_pool_component"].abs().sum())
        aalloc = float(g["individual_allocation_component"].abs().sum())
        denom = apool + aalloc
        ps = apool / denom if denom else 0.0
        als = aalloc / denom if denom else 0.0
        if ps >= .55:
            dom = "TEAM_TE_POOL"
        elif als >= .55:
            dom = "INDIVIDUAL_ALLOCATION"
        else:
            dom = "MIXED"
        m = metric(g["targets"], g["b0_expected_targets"], (4, 6))
        mode = g["player"].mode()
        profiles.append({
            "player_clean_key": key,
            "player": mode.iloc[0] if len(mode) else key,
            "games": int(len(g)), "seasons": int(g["season"].nunique()),
            "dominant_submechanism": dom,
            "team_te_pool_abs_mass_share": ps,
            "individual_allocation_abs_mass_share": als,
            "mean_abs_team_te_pool_component": float(g["team_te_pool_component"].abs().mean()),
            "mean_abs_individual_allocation_component": float(g["individual_allocation_component"].abs().mean()),
            "target_mae": m["mae"], "target_p90_abs": m["p90_abs"],
            "miss4": m["miss4"], "miss6": m["miss6"],
        })
    prof = pd.DataFrame(profiles)

    season_counts = {str(s): int(x["season"].eq(s).sum()) for s in SEASONS}
    gates = {
        "scoreable_ge4000": bool(len(x) >= 4000),
        "each_season_ge500": bool(all(v >= 500 for v in season_counts.values())),
        "target_identity_reconstruction_le1e_9": bool(max_identity <= 1e-9),
        "shapley_reconstruction_le1e_9": bool(max_shapley <= 1e-9),
        "players20_ge40": bool(len(prof) >= 40),
        "sportsbook_inputs_zero": True,
    }
    integrity_pass = all(gates.values())
    pooled_route = route(pooled_pool_share, pooled_alloc_share) if integrity_pass else "TE_R2_INTEGRITY_FAIL"
    high_route = route(high_pool_share, high_alloc_share) if integrity_pass else "TE_R2_INTEGRITY_FAIL"
    if not integrity_pass:
        disposition = "TE_R2_INTEGRITY_FAIL"
    elif pooled_route == high_route:
        disposition = pooled_route
    else:
        disposition = "TE_SPLIT_LAYER_ROUTE_DISAGREEMENT"

    result = {
        "migration": "TE_R2_POOL_VS_INDIVIDUAL_ALLOCATION",
        "parent_te_r1_run": 34123413402,
        "source_joint_run": 34081764151,
        "rows": int(len(x)),
        "team_games": int(len(team)),
        "season_counts": season_counts,
        "players_ge20": int(len(prof)),
        "max_target_identity_gap": max_identity,
        "max_shapley_gap": max_shapley,
        "pooled_absolute_mass": {
            "TEAM_TE_POOL": abs_pool,
            "INDIVIDUAL_ALLOCATION": abs_alloc,
        },
        "pooled_absolute_mass_share": {
            "TEAM_TE_POOL": pooled_pool_share,
            "INDIVIDUAL_ALLOCATION": pooled_alloc_share,
        },
        "pooled_route": pooled_route,
        "highest_error_quartile": {
            "threshold_abs_target_error": q75,
            "rows": int(len(high)),
            "dominant_counts": {str(k): int(v) for k, v in high["game_dominant"].value_counts().to_dict().items()},
            "absolute_mass_share": {
                "TEAM_TE_POOL": high_pool_share,
                "INDIVIDUAL_ALLOCATION": high_alloc_share,
            },
            "route": high_route,
        },
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(args.out_dir / "te_r2_casebook.csv", index=False)
    team.to_csv(args.out_dir / "te_r2_team_pool_casebook.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "te_r2_player_metric_summary.csv", index=False)
    pd.DataFrame(team_metric_rows).to_csv(args.out_dir / "te_r2_team_pool_metric_summary.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(args.out_dir / "te_r2_mechanism_summary.csv", index=False)
    pd.DataFrame(tier_rows).to_csv(args.out_dir / "te_r2_target_tier_summary.csv", index=False)
    prof.sort_values(["target_mae", "games"], ascending=[False, False]).to_csv(args.out_dir / "te_r2_player_profiles.csv", index=False)
    high.to_csv(args.out_dir / "te_r2_high_error_quartile.csv", index=False)
    (args.out_dir / "te_r2_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nPLAYER METRICS")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nMECHANISM SUMMARY")
    print(pd.DataFrame(mechanism_rows).to_string(index=False))
    print("\nTEAM POOL METRICS")
    print(pd.DataFrame(team_metric_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
