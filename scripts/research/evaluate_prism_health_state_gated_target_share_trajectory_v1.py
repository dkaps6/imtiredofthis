#!/usr/bin/env python3
"""PRISM Health-State-Gated Target Share Trajectory V1.

Frozen integration test over immutable trajectory OOS-fold rows. The only
candidate change is neutralizing trajectory_delta when the immutable source
bridge marks RECENT2 as observation-state contaminated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1"
FREEZE_COMMIT = "6c721623d14d8990aef5144c54dbbe48e2f2b434"
TOL = 1e-12
BOOT_REPS = 5000
BOOT_SEED = 20261018

KEYS = ["season", "week", "team", "player_clean_key", "position_group"]
ROOM_KEYS = ["season", "week", "event_id", "team", "position_group"]

INTEGRATION_COLS = [
    "season", "week", "event_id", "team", "player_clean_key", "player",
    "is_wr1", "position_group", "trajectory_delta", "feature_max_week",
    "baseline_targets", "baseline_ypt", "shadow_targets", "actual_targets",
    "actual_rec_yards", "shadow_rec_yards", "shadow_room_share",
    "actual_room_share",
]
BRIDGE_COLS = KEYS + [
    "recent2_observation_contaminated",
    "recent2_participation_certified",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def boolish(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = "" if v is None or pd.isna(v) else str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty integration rows: {path}")
    x = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in set(INTEGRATION_COLS), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = set(INTEGRATION_COLS) - set(x.columns)
    if miss:
        raise RuntimeError(f"integration authority missing columns: {sorted(miss)}")
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["feature_max_week"] = num(x["feature_max_week"]).astype(int)
    x["position_group"] = x["position_group"].astype(str).str.upper().str.strip()
    x["is_wr1"] = x["is_wr1"].map(boolish)
    for c in [
        "trajectory_delta", "baseline_targets", "baseline_ypt", "shadow_targets",
        "actual_targets", "actual_rec_yards", "shadow_rec_yards",
        "shadow_room_share", "actual_room_share",
    ]:
        x[c] = num(x[c])
    if len(x) != 3328:
        raise RuntimeError(f"integration authority row drift {len(x)} != 3328")
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate integration authority identity")
    expected = {("WR", 2024): 1619, ("TE", 2024): 844, ("TE", 2025): 865}
    got = {
        (str(p), int(s)): int(len(g))
        for (p, s), g in x.groupby(["position_group", "season"], sort=False)
    }
    if got != expected:
        raise RuntimeError(f"integration cell drift {got} != {expected}")
    if (x["feature_max_week"] >= x["week"]).any():
        raise RuntimeError("same/future trajectory chronology in integration authority")
    return x.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def load_bridge(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty source bridge: {path}")
    x = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in set(BRIDGE_COLS), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = set(BRIDGE_COLS) - set(x.columns)
    if miss:
        raise RuntimeError(f"source bridge missing columns: {sorted(miss)}")
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["position_group"] = x["position_group"].astype(str).str.upper().str.strip()
    x["recent2_observation_contaminated"] = x["recent2_observation_contaminated"].map(boolish)
    x["recent2_participation_certified"] = x["recent2_participation_certified"].map(boolish)
    if x.duplicated(KEYS).any():
        raise RuntimeError("duplicate source bridge identity")
    expected = {("WR", 2024): 260, ("TE", 2024): 100, ("TE", 2025): 108}
    got = {
        (str(p), int(s)): int(g["recent2_observation_contaminated"].sum())
        for (p, s), g in x.groupby(["position_group", "season"], sort=False)
    }
    if got != expected:
        raise RuntimeError(f"contamination count drift {got} != {expected}")
    if (~(x["recent2_observation_contaminated"] ^ x["recent2_participation_certified"])).any():
        raise RuntimeError("bridge contamination/certified flags are not complements")
    return x


def renormalize(values: np.ndarray, deltas: np.ndarray, pool: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(deltas).all():
        raise RuntimeError("nonfinite allocation input")
    if (values < -TOL).any():
        raise RuntimeError("negative baseline target opportunity")
    if pool <= TOL:
        return np.zeros(len(values), dtype=float)
    weights = values * np.exp(deltas)
    denom = float(weights.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("invalid candidate room weights")
    cand = pool * weights / denom
    gap = float(pool) - float(cand.sum())
    cand[int(np.argmax(weights))] += gap
    return cand


def apply_candidate(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = rows.copy()
    out["health_state_trajectory_delta"] = np.where(
        out["recent2_observation_contaminated"],
        0.0,
        out["trajectory_delta"],
    )
    out["candidate_targets"] = out["baseline_targets"].astype(float)
    audits = []

    for room, idx in out.groupby(ROOM_KEYS, sort=False).groups.items():
        g = out.loc[idx]
        pos = str(room[-1])
        if pos == "WR":
            anchors = g["is_wr1"].astype(bool)
            if int(anchors.sum()) != 1:
                raise RuntimeError(f"WR room anchor count drift {room}: {anchors.sum()}")
            anchor_idx = g.index[anchors]
            sec_idx = g.index[~anchors]
            sec_pool = float(g.loc[sec_idx, "baseline_targets"].sum())
            if len(sec_idx):
                cand = renormalize(
                    g.loc[sec_idx, "baseline_targets"].to_numpy(float),
                    g.loc[sec_idx, "health_state_trajectory_delta"].to_numpy(float),
                    sec_pool,
                )
                out.loc[sec_idx, "candidate_targets"] = cand
            anchor_gap = float(
                (out.loc[anchor_idx, "candidate_targets"] - g.loc[anchor_idx, "baseline_targets"]).abs().max()
            )
            protected_gap = abs(float(out.loc[sec_idx, "candidate_targets"].sum()) - sec_pool) if len(sec_idx) else 0.0
        elif pos == "TE":
            anchor_gap = 0.0
            pool = float(g["baseline_targets"].sum())
            cand = renormalize(
                g["baseline_targets"].to_numpy(float),
                g["health_state_trajectory_delta"].to_numpy(float),
                pool,
            )
            out.loc[g.index, "candidate_targets"] = cand
            protected_gap = abs(float(out.loc[g.index, "candidate_targets"].sum()) - pool)
        else:
            raise RuntimeError(f"unexpected position in room {room}")

        room_gap = abs(
            float(out.loc[g.index, "candidate_targets"].sum())
            - float(g["baseline_targets"].sum())
        )
        audits.append({
            **dict(zip(ROOM_KEYS, room)),
            "wr1_max_abs_gap": anchor_gap,
            "protected_pool_max_abs_gap": protected_gap,
            "room_pool_max_abs_gap": room_gap,
        })

    out["candidate_rec_yards"] = out["candidate_targets"] * out["baseline_ypt"]
    base_pool = out.groupby(ROOM_KEYS)["baseline_targets"].transform("sum")
    out["candidate_room_share"] = np.where(
        base_pool.gt(0),
        out["candidate_targets"] / base_pool,
        np.nan,
    )
    return out, pd.DataFrame(audits)


def metrics(df: pd.DataFrame) -> dict:
    z = df.dropna(subset=["actual_targets", "actual_rec_yards"]).copy()
    if z.empty:
        raise RuntimeError("zero rows for metric")
    orig_t = (z["shadow_targets"] - z["actual_targets"]).abs()
    cand_t = (z["candidate_targets"] - z["actual_targets"]).abs()
    orig_y = (z["shadow_rec_yards"] - z["actual_rec_yards"]).abs()
    cand_y = (z["candidate_rec_yards"] - z["actual_rec_yards"]).abs()
    room = z.dropna(subset=["actual_room_share", "shadow_room_share", "candidate_room_share"]).copy()
    orig_r = (room["shadow_room_share"] - room["actual_room_share"]).abs()
    cand_r = (room["candidate_room_share"] - room["actual_room_share"]).abs()
    return {
        "rows": int(len(z)),
        "players": int(z["player_clean_key"].nunique()),
        "original_target_mae": float(orig_t.mean()),
        "candidate_target_mae": float(cand_t.mean()),
        "target_mae_improvement": float(orig_t.mean() - cand_t.mean()),
        "original_rec_yards_mae": float(orig_y.mean()),
        "candidate_rec_yards_mae": float(cand_y.mean()),
        "rec_yards_mae_improvement": float(orig_y.mean() - cand_y.mean()),
        "original_room_share_mae": float(orig_r.mean()),
        "candidate_room_share_mae": float(cand_r.mean()),
        "room_share_mae_improvement": float(orig_r.mean() - cand_r.mean()),
    }


def affected_player_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    z = df.loc[df["recent2_observation_contaminated"]].dropna(subset=["actual_targets"]).copy()
    if z.empty:
        raise RuntimeError("zero directly affected rows")
    z["target_ae_improvement"] = (
        (z["shadow_targets"] - z["actual_targets"]).abs()
        - (z["candidate_targets"] - z["actual_targets"]).abs()
    )
    by = (
        z.groupby(["position_group", "player_clean_key"], as_index=False)["target_ae_improvement"]
        .mean()
        .rename(columns={"target_ae_improvement": "mean_target_ae_improvement"})
    )
    vals = by["mean_target_ae_improvement"].to_numpy(float)
    summary = {
        "rows": int(len(z)),
        "players": int(len(by)),
        "mean_row_target_ae_improvement": float(z["target_ae_improvement"].mean()),
        "equal_player_mean_improvement": float(vals.mean()),
        "median_player_improvement": float(np.median(vals)),
        "p10_player_improvement": float(np.quantile(vals, 0.10)),
        "player_improved_share": float((vals > 0).mean()),
        "player_tied_share": float(np.isclose(vals, 0.0, atol=1e-15).mean()),
        "player_worsened_share": float((vals < 0).mean()),
    }

    clusters = {
        f"{p}|{k}": g["target_ae_improvement"].to_numpy(float)
        for (p, k), g in z.groupby(["position_group", "player_clean_key"], sort=False)
    }
    keys = sorted(clusters)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sample = rng.choice(keys, size=len(keys), replace=True)
        arr = np.concatenate([clusters[k] for k in sample])
        boots[i] = float(arr.mean())
    summary["bootstrap"] = {
        "reps": BOOT_REPS,
        "seed": BOOT_SEED,
        "clusters": int(len(keys)),
        "p_improvement_gt_0": float((boots > 0).mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }
    return summary, by


def json_default(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    raise TypeError(f"Object of type {v.__class__.__name__} is not JSON serializable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--integration-rows", type=Path, required=True)
    ap.add_argument("--bridge-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(a.integration_rows)
    bridge = load_bridge(a.bridge_rows)
    rows = rows.merge(bridge, on=KEYS, how="left", validate="one_to_one")
    if rows[["recent2_observation_contaminated", "recent2_participation_certified"]].isna().any().any():
        raise RuntimeError("source bridge coverage incomplete on exact integration population")

    candidate, audit = apply_candidate(rows)
    max_wr1 = float(audit["wr1_max_abs_gap"].max())
    max_protected = float(audit["protected_pool_max_abs_gap"].max())
    max_room = float(audit["room_pool_max_abs_gap"].max())
    if max(max_wr1, max_protected, max_room) > TOL:
        raise RuntimeError(f"conservation failure wr1={max_wr1} protected={max_protected} room={max_room}")

    primary = candidate.loc[candidate["season"].eq(2024)].copy()
    primary_wr = primary.loc[primary["position_group"].eq("WR")].copy()
    primary_te = primary.loc[primary["position_group"].eq("TE")].copy()
    primary_affected = primary.loc[primary["recent2_observation_contaminated"]].copy()
    primary_clean = primary.loc[primary["recent2_participation_certified"]].copy()
    te25 = candidate.loc[candidate["season"].eq(2025) & candidate["position_group"].eq("TE")].copy()
    te25_affected = te25.loc[te25["recent2_observation_contaminated"]].copy()

    m_all = metrics(primary)
    m_wr = metrics(primary_wr)
    m_te = metrics(primary_te)
    m_aff = metrics(primary_affected)
    m_clean = metrics(primary_clean)
    m_te25 = metrics(te25)
    m_te25_aff = metrics(te25_affected)
    player_summary, player_rows = affected_player_metrics(primary)

    gates = {
        "primary_affected_target_mae_improves": bool(m_aff["target_mae_improvement"] > 0),
        "primary_affected_equal_player_mean_positive": bool(player_summary["equal_player_mean_improvement"] > 0),
        "primary_affected_median_player_nonnegative": bool(player_summary["median_player_improvement"] >= 0),
        "primary_affected_improved_share_gt_worsened": bool(player_summary["player_improved_share"] > player_summary["player_worsened_share"]),
        "primary_affected_bootstrap_p_ge_0p80": bool(player_summary["bootstrap"]["p_improvement_gt_0"] >= 0.80),
        "primary_all_target_mae_nonworse": bool(m_all["target_mae_improvement"] >= -TOL),
        "primary_wr_target_mae_nonworse": bool(m_wr["target_mae_improvement"] >= -TOL),
        "primary_te_target_mae_nonworse": bool(m_te["target_mae_improvement"] >= -TOL),
        "primary_all_rec_yards_nonworse": bool(m_all["rec_yards_mae_improvement"] >= -TOL),
        "primary_wr_rec_yards_nonworse": bool(m_wr["rec_yards_mae_improvement"] >= -TOL),
        "primary_te_rec_yards_nonworse": bool(m_te["rec_yards_mae_improvement"] >= -TOL),
        "primary_room_share_nonworse": bool(m_all["room_share_mae_improvement"] >= -TOL),
        "primary_clean_target_mae_nonworse": bool(m_clean["target_mae_improvement"] >= -TOL),
        "conservation_pass": bool(max(max_wr1, max_protected, max_room) <= TOL),
        "te2025_affected_target_mae_improves": bool(m_te25_aff["target_mae_improvement"] > 0),
        "te2025_all_target_mae_nonworse": bool(m_te25["target_mae_improvement"] >= -TOL),
        "te2025_rec_yards_nonworse": bool(m_te25["rec_yards_mae_improvement"] >= -TOL),
        "te2025_room_share_nonworse": bool(m_te25["room_share_mae_improvement"] >= -TOL),
    }
    qualified = bool(all(gates.values()))

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "disposition": (
            "PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_QUALIFIED"
            if qualified else
            "PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_CLOSED"
        ),
        "primary_all_2024": m_all,
        "primary_wr_2024": m_wr,
        "primary_te_2024": m_te,
        "primary_affected_2024": m_aff,
        "primary_clean_2024": m_clean,
        "primary_affected_player_summary": player_summary,
        "secondary_te_2025": m_te25,
        "secondary_te_2025_affected": m_te25_aff,
        "gates": gates,
        "max_wr1_abs_gap": max_wr1,
        "max_protected_pool_abs_gap": max_protected,
        "max_room_pool_abs_gap": max_room,
        "contamination_counts": {
            "WR_2024": int(primary_wr["recent2_observation_contaminated"].sum()),
            "TE_2024": int(primary_te["recent2_observation_contaminated"].sum()),
            "TE_2025": int(te25["recent2_observation_contaminated"].sum()),
        },
        "candidate_action": "trajectory_delta_to_zero_when_recent2_observation_contaminated",
        "original_trajectory_formula_changed": False,
        "recent_window_changed": False,
        "efficiency_changed": False,
        "distribution_changed": False,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "provider_rebuilds": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    candidate.to_csv(a.out_dir / "health_state_gated_trajectory_rows.csv", index=False)
    audit.to_csv(a.out_dir / "health_state_gated_trajectory_conservation.csv", index=False)
    player_rows.to_csv(a.out_dir / "health_state_gated_trajectory_affected_player_summary.csv", index=False)
    (a.out_dir / "health_state_gated_trajectory_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
