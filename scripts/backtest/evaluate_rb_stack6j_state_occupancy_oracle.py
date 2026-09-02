#!/usr/bin/env python3
"""RB STACK6J: no-fit M94C state-occupancy oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_WEEK = 6
ALPHA = 0.75
STACK6I_RATE_HEADROOM = 3.240694394219671
EXPECTED_N = 388
EXPECTED_BASE_MAE = 6.203454780519527
TEAM_MAP = {"JAX": "JAC", "LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}
STATES = ("lead", "neutral", "trail")


def num(x):
    return pd.to_numeric(x, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def team(v):
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s)


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def metric(actual: pd.Series, pred: pd.Series) -> dict:
    y, p = num(actual), num(pred)
    ok = y.notna() & p.notna()
    y, p = y.loc[ok], p.loc[ok]
    e = p - y
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(p.corr(y)) if len(y) >= 3 and p.nunique() > 1 and y.nunique() > 1 else np.nan,
    }


def score_pop(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for arm in ["BASE_M94C_TOTAL_RUSH", "ORACLE_STATE_OCCUPANCY"]:
        rows.append({"population": label, "arm": arm, **metric(df.actual_team_rush_att, df[arm])})
    out = pd.DataFrame(rows)
    b = float(out.loc[out.arm.eq("BASE_M94C_TOTAL_RUSH"), "mae"].iloc[0])
    out["mae_recovery_vs_base"] = b - out.mae
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    m = one(a.m94c_root, "m94c_2025_team_trace.csv")
    h = one(a.stack6h_root, "stack6h_team_trace.csv")
    for d in (m, h):
        d["season"] = num(d.season).astype(int)
        d["week"] = num(d.week).astype(int)
        d["team"] = d.team.map(team)

    req = [
        "season", "week", "team", "actual_team_rush_att", "baseline_team_rush_att",
        "pred_off_plays", "structured_team_rush_att", "candidate_team_rush_att",
    ]
    for s in STATES:
        req += [f"pred_{s}_play_share", f"{s}_play_share", f"gs_team_{s}_rush_rate_shrunk"]
    missing = [c for c in req if c not in m.columns]
    if missing:
        raise RuntimeError(f"M94C trace missing {missing}")

    bin_cols = ["pool_over_5", "pool_under_5", "pool_abs_5", "non_extreme_abs_lt3"]
    t = m[req].merge(h[["season", "week", "team", *bin_cols]], on=["season", "week", "team"], how="inner", validate="one_to_one")
    if len(t) != 544:
        raise RuntimeError(f"expected 544 joined team-games, found {len(t)}")

    for c in req:
        if c not in {"team"}:
            t[c] = num(t[c])

    pred_rate = pd.Series(0.0, index=t.index)
    oracle_rate = pd.Series(0.0, index=t.index)
    pred_share_sum = pd.Series(0.0, index=t.index)
    actual_share_sum = pd.Series(0.0, index=t.index)
    for s in STATES:
        ps = t[f"pred_{s}_play_share"]
        ac = t[f"{s}_play_share"]
        rr = t[f"gs_team_{s}_rush_rate_shrunk"]
        pred_rate += ps * rr
        oracle_rate += ac * rr
        pred_share_sum += ps
        actual_share_sum += ac

    t["structured_rebuilt"] = t.pred_off_plays * pred_rate
    t["base_rebuilt"] = (1 - ALPHA) * t.baseline_team_rush_att + ALPHA * t.structured_rebuilt
    t["BASE_M94C_TOTAL_RUSH"] = t.candidate_team_rush_att
    t["ORACLE_STATE_OCCUPANCY"] = (1 - ALPHA) * t.baseline_team_rush_att + ALPHA * t.pred_off_plays * oracle_rate
    t["pred_state_share_sum"] = pred_share_sum
    t["actual_state_share_sum"] = actual_share_sum

    w = t.loc[t.season.eq(2025) & t.week.ge(START_WEEK)].copy()
    overall = score_pop(w, "ALL_W6_18")
    masks = {
        "POOL_OVER_5": w.pool_over_5.eq(1),
        "POOL_UNDER_5": w.pool_under_5.eq(1),
        "POOL_ABS_5": w.pool_abs_5.eq(1),
        "NON_EXTREME_ABS_LT3": w.non_extreme_abs_lt3.eq(1),
    }
    bins = pd.concat([score_pop(w.loc[mask], label) for label, mask in masks.items()], ignore_index=True)

    state_scores = pd.DataFrame([
        {"state": s, **metric(w[f"{s}_play_share"], w[f"pred_{s}_play_share"])} for s in STATES
    ])

    def rec(pop: str) -> float:
        q = (overall if pop == "ALL_W6_18" else bins)
        x = q.loc[q.population.eq(pop) & q.arm.eq("ORACLE_STATE_OCCUPANCY"), "mae_recovery_vs_base"]
        return float(x.iloc[0])

    overall_rec = rec("ALL_W6_18")
    over_rec = rec("POOL_OVER_5")
    under_rec = rec("POOL_UNDER_5")
    frac = overall_rec / STACK6I_RATE_HEADROOM

    if overall_rec >= 1.00 and over_rec > 0 and under_rec > 0:
        disposition = "STATE_OCCUPANCY_MATERIAL"
    elif overall_rec < 0.50 or over_rec <= 0 or under_rec <= 0:
        disposition = "STATE_OCCUPANCY_NOT_PRIMARY"
    else:
        disposition = "STATE_OCCUPANCY_PARTIAL"

    structured_err = float((t.structured_rebuilt - t.structured_team_rush_att).abs().max())
    candidate_err = float((t.base_rebuilt - t.candidate_team_rush_att).abs().max())
    pred_share_err = float((t.pred_state_share_sum - 1.0).abs().max())
    actual_share_err = float((t.actual_state_share_sum - 1.0).abs().max())
    base_mae = float(overall.loc[overall.arm.eq("BASE_M94C_TOTAL_RUSH"), "mae"].iloc[0])
    integrity_pass = int(
        len(w) == EXPECTED_N
        and structured_err <= 1e-9
        and candidate_err <= 1e-9
        and pred_share_err <= 1e-9
        and actual_share_err <= 1e-9
        and abs(base_mae - EXPECTED_BASE_MAE) <= 1e-9
    )
    if not integrity_pass:
        disposition = "STACK6J_INTEGRITY_FAILURE_DO_NOT_INTERPRET"

    integrity = pd.DataFrame([{
        "m94c_rows": len(m), "stack6h_rows": len(h), "joined_rows": len(t), "w6_18_n": len(w),
        "structured_rebuild_max_abs_error": structured_err,
        "candidate_rebuild_max_abs_error": candidate_err,
        "pred_state_share_sum_max_abs_error": pred_share_err,
        "actual_state_share_sum_max_abs_error": actual_share_err,
        "expected_base_mae": EXPECTED_BASE_MAE, "observed_base_mae": base_mae,
        "integrity_pass": integrity_pass, "fitted_models": 0, "feature_search": 0,
        "hyperparameter_search": 0, "threshold_search": 0, "sportsbook_inputs": 0,
        "target_game_state_shares_used_as_oracle_only": 1,
    }])
    disposition_df = pd.DataFrame([{
        "occupancy_mae_recovery": overall_rec,
        "stack6i_rate_headroom": STACK6I_RATE_HEADROOM,
        "occupancy_headroom_fraction": frac,
        "pool_over5_occupancy_recovery": over_rec,
        "pool_under5_occupancy_recovery": under_rec,
        "disposition": disposition,
        "production_change": 0,
        "predictive_model_authorized": 0,
    }])

    t.to_csv(a.out_dir / "stack6j_team_trace.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6j_integrity.csv", index=False)
    state_scores.to_csv(a.out_dir / "stack6j_state_share_scores.csv", index=False)
    overall.to_csv(a.out_dir / "stack6j_overall_scores.csv", index=False)
    bins.to_csv(a.out_dir / "stack6j_bin_scores.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6j_disposition.csv", index=False)

    print("=== STACK6J integrity ==="); print(integrity.to_string(index=False))
    print("=== STACK6J state shares ==="); print(state_scores.to_string(index=False))
    print("=== STACK6J overall ==="); print(overall.to_string(index=False))
    print("=== STACK6J bins ==="); print(bins.to_string(index=False))
    print("=== STACK6J disposition ==="); print(disposition_df.to_string(index=False))
    print(f"STACK6J_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
