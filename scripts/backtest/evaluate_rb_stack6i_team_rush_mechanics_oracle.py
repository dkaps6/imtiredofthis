#!/usr/bin/env python3
"""RB STACK6I: no-fit oracle decomposition of M94C team-rush mechanics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_WEEK = 6
TEAM_MAP = {
    "JAX": "JAC",
    "LAR": "LA",
    "STL": "LA",
    "OAK": "LV",
    "SD": "LAC",
}

EXPECTED_W6_18_N = 388
EXPECTED_BASE_MAE = 6.203454780519527
EXPECTED_BASE_RMSE = 7.741320436123004
EXPECTED_BASE_BIAS = 0.17958778796400207
EXPECTED_BASE_CORR = 0.2503068366819914


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def canon_team(v):
    s = str(v).strip().upper()
    return TEAM_MAP.get(s, s)


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def metric_pair(actual: pd.Series, pred: pd.Series) -> dict:
    y = num(actual)
    p = num(pred)
    ok = y.notna() & p.notna()
    y, p = y.loc[ok], p.loc[ok]
    e = p - y
    corr = float(p.corr(y)) if len(y) >= 3 and p.nunique() > 1 and y.nunique() > 1 else np.nan
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()) if len(e) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        "bias": float(e.mean()) if len(e) else np.nan,
        "corr": corr,
    }


def score_population(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for arm in ["BASE_M94C_TOTAL_RUSH", "ORACLE_PLAYS", "ORACLE_RUSH_RATE", "ORACLE_BOTH"]:
        r = metric_pair(df["actual_team_rush_att"], df[arm])
        rows.append({"population": label, "arm": arm, **r})
    out = pd.DataFrame(rows)
    base_mae = float(out.loc[out.arm.eq("BASE_M94C_TOTAL_RUSH"), "mae"].iloc[0])
    out["mae_recovery_vs_base"] = base_mae - out.mae
    return out[["population", "arm", "n", "mae", "rmse", "bias", "corr", "mae_recovery_vs_base"]]


def build_trace(m94c_root: Path, stack6h_root: Path) -> pd.DataFrame:
    m = one(m94c_root, "m94c_2025_team_trace.csv")
    h = one(stack6h_root, "stack6h_team_trace.csv")

    req_m = [
        "season", "week", "team", "candidate_team_rush_att", "actual_team_rush_att",
        "pred_off_plays", "actual_off_plays",
    ]
    req_h = [
        "season", "week", "team", "pool_over_5", "pool_under_5",
        "pool_abs_5", "non_extreme_abs_lt3",
    ]
    miss_m = [c for c in req_m if c not in m.columns]
    miss_h = [c for c in req_h if c not in h.columns]
    if miss_m or miss_h:
        raise RuntimeError(f"missing columns: m94c={miss_m} stack6h={miss_h}")

    for d in (m, h):
        d["season"] = num(d["season"]).astype(int)
        d["week"] = num(d["week"]).astype(int)
        d["team"] = d["team"].map(canon_team)

    cols = ["season", "week", "team", "pool_over_5", "pool_under_5", "pool_abs_5", "non_extreme_abs_lt3"]
    t = m[req_m].merge(h[cols], on=["season", "week", "team"], how="inner", validate="one_to_one")

    for c in ["candidate_team_rush_att", "actual_team_rush_att", "pred_off_plays", "actual_off_plays"]:
        t[c] = num(t[c])

    bad = (
        t[["candidate_team_rush_att", "actual_team_rush_att", "pred_off_plays", "actual_off_plays"]]
        .isna().any(axis=1)
        | t["pred_off_plays"].le(0)
        | t["actual_off_plays"].le(0)
    )
    if bad.any():
        sample = t.loc[bad, ["season", "week", "team", "pred_off_plays", "actual_off_plays"]].head(10)
        raise RuntimeError(f"STACK6I denominator/missing integrity failure on {int(bad.sum())} rows:\n{sample}")

    t["pred_effective_rush_rate"] = t["candidate_team_rush_att"] / t["pred_off_plays"]
    t["actual_effective_rush_rate"] = t["actual_team_rush_att"] / t["actual_off_plays"]

    t["BASE_M94C_TOTAL_RUSH"] = t["candidate_team_rush_att"]
    t["ORACLE_PLAYS"] = t["actual_off_plays"] * t["pred_effective_rush_rate"]
    t["ORACLE_RUSH_RATE"] = t["pred_off_plays"] * t["actual_effective_rush_rate"]
    t["ORACLE_BOTH"] = t["actual_off_plays"] * t["actual_effective_rush_rate"]
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    trace = build_trace(a.m94c_root, a.stack6h_root)
    w = trace.loc[trace.season.eq(2025) & trace.week.ge(START_WEEK)].copy()

    overall = score_population(w, "ALL_W6_18")
    masks = {
        "POOL_OVER_5": w.pool_over_5.eq(1),
        "POOL_UNDER_5": w.pool_under_5.eq(1),
        "POOL_ABS_5": w.pool_abs_5.eq(1),
        "NON_EXTREME_ABS_LT3": w.non_extreme_abs_lt3.eq(1),
    }
    bins = pd.concat([score_population(w.loc[mask], label) for label, mask in masks.items()], ignore_index=True)

    components = pd.DataFrame([
        {"component": "OFFENSIVE_PLAYS", **metric_pair(w.actual_off_plays, w.pred_off_plays)},
        {"component": "EFFECTIVE_RUSH_RATE", **metric_pair(w.actual_effective_rush_rate, w.pred_effective_rush_rate)},
    ])

    def recovery(table: pd.DataFrame, pop: str, arm: str) -> float:
        q = table.loc[table.population.eq(pop) & table.arm.eq(arm), "mae_recovery_vs_base"]
        return float(q.iloc[0]) if len(q) else np.nan

    play_rec = recovery(overall, "ALL_W6_18", "ORACLE_PLAYS")
    rate_rec = recovery(overall, "ALL_W6_18", "ORACLE_RUSH_RATE")
    play_over = recovery(bins, "POOL_OVER_5", "ORACLE_PLAYS")
    rate_over = recovery(bins, "POOL_OVER_5", "ORACLE_RUSH_RATE")
    play_under = recovery(bins, "POOL_UNDER_5", "ORACLE_PLAYS")
    rate_under = recovery(bins, "POOL_UNDER_5", "ORACLE_RUSH_RATE")

    if play_rec - rate_rec >= 0.50 and play_over >= rate_over and play_under >= rate_under:
        disposition = "PLAY_VOLUME_DOMINANT"
    elif rate_rec - play_rec >= 0.50 and rate_over >= play_over and rate_under >= play_under:
        disposition = "RUSH_RATE_DOMINANT"
    else:
        disposition = "MIXED_TOTAL_RUSH_MECHANICS"

    base = overall.loc[overall.arm.eq("BASE_M94C_TOTAL_RUSH")].iloc[0]
    base_identity_err = float((w.BASE_M94C_TOTAL_RUSH - (w.pred_off_plays * w.pred_effective_rush_rate)).abs().max())
    actual_identity_err = float((w.ORACLE_BOTH - w.actual_team_rush_att).abs().max())
    integrity_pass = int(
        len(w) == EXPECTED_W6_18_N
        and base_identity_err <= 1e-9
        and actual_identity_err <= 1e-9
        and abs(float(base.mae) - EXPECTED_BASE_MAE) <= 1e-9
        and abs(float(base.rmse) - EXPECTED_BASE_RMSE) <= 1e-9
        and abs(float(base.bias) - EXPECTED_BASE_BIAS) <= 1e-9
        and abs(float(base.corr) - EXPECTED_BASE_CORR) <= 1e-9
    )
    if not integrity_pass:
        disposition = "STACK6I_INTEGRITY_FAILURE_DO_NOT_INTERPRET"

    integrity = pd.DataFrame([{
        "m94c_rows": int(len(one(a.m94c_root, "m94c_2025_team_trace.csv"))),
        "stack6h_rows": int(len(one(a.stack6h_root, "stack6h_team_trace.csv"))),
        "joined_rows": int(len(trace)),
        "w6_18_team_games": int(len(w)),
        "base_factorization_max_abs_error": base_identity_err,
        "actual_factorization_max_abs_error": actual_identity_err,
        "expected_base_mae": EXPECTED_BASE_MAE,
        "observed_base_mae": float(base.mae),
        "expected_base_rmse": EXPECTED_BASE_RMSE,
        "observed_base_rmse": float(base.rmse),
        "expected_base_bias": EXPECTED_BASE_BIAS,
        "observed_base_bias": float(base.bias),
        "expected_base_corr": EXPECTED_BASE_CORR,
        "observed_base_corr": float(base.corr),
        "integrity_pass": integrity_pass,
        "fitted_models": 0,
        "hyperparameter_search": 0,
        "feature_search": 0,
        "threshold_search": 0,
        "sportsbook_inputs": 0,
        "actual_plays_used_as_oracle_only": 1,
        "actual_effective_rush_rate_used_as_oracle_only": 1,
    }])

    disposition_df = pd.DataFrame([{
        "oracle_plays_mae_recovery": play_rec,
        "oracle_rush_rate_mae_recovery": rate_rec,
        "rush_rate_minus_play_recovery": rate_rec - play_rec,
        "pool_over5_plays_recovery": play_over,
        "pool_over5_rush_rate_recovery": rate_over,
        "pool_under5_plays_recovery": play_under,
        "pool_under5_rush_rate_recovery": rate_under,
        "disposition": disposition,
        "production_change": 0,
        "predictive_model_authorized": 0,
    }])

    trace.to_csv(a.out_dir / "stack6i_team_trace.csv", index=False)
    components.to_csv(a.out_dir / "stack6i_component_scores.csv", index=False)
    overall.to_csv(a.out_dir / "stack6i_overall_oracle_scores.csv", index=False)
    bins.to_csv(a.out_dir / "stack6i_bin_oracle_scores.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6i_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6i_disposition.csv", index=False)

    print("=== STACK6I integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6I component scores ===")
    print(components.to_string(index=False))
    print("=== STACK6I overall oracle scores ===")
    print(overall.to_string(index=False))
    print("=== STACK6I bin oracle scores ===")
    print(bins.to_string(index=False))
    print("=== STACK6I disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6I_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
