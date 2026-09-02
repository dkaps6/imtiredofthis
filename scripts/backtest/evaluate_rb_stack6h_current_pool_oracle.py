#!/usr/bin/env python3
"""RB STACK6H: no-fit oracle decomposition of current P3 team-RB-pool error."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2

RB_POS = {"RB", "HB", "FB"}
START_WEEK = 6


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def build_trace(stack6_root: Path, m94c_root: Path) -> tuple[pd.DataFrame, dict]:
    casebook = one(stack6_root, "stack6_2025_casebook.csv")
    team = one(m94c_root, "m94c_2025_team_trace.csv")

    casebook["season"] = num(casebook.get("season", 2025)).fillna(2025).astype(int)
    casebook["week"] = num(casebook["week"]).astype(int)
    casebook["team"] = casebook["team"].map(s2.tm)
    if "parent_att" not in casebook.columns:
        raise RuntimeError("STACK6 casebook missing frozen parent_att")
    casebook["parent_att"] = num(casebook["parent_att"])
    if casebook.parent_att.isna().any():
        raise RuntimeError("STACK6 casebook contains missing parent_att")

    p3 = (
        casebook.groupby(["season", "week", "team"], as_index=False)
        .agg(p3_rb_pool=("parent_att", "sum"))
    )

    logs = s2.load_weekly_logs([2025])
    logs = logs.loc[logs.season.eq(2025)].copy()
    rb = (
        logs.loc[logs.position.isin(RB_POS)]
        .groupby(["season", "week", "team"], as_index=False)
        .agg(actual_rb_carries=("rushes", "sum"))
    )

    team["season"] = num(team.get("season", 2025)).fillna(2025).astype(int)
    team["week"] = num(team["week"]).astype(int)
    team["team"] = team["team"].map(s2.tm)
    required = ["candidate_team_rush_att", "actual_team_rush_att"]
    missing = [c for c in required if c not in team.columns]
    if missing:
        raise RuntimeError(f"M94C team trace missing required columns: {missing}")
    team["candidate_team_rush_att"] = num(team["candidate_team_rush_att"])
    team["actual_team_rush_att"] = num(team["actual_team_rush_att"])

    t = p3.merge(rb, on=["season", "week", "team"], how="left", validate="one_to_one")
    t = t.merge(
        team[["season", "week", "team", "candidate_team_rush_att", "actual_team_rush_att"]],
        on=["season", "week", "team"], how="inner", validate="one_to_one"
    )

    t["T_hat"] = t.candidate_team_rush_att
    t["T"] = t.actual_team_rush_att
    t["R_hat"] = t.p3_rb_pool
    t["R"] = t.actual_rb_carries
    bad = (
        t[["T_hat", "T", "R_hat", "R"]].isna().any(axis=1)
        | t.T_hat.le(0) | t.T.le(0)
    )
    invalid_rows = int(bad.sum())
    t = t.loc[~bad].copy()

    t["S_hat"] = t.R_hat / t.T_hat
    t["S"] = t.R / t.T
    t["BASE_P3_POOL"] = t.R_hat
    t["ORACLE_TOTAL_RUSH"] = t.T * t.S_hat
    t["ORACLE_RB_SHARE"] = t.T_hat * t.S
    t["ORACLE_BOTH"] = t.T * t.S
    t["base_residual"] = t.BASE_P3_POOL - t.R
    t["base_abs_residual"] = t.base_residual.abs()

    r = t.base_residual
    t["POOL_OVER_3"] = r.ge(3).astype(int)
    t["POOL_OVER_5"] = r.ge(5).astype(int)
    t["POOL_UNDER_3"] = r.le(-3).astype(int)
    t["POOL_UNDER_5"] = r.le(-5).astype(int)
    t["POOL_ABS_5"] = r.abs().ge(5).astype(int)
    t["NON_EXTREME_ABS_LT3"] = r.abs().lt(3).astype(int)

    meta = {
        "stack6_casebook_rows": len(casebook),
        "p3_team_games": len(p3),
        "actual_rb_team_games": len(rb),
        "m94c_team_games": len(team),
        "joined_team_games_before_invalid_drop": len(t) + invalid_rows,
        "invalid_denominator_or_missing_rows": invalid_rows,
    }
    return t, meta


def score(df: pd.DataFrame, pred_col: str) -> dict:
    y = num(df.R)
    p = num(df[pred_col])
    ok = y.notna() & p.notna()
    y, p = y.loc[ok], p.loc[ok]
    e = p - y
    corr = float(p.corr(y)) if len(y) >= 3 and p.nunique() > 1 and y.nunique() > 1 else np.nan
    return {
        "arm": pred_col,
        "n": int(len(y)),
        "mae": float(e.abs().mean()) if len(e) else np.nan,
        "rmse": float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan,
        "bias": float(e.mean()) if len(e) else np.nan,
        "corr": corr,
    }


def score_population(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for arm in ["BASE_P3_POOL", "ORACLE_TOTAL_RUSH", "ORACLE_RB_SHARE", "ORACLE_BOTH"]:
        r = score(df, arm)
        r["population"] = label
        rows.append(r)
    out = pd.DataFrame(rows)
    base_mae = float(out.loc[out.arm.eq("BASE_P3_POOL"), "mae"].iloc[0])
    out["mae_recovery_vs_base"] = base_mae - out.mae
    return out[["population", "arm", "n", "mae", "rmse", "bias", "corr", "mae_recovery_vs_base"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    trace, meta = build_trace(a.stack6_root, a.m94c_root)
    w = trace.loc[trace.season.eq(2025) & trace.week.ge(START_WEEK)].copy()

    overall = score_population(w, "ALL_W6_18")
    masks = {
        "POOL_OVER_3": w.POOL_OVER_3.eq(1),
        "POOL_OVER_5": w.POOL_OVER_5.eq(1),
        "POOL_UNDER_3": w.POOL_UNDER_3.eq(1),
        "POOL_UNDER_5": w.POOL_UNDER_5.eq(1),
        "POOL_ABS_5": w.POOL_ABS_5.eq(1),
        "NON_EXTREME_ABS_LT3": w.NON_EXTREME_ABS_LT3.eq(1),
    }
    bins = pd.concat([score_population(w.loc[m], name) for name, m in masks.items()], ignore_index=True)

    def recovery(table: pd.DataFrame, pop: str, arm: str) -> float:
        q = table.loc[table.population.eq(pop) & table.arm.eq(arm), "mae_recovery_vs_base"]
        return float(q.iloc[0]) if len(q) else np.nan

    total_rec = recovery(overall, "ALL_W6_18", "ORACLE_TOTAL_RUSH")
    share_rec = recovery(overall, "ALL_W6_18", "ORACLE_RB_SHARE")
    total_over5 = recovery(bins, "POOL_OVER_5", "ORACLE_TOTAL_RUSH")
    share_over5 = recovery(bins, "POOL_OVER_5", "ORACLE_RB_SHARE")
    total_under5 = recovery(bins, "POOL_UNDER_5", "ORACLE_TOTAL_RUSH")
    share_under5 = recovery(bins, "POOL_UNDER_5", "ORACLE_RB_SHARE")

    if (
        total_rec - share_rec >= 0.50
        and total_over5 >= share_over5
        and total_under5 >= share_under5
    ):
        disposition = "TOTAL_TEAM_RUSHING_DOMINANT"
    elif (
        share_rec - total_rec >= 0.50
        and share_over5 >= total_over5
        and share_under5 >= total_under5
    ):
        disposition = "RB_SHARE_DOMINANT"
    else:
        disposition = "MIXED_TEAM_POOL_BOTTLENECK"

    both_max_abs = float((w.ORACLE_BOTH - w.R).abs().max()) if len(w) else np.nan
    base_row = overall.loc[overall.arm.eq("BASE_P3_POOL")].iloc[0]
    expected_stack6f_mae = 5.741945977117434
    stack6f_mae_delta = abs(float(base_row.mae) - expected_stack6f_mae)
    integrity_pass = int(
        len(w) > 0
        and both_max_abs <= 1e-9
        and stack6f_mae_delta <= 1e-6
        and meta["invalid_denominator_or_missing_rows"] == 0
    )
    if not integrity_pass:
        disposition = "STACK6H_INTEGRITY_FAILURE_DO_NOT_INTERPRET"

    integrity = pd.DataFrame([{
        **meta,
        "w6_18_team_games": len(w),
        "oracle_both_max_abs_error": both_max_abs,
        "expected_stack6f_p3_mae": expected_stack6f_mae,
        "observed_base_p3_mae": float(base_row.mae),
        "stack6f_mae_abs_delta": stack6f_mae_delta,
        "integrity_pass": integrity_pass,
        "fitted_models": 0,
        "hyperparameter_search": 0,
        "feature_search": 0,
        "threshold_search": 0,
        "sportsbook_inputs": 0,
        "actual_total_rush_used_as_oracle_only": 1,
        "actual_rb_share_used_as_oracle_only": 1,
    }])

    disposition_df = pd.DataFrame([{
        "total_rush_mae_recovery": total_rec,
        "rb_share_mae_recovery": share_rec,
        "total_minus_share_recovery": total_rec - share_rec,
        "pool_over5_total_recovery": total_over5,
        "pool_over5_share_recovery": share_over5,
        "pool_under5_total_recovery": total_under5,
        "pool_under5_share_recovery": share_under5,
        "disposition": disposition,
        "production_change": 0,
        "predictive_model_authorized": 0,
    }])

    trace.to_csv(a.out_dir / "stack6h_team_trace.csv", index=False)
    overall.to_csv(a.out_dir / "stack6h_overall_oracle_scores.csv", index=False)
    bins.to_csv(a.out_dir / "stack6h_bin_oracle_scores.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6h_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6h_disposition.csv", index=False)

    print("=== STACK6H integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6H overall ===")
    print(overall.to_string(index=False))
    print("=== STACK6H bins ===")
    print(bins.to_string(index=False))
    print("=== STACK6H disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6H_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
