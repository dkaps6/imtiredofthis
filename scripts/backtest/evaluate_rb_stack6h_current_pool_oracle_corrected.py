#!/usr/bin/env python3
"""Mechanical wrapper for STACK6H explicit pandas `T` column access."""
from __future__ import annotations

import pandas as pd

from scripts.backtest import evaluate_rb_stack6h_current_pool_oracle as base
from scripts.backtest import evaluate_rb_stack2_enriched_allocation as s2


def corrected_build_trace(stack6_root, m94c_root):
    casebook = base.one(stack6_root, "stack6_2025_casebook.csv")
    team = base.one(m94c_root, "m94c_2025_team_trace.csv")

    casebook["season"] = base.num(casebook.get("season", 2025)).fillna(2025).astype(int)
    casebook["week"] = base.num(casebook["week"]).astype(int)
    casebook["team"] = casebook["team"].map(s2.tm)
    if "parent_att" not in casebook.columns:
        raise RuntimeError("STACK6 casebook missing frozen parent_att")
    casebook["parent_att"] = base.num(casebook["parent_att"])
    if casebook.parent_att.isna().any():
        raise RuntimeError("STACK6 casebook contains missing parent_att")

    p3 = (
        casebook.groupby(["season", "week", "team"], as_index=False)
        .agg(p3_rb_pool=("parent_att", "sum"))
    )

    logs = s2.load_weekly_logs([2025])
    logs = logs.loc[logs.season.eq(2025)].copy()
    rb = (
        logs.loc[logs.position.isin(base.RB_POS)]
        .groupby(["season", "week", "team"], as_index=False)
        .agg(actual_rb_carries=("rushes", "sum"))
    )

    team["season"] = base.num(team.get("season", 2025)).fillna(2025).astype(int)
    team["week"] = base.num(team["week"]).astype(int)
    team["team"] = team["team"].map(s2.tm)
    required = ["candidate_team_rush_att", "actual_team_rush_att"]
    missing = [c for c in required if c not in team.columns]
    if missing:
        raise RuntimeError(f"M94C team trace missing required columns: {missing}")
    team["candidate_team_rush_att"] = base.num(team["candidate_team_rush_att"])
    team["actual_team_rush_att"] = base.num(team["actual_team_rush_att"])

    t = p3.merge(rb, on=["season", "week", "team"], how="left", validate="one_to_one")
    t = t.merge(
        team[["season", "week", "team", "candidate_team_rush_att", "actual_team_rush_att"]],
        on=["season", "week", "team"], how="inner", validate="one_to_one"
    )

    t["T_hat"] = t["candidate_team_rush_att"]
    t["T"] = t["actual_team_rush_att"]
    t["R_hat"] = t["p3_rb_pool"]
    t["R"] = t["actual_rb_carries"]
    bad = (
        t[["T_hat", "T", "R_hat", "R"]].isna().any(axis=1)
        | t["T_hat"].le(0) | t["T"].le(0)
    )
    invalid_rows = int(bad.sum())
    t = t.loc[~bad].copy()

    t["S_hat"] = t["R_hat"] / t["T_hat"]
    t["S"] = t["R"] / t["T"]
    t["BASE_P3_POOL"] = t["R_hat"]
    t["ORACLE_TOTAL_RUSH"] = t["T"] * t["S_hat"]
    t["ORACLE_RB_SHARE"] = t["T_hat"] * t["S"]
    t["ORACLE_BOTH"] = t["T"] * t["S"]
    t["base_residual"] = t["BASE_P3_POOL"] - t["R"]
    t["base_abs_residual"] = t["base_residual"].abs()

    r = t["base_residual"]
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


base.build_trace = corrected_build_trace

if __name__ == "__main__":
    raise SystemExit(base.main())
