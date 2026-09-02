#!/usr/bin/env python3
"""STACK6F scored-rerun wrapper correcting only pandas Series metric access."""
from __future__ import annotations

import pandas as pd

from scripts.backtest import evaluate_rb_stack6f_team_pool as base


def retention(scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    def r(scope, arm):
        q = scores.loc[scores.scope.eq(scope) & scores.arm.eq(arm)]
        if q.empty:
            raise RuntimeError(f"STACK6F missing score {scope}/{arm}")
        return q.iloc[0]

    parent = r("w6_18", "P3_PARENT_POOL")
    late_parent = r("w13_18", "P3_PARENT_POOL")
    rows = []
    for arm in base.ARMS:
        cur = r("w6_18", arm)
        late = r("w13_18", arm)
        mae_gain = float(parent["mae"] - cur["mae"])
        rmse_gain = float(parent["rmse"] - cur["rmse"])
        corr_gain = float(cur["corr"] - parent["corr"])
        late_gain = float(late_parent["mae"] - late["mae"])
        abs_bias = abs(float(cur["bias"]))
        passed = int(
            mae_gain >= 0.30
            and rmse_gain > 0
            and corr_gain >= 0.05
            and abs_bias <= 0.50
            and late_gain > 0
        )
        rows.append({
            "arm": arm,
            "team_carry_mae_gain": mae_gain,
            "team_carry_rmse_gain": rmse_gain,
            "team_carry_corr_gain": corr_gain,
            "team_carry_abs_bias": abs_bias,
            "late_team_carry_mae_gain": late_gain,
            "gate_mae_gain_ge_030": int(mae_gain >= 0.30),
            "gate_rmse_gain_gt_0": int(rmse_gain > 0),
            "gate_corr_gain_ge_005": int(corr_gain >= 0.05),
            "gate_abs_bias_le_050": int(abs_bias <= 0.50),
            "gate_late_gain_gt_0": int(late_gain > 0),
            "gate_pass": passed,
        })
    g = pd.DataFrame(rows)
    passing = g.loc[g.gate_pass.eq(1)].copy()
    selected = "NONE"
    if len(passing):
        best = float(passing.team_carry_mae_gain.max())
        blend = passing.loc[passing.arm.eq("P3_HISTORY_50")]
        if len(blend) and float(blend.iloc[0].team_carry_mae_gain) >= best - 0.10:
            selected = "P3_HISTORY_50"
        else:
            selected = str(passing.sort_values(["team_carry_mae_gain", "arm"], ascending=[False, True]).iloc[0].arm)
    g["selected_arm"] = selected
    return g, selected


def main() -> int:
    base.retention = retention
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
