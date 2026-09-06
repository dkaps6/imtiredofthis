#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES = [
    "state_depth_vs_carry_order_mismatch",
    "state_limited_prior_history",
    "state_no_prior_same_team_game",
    "state_rookie",
    "state_injury_created_context",
]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, got {len(hits)}")
    return hits[0]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)


def summarize(g: pd.DataFrame) -> dict:
    carry_resid = num(g["actual_att"]) - num(g["pred_att"])
    yard_resid = num(g["actual_yards"]) - num(g["pred_yards"])
    return {
        "rows": int(len(g)),
        "carry_mae": float(carry_resid.abs().mean()) if len(g) else np.nan,
        "carry_component_abs": float(num(g["carry_component"]).abs().mean()) if len(g) else np.nan,
        "carry_residual_actual_minus_pred": float(carry_resid.mean()) if len(g) else np.nan,
        "rush_yard_mae": float(yard_resid.abs().mean()) if len(g) else np.nan,
        "yard_residual_abs": float(yard_resid.abs().mean()) if len(g) else np.nan,
    }


def evaluate_cohort(x: pd.DataFrame, mechanism: str) -> tuple[pd.DataFrame, dict]:
    g0 = x.loc[x["dominant_mechanism"].eq(mechanism)].copy()
    rows = []
    ratios = {}
    for state in STATES:
        if state not in g0.columns:
            raise RuntimeError(f"missing frozen state {state}")
        s1 = g0.loc[num(g0[state]).eq(1)]
        s0 = g0.loc[num(g0[state]).eq(0)]
        m1 = summarize(s1)
        m0 = summarize(s0)
        r = {
            "carry_mae_ratio": safe_ratio(m1["carry_mae"], m0["carry_mae"]),
            "carry_component_abs_ratio": safe_ratio(m1["carry_component_abs"], m0["carry_component_abs"]),
            "rush_yard_mae_ratio": safe_ratio(m1["rush_yard_mae"], m0["rush_yard_mae"]),
            "abs_carry_residual_mean_difference": abs(float(m1["carry_residual_actual_minus_pred"] - m0["carry_residual_actual_minus_pred"])) if np.isfinite(m1["carry_residual_actual_minus_pred"]) and np.isfinite(m0["carry_residual_actual_minus_pred"]) else np.nan,
        }
        ratios[state] = r
        for value, m in [(1, m1), (0, m0)]:
            rows.append({"mechanism": mechanism, "state": state, "state_value": value, **m})
    return pd.DataFrame(rows), {"rows": int(len(g0)), "players": int(g0["player_key"].nunique()), "ratios": ratios}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    case = pd.read_csv(one(a.rb_root, "rb_mechanism_casebook.csv"), low_memory=False)
    prof = pd.read_csv(one(a.rb_root, "rb_individual_mechanisms.csv"), low_memory=False)
    case.columns = [str(c).strip().lower() for c in case.columns]
    prof.columns = [str(c).strip().lower() for c in prof.columns]
    if len(case) != 1393:
        raise RuntimeError(f"casebook row drift {len(case)}")
    if prof["player_key"].duplicated().any():
        raise RuntimeError("duplicate player profile keys")

    x = case.merge(prof[["player_key", "dominant_mechanism"]], on="player_key", how="left", validate="many_to_one")
    if x["dominant_mechanism"].isna().all():
        raise RuntimeError("no mechanism profiles merged")

    carry_table, carry = evaluate_cohort(x, "CARRIES")
    ypc_table, ypc = evaluate_cohort(x, "YPC")
    table = pd.concat([carry_table, ypc_table], ignore_index=True)

    mismatch = carry["ratios"]["state_depth_vs_carry_order_mismatch"]
    comp15 = sum(
        int(np.isfinite(v["carry_component_abs_ratio"]) and v["carry_component_abs_ratio"] >= 1.15)
        for v in carry["ratios"].values()
    )
    yards10 = sum(
        int(np.isfinite(v["rush_yard_mae_ratio"]) and v["rush_yard_mae_ratio"] >= 1.10)
        for v in carry["ratios"].values()
    )
    gates = {
        "carry_rows_ge250": bool(carry["rows"] >= 250),
        "mismatch_carry_component_ratio_ge1_20": bool(mismatch["carry_component_abs_ratio"] >= 1.20),
        "mismatch_carry_mae_ratio_ge1_15": bool(mismatch["carry_mae_ratio"] >= 1.15),
        "three_of_five_component_ratios_ge1_15": bool(comp15 >= 3),
        "three_of_five_yard_mae_ratios_ge1_10": bool(yards10 >= 3),
    }
    disposition = "ROLE_CONTEXT_CONCENTRATES_CARRY_MECHANISM_ERROR" if all(gates.values()) else "NO_STRONG_ROLE_CONTEXT_CONCENTRATION_FOR_CARRY_ERRORS"

    result = {
        "migration": "RB_CARRY_MECHANISM_ROLE_CONTEXT_CONCENTRATION",
        "casebook_rows": int(len(case)),
        "profile_rows": int(len(prof)),
        "carry_dominant": carry,
        "ypc_dominant_secondary": ypc,
        "carry_states_component_ratio_ge1_15_count": int(comp15),
        "carry_states_yard_mae_ratio_ge1_10_count": int(yards10),
        "gates": gates,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out_dir / "rb_role_context_by_mechanism.csv", index=False)
    x.to_csv(a.out_dir / "rb_role_context_mechanism_casebook.csv", index=False)
    (a.out_dir / "rb_carry_role_context_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
