#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES = [
    "state_depth_vs_carry_order_mismatch",
    "state_injury_created_context",
    "state_no_prior_same_team_game",
    "state_rookie",
    "state_limited_prior_history",
]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(v):
    return pd.to_numeric(v, errors="coerce")


def ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def state_ratio(g: pd.DataFrame, state: str) -> float:
    one_mask = num(g[state]).eq(1)
    zero_mask = num(g[state]).eq(0)
    if not one_mask.any() or not zero_mask.any():
        return np.nan
    a = float(g.loc[one_mask, "individual_allocation_component"].abs().mean())
    b = float(g.loc[zero_mask, "individual_allocation_component"].abs().mean())
    return ratio(a,b)


def score_state(g: pd.DataFrame, state: str) -> dict:
    s = num(g[state])
    pos = g.loc[s.eq(1)].copy()
    neg = g.loc[s.eq(0)].copy()
    pos_alloc = float(pos.individual_allocation_component.abs().mean()) if len(pos) else np.nan
    neg_alloc = float(neg.individual_allocation_component.abs().mean()) if len(neg) else np.nan
    pos_mae = float(pos.carry_residual_actual_minus_pred.abs().mean()) if len(pos) else np.nan
    neg_mae = float(neg.carry_residual_actual_minus_pred.abs().mean()) if len(neg) else np.nan
    alloc_ratio = ratio(pos_alloc, neg_alloc)
    mae_ratio = ratio(pos_mae, neg_mae)
    alloc_diff = float(pos_alloc-neg_alloc) if np.isfinite(pos_alloc) and np.isfinite(neg_alloc) else np.nan
    w2_ratio = state_ratio(g.loc[num(g.week).between(2,18)], state)
    w13_ratio = state_ratio(g.loc[num(g.week).between(13,18)], state)
    gates = {
        "state1_n_ge_25": bool(len(pos) >= 25),
        "state0_n_ge_50": bool(len(neg) >= 50),
        "allocation_ratio_ge_1_25": bool(np.isfinite(alloc_ratio) and alloc_ratio >= 1.25),
        "carry_mae_ratio_ge_1_15": bool(np.isfinite(mae_ratio) and mae_ratio >= 1.15),
        "allocation_abs_difference_ge_0_75": bool(np.isfinite(alloc_diff) and alloc_diff >= 0.75),
        "w2_18_allocation_ratio_gt_1": bool(np.isfinite(w2_ratio) and w2_ratio > 1.0),
        "w13_18_allocation_ratio_gt_1": bool(np.isfinite(w13_ratio) and w13_ratio > 1.0),
    }
    return {
        "state": state,
        "state1_n": int(len(pos)),
        "state0_n": int(len(neg)),
        "state1_allocation_component_abs": pos_alloc,
        "state0_allocation_component_abs": neg_alloc,
        "allocation_component_ratio": alloc_ratio,
        "allocation_component_abs_difference": alloc_diff,
        "state1_carry_mae": pos_mae,
        "state0_carry_mae": neg_mae,
        "carry_mae_ratio": mae_ratio,
        "state1_carry_bias_actual_minus_pred": float(pos.carry_residual_actual_minus_pred.mean()) if len(pos) else np.nan,
        "state0_carry_bias_actual_minus_pred": float(neg.carry_residual_actual_minus_pred.mean()) if len(neg) else np.nan,
        "w2_18_allocation_component_ratio": w2_ratio,
        "w13_18_allocation_component_ratio": w13_ratio,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    x = pd.read_csv(one(a.rb_r1_root, "rb_r1_room_allocation_casebook.csv"), low_memory=False)
    p = pd.read_csv(one(a.rb_r1_root, "rb_r1_player_submechanisms.csv"), low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    p.columns = [str(c).strip().lower() for c in p.columns]
    if len(x) != 1393:
        raise RuntimeError(f"RB-R1 casebook drift expected=1393 got={len(x)}")
    if len(p) != 86:
        raise RuntimeError(f"RB-R1 profile drift expected=86 got={len(p)}")
    need = set(["player_key","week","individual_allocation_component","carry_residual_actual_minus_pred"] + STATES)
    if need-set(x.columns):
        raise RuntimeError(f"casebook missing {sorted(need-set(x.columns))}")
    needp = {"player_key","parent_dominant_mechanism","carry_submechanism"}
    if needp-set(p.columns):
        raise RuntimeError(f"profile missing {sorted(needp-set(p.columns))}")

    alloc_players = p.loc[
        p.parent_dominant_mechanism.astype(str).eq("CARRIES")
        & p.carry_submechanism.astype(str).eq("INDIVIDUAL_ALLOCATION"),
        "player_key"
    ].astype(str).tolist()
    if len(alloc_players) != 17:
        raise RuntimeError(f"allocation-dominant player drift expected=17 got={len(alloc_players)}")

    g = x.loc[x.player_key.astype(str).isin(set(alloc_players))].copy()
    if g.empty:
        raise RuntimeError("empty conditioned RB-R2 population")
    for c in ["individual_allocation_component","carry_residual_actual_minus_pred","week"] + STATES:
        g[c] = num(g[c])

    rows = [score_state(g, s) for s in STATES]
    primary = rows[0]
    secondary_passes = [r for r in rows[1:] if r["passed"]]
    if primary["passed"]:
        disposition = "RB_ALLOCATION_SUBGROUP_DEPTH_SIGNAL_PASS"
    elif len(secondary_passes) >= 2:
        disposition = "RB_ALLOCATION_SUBGROUP_TRANSITION_SIGNAL_PASS"
    else:
        disposition = "NO_ACTIONABLE_RB_ALLOCATION_SUBGROUP_ROLE_CONTEXT_SIGNAL"

    result = {
        "migration": "RB_R2_ALLOCATION_SUBGROUP_ROLE_CONTEXT",
        "source_rows": int(len(x)),
        "source_profiles": int(len(p)),
        "allocation_dominant_players": int(len(alloc_players)),
        "conditioned_rows": int(len(g)),
        "primary_state_passed": bool(primary["passed"]),
        "secondary_states_passed": [r["state"] for r in secondary_passes],
        "states": {r["state"]: {k:v for k,v in r.items() if k!="state"} for r in rows},
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.out_dir / "rb_r2_allocation_conditioned_casebook.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!="gates"} for r in rows]).to_csv(a.out_dir / "rb_r2_state_summary.csv", index=False)
    (a.out_dir / "rb_r2_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(pd.DataFrame([{k:v for k,v in r.items() if k!="gates"} for r in rows]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
