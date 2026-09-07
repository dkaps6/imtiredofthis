#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 1393
EXPECTED_PROFILES = 86
EXPECTED_ALLOCATION_PLAYERS = 17
EXPECTED_CONDITIONED_ROWS = 212

SIGNALS = [
    "prior1_room_carry_share",
    "room_carry_share_accel_1v4",
    "prior1_carries",
    "carries_accel_1v4",
]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def corr(a, b) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method="spearman"))


def player_consistency(x: pd.DataFrame, signal: str) -> tuple[int, int, float]:
    eligible = computable = positive = 0
    for _, g in x.groupby("player_key", sort=False):
        z = g.loc[num(g[signal]).notna() & num(g["individual_allocation_component"]).notna()].copy()
        if len(z) < 6:
            continue
        eligible += 1
        r = corr(z[signal], z["individual_allocation_component"])
        if np.isfinite(r):
            computable += 1
            positive += int(r > 0)
    rate = float(positive / computable) if computable else np.nan
    return eligible, computable, rate


def score(x: pd.DataFrame, signal: str) -> dict:
    v = num(x[signal])
    valid = v.notna() & num(x["individual_allocation_component"]).notna()
    q25 = float(v.loc[valid].quantile(.25)) if valid.any() else np.nan
    q75 = float(v.loc[valid].quantile(.75)) if valid.any() else np.nan
    hi = valid & v.ge(q75)
    lo = valid & v.le(q25)

    def gap(outcome: str, mask: pd.Series | None = None) -> float:
        use = pd.Series(True, index=x.index) if mask is None else mask
        h = hi & use; l = lo & use
        if int(h.sum()) == 0 or int(l.sum()) == 0:
            return np.nan
        return float(num(x.loc[h, outcome]).mean() - num(x.loc[l, outcome]).mean())

    alloc_gap = gap("individual_allocation_component")
    carry_gap = gap("carry_residual_actual_minus_pred")
    w2 = gap("individual_allocation_component", x["week"].between(2,18))
    w13 = gap("individual_allocation_component", x["week"].between(13,18))
    r = corr(x.loc[valid, signal], x.loc[valid, "individual_allocation_component"])
    tail = num(x["individual_allocation_component"]).ge(3.0)
    overall_tail = float(tail.loc[valid].mean()) if int(valid.sum()) else np.nan
    hi_tail = float(tail.loc[hi].mean()) if int(hi.sum()) else np.nan
    enrich = float(hi_tail / overall_tail) if np.isfinite(hi_tail) and np.isfinite(overall_tail) and overall_tail > 0 else np.nan
    p6, pcomp, prate = player_consistency(x, signal)

    gates = {
        "valid_n_ge_150": bool(int(valid.sum()) >= 150),
        "coverage_ge_0_70": bool(float(valid.mean()) >= .70),
        "spearman_ge_0_10": bool(np.isfinite(r) and r >= .10),
        "allocation_gap_ge_1_00": bool(np.isfinite(alloc_gap) and alloc_gap >= 1.00),
        "carry_residual_gap_ge_1_50": bool(np.isfinite(carry_gap) and carry_gap >= 1.50),
        "tail_enrichment_ge_1_20": bool(np.isfinite(enrich) and enrich >= 1.20),
        "w2_18_gap_positive": bool(np.isfinite(w2) and w2 > 0),
        "w13_18_gap_positive": bool(np.isfinite(w13) and w13 > 0),
        "players_6plus_valid_ge_10": bool(p6 >= 10),
        "positive_within_player_rate_ge_0_60": bool(np.isfinite(prate) and prate >= .60),
    }
    return {
        "signal": signal,
        "conditioned_n": int(len(x)),
        "valid_n": int(valid.sum()),
        "coverage": float(valid.mean()),
        "q25": q25,
        "q75": q75,
        "spearman_allocation_component": r,
        "q4_q1_allocation_component_gap": alloc_gap,
        "q4_q1_carry_residual_gap": carry_gap,
        "underallocation_tail_enrichment": enrich,
        "w2_18_allocation_gap": w2,
        "w13_18_allocation_gap": w13,
        "players_with_6plus_valid": int(p6),
        "players_with_6plus_computable": int(pcomp),
        "positive_within_player_association_rate": prate,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    case = pd.read_csv(one(a.rb_r1_root, "rb_r1_room_allocation_casebook.csv"), low_memory=False)
    prof = pd.read_csv(one(a.rb_r1_root, "rb_r1_player_submechanisms.csv"), low_memory=False)
    case.columns = [str(c).strip().lower() for c in case.columns]
    prof.columns = [str(c).strip().lower() for c in prof.columns]
    if len(case) != EXPECTED_ROWS:
        raise RuntimeError(f"RB-R1 row drift expected={EXPECTED_ROWS} got={len(case)}")
    if len(prof) != EXPECTED_PROFILES:
        raise RuntimeError(f"RB-R1 profile drift expected={EXPECTED_PROFILES} got={len(prof)}")

    need = {
        "season", "week", "team", "player_key", "actual_att", "actual_room_att",
        "actual_room_share", "individual_allocation_component", "carry_residual_actual_minus_pred",
    }
    miss = need - set(case.columns)
    if miss:
        raise RuntimeError(f"RB-R1 casebook missing {sorted(miss)}")
    if not {"player_key", "parent_dominant_mechanism", "carry_submechanism"}.issubset(prof.columns):
        raise RuntimeError("RB-R1 profiles missing mechanism fields")

    allocation_players = set(prof.loc[
        prof["parent_dominant_mechanism"].astype(str).eq("CARRIES")
        & prof["carry_submechanism"].astype(str).eq("INDIVIDUAL_ALLOCATION"),
        "player_key"
    ].astype(str))
    if len(allocation_players) != EXPECTED_ALLOCATION_PLAYERS:
        raise RuntimeError(f"allocation-player drift expected={EXPECTED_ALLOCATION_PLAYERS} got={len(allocation_players)}")

    x = case.copy()
    for c in ["season", "week", "actual_att", "actual_room_att", "actual_room_share", "individual_allocation_component", "carry_residual_actual_minus_pred"]:
        x[c] = num(x[c])
    x["player_key"] = x["player_key"].astype(str)
    x["team"] = x["team"].astype(str)
    x = x.sort_values(["player_key", "team", "season", "week"], kind="stable").reset_index(drop=True)

    gp = x.groupby(["player_key", "team"], sort=False)
    x["prior1_room_carry_share"] = gp["actual_room_share"].shift(1)
    share_hist = gp["actual_room_share"].shift(1)
    x["prior4_room_carry_share_mean"] = share_hist.groupby([x["player_key"], x["team"]]).transform(
        lambda s: s.rolling(4, min_periods=3).mean()
    )
    x["room_carry_share_accel_1v4"] = x["prior1_room_carry_share"] - x["prior4_room_carry_share_mean"]

    x["prior1_carries"] = gp["actual_att"].shift(1)
    carry_hist = gp["actual_att"].shift(1)
    x["prior4_carries_mean"] = carry_hist.groupby([x["player_key"], x["team"]]).transform(
        lambda s: s.rolling(4, min_periods=3).mean()
    )
    x["carries_accel_1v4"] = x["prior1_carries"] - x["prior4_carries_mean"]

    cond = x.loc[x["player_key"].isin(allocation_players)].copy()
    if len(cond) != EXPECTED_CONDITIONED_ROWS:
        raise RuntimeError(f"conditioned row drift expected={EXPECTED_CONDITIONED_ROWS} got={len(cond)}")
    if cond["player_key"].nunique() != EXPECTED_ALLOCATION_PLAYERS:
        raise RuntimeError("allocation-player conditioned join incomplete")

    rows = [score(cond, s) for s in SIGNALS]
    passed = [r["signal"] for r in rows if r["passed"]]
    disposition = "RB_DYNAMIC_WORKLOAD_ALLOCATION_DISCOVERY_PASS" if passed else "NO_ACTIONABLE_RB_DYNAMIC_WORKLOAD_ALLOCATION_SIGNAL"
    result = {
        "migration": "RB_R3_DYNAMIC_WORKLOAD_ALLOCATION",
        "source_rows": int(len(case)),
        "source_profiles": int(len(prof)),
        "allocation_dominant_players": int(len(allocation_players)),
        "conditioned_rows": int(len(cond)),
        "passing_signals": passed,
        "signals": {r["signal"]: {k:v for k,v in r.items() if k != "signal"} for r in rows},
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cond.to_csv(a.out_dir / "rb_r3_dynamic_workload_casebook.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k != "gates"} for r in rows]).to_csv(a.out_dir / "rb_r3_signal_summary.csv", index=False)
    (a.out_dir / "rb_r3_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(pd.DataFrame([{k:v for k,v in r.items() if k != "gates"} for r in rows]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
