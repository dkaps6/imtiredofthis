#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ND5_ROWS = 2130
EXPECTED_PROFILES = 133
EXPECTED_TARGET_PLAYERS = 72

SIGNALS = {
    "SNAP_LEVEL_PRIOR1": ("snap_level_prior1", "quartile"),
    "SNAP_ACCEL_1V4": ("snap_accel_1v4", "quartile"),
    "DEPTH_TOP2_STATE": ("depth_top2_state", "binary"),
    "DEPTH_RANK_PROMOTION": ("depth_rank_promotion", "positive_vs_nonpositive"),
}


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def spearman(a, b) -> float:
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan
    return float(z.a.corr(z.b, method="spearman"))


def masks(x: pd.DataFrame, col: str, mode: str) -> tuple[pd.Series, pd.Series, pd.Series, float, float]:
    v = num(x[col])
    valid = v.notna()
    if mode == "quartile":
        q25 = float(v.loc[valid].quantile(.25)) if valid.any() else np.nan
        q75 = float(v.loc[valid].quantile(.75)) if valid.any() else np.nan
        hi = valid & v.ge(q75)
        lo = valid & v.le(q25)
        return hi, lo, valid, q25, q75
    if mode == "binary":
        hi = valid & v.eq(1.0)
        lo = valid & v.eq(0.0)
        return hi, lo, valid, 0.0, 1.0
    if mode == "positive_vs_nonpositive":
        hi = valid & v.gt(0.0)
        lo = valid & v.le(0.0)
        return hi, lo, valid, 0.0, 0.0
    raise RuntimeError(mode)


def gap(x: pd.DataFrame, hi: pd.Series, lo: pd.Series, outcome: str) -> float:
    if int(hi.sum()) == 0 or int(lo.sum()) == 0:
        return np.nan
    return float(num(x.loc[hi, outcome]).mean() - num(x.loc[lo, outcome]).mean())


def slice_gap(x: pd.DataFrame, col: str, mode: str, q25: float, q75: float, mask: pd.Series) -> float:
    z = x.loc[mask].copy()
    v = num(z[col]); valid = v.notna()
    if mode == "quartile":
        hi = valid & v.ge(q75); lo = valid & v.le(q25)
    elif mode == "binary":
        hi = valid & v.eq(1.0); lo = valid & v.eq(0.0)
    else:
        hi = valid & v.gt(0.0); lo = valid & v.le(0.0)
    return gap(z, hi, lo, "allocation_residual")


def player_consistency(x: pd.DataFrame, col: str) -> tuple[int, int, float]:
    eligible = 0
    computable = 0
    positive = 0
    for _, g in x.groupby("player_clean_key", sort=False):
        z = g.loc[num(g[col]).notna() & num(g["allocation_residual"]).notna()].copy()
        if len(z) < 8:
            continue
        eligible += 1
        r = spearman(z[col], z["allocation_residual"])
        if np.isfinite(r):
            computable += 1
            positive += int(r > 0)
    rate = float(positive / computable) if computable else np.nan
    return eligible, computable, rate


def score(x: pd.DataFrame, name: str, col: str, mode: str) -> dict:
    hi, lo, valid, q25, q75 = masks(x, col, mode)
    alloc_gap = gap(x, hi, lo, "allocation_residual")
    target_gap = gap(x, hi, lo, "raw_target_error")
    tail = x["entitlement_miss_tail"].astype(bool)
    overall_tail = float(tail.loc[valid].mean()) if int(valid.sum()) else np.nan
    hi_tail = float(tail.loc[hi].mean()) if int(hi.sum()) else np.nan
    enrich = float(hi_tail / overall_tail) if np.isfinite(hi_tail) and np.isfinite(overall_tail) and overall_tail > 0 else np.nan
    w2 = slice_gap(x, col, mode, q25, q75, x["week"].ge(2))
    w13 = slice_gap(x, col, mode, q25, q75, x["week"].ge(13))
    p8, pcomp, prate = player_consistency(x, col)
    r = spearman(x.loc[valid, col], x.loc[valid, "allocation_residual"])
    gates = {
        "conditioned_n_ge_600": bool(len(x) >= 600),
        "coverage_ge_0_80": bool(float(valid.mean()) >= .80),
        "spearman_ge_0_10": bool(np.isfinite(r) and r >= .10),
        "allocation_gap_ge_0_030": bool(np.isfinite(alloc_gap) and alloc_gap >= .030),
        "raw_target_gap_ge_1_00": bool(np.isfinite(target_gap) and target_gap >= 1.00),
        "tail_enrichment_ge_1_20": bool(np.isfinite(enrich) and enrich >= 1.20),
        "w2_18_gap_positive": bool(np.isfinite(w2) and w2 > 0),
        "w13_18_gap_positive": bool(np.isfinite(w13) and w13 > 0),
        "players_8plus_valid_ge_20": bool(p8 >= 20),
        "positive_within_player_rate_ge_0_58": bool(np.isfinite(prate) and prate >= .58),
    }
    return {
        "signal": name,
        "column": col,
        "mode": mode,
        "conditioned_n": int(len(x)),
        "valid_n": int(valid.sum()),
        "coverage": float(valid.mean()),
        "threshold_low": q25,
        "threshold_high": q75,
        "spearman_allocation_residual": r,
        "high_low_allocation_residual_gap": alloc_gap,
        "high_low_raw_target_error_gap": target_gap,
        "tail_enrichment": enrich,
        "w2_18_gap": w2,
        "w13_18_gap": w13,
        "players_with_8plus_valid": int(p8),
        "players_with_8plus_computable_association": int(pcomp),
        "positive_within_player_association_rate": prate,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-r5-root", type=Path, required=True)
    ap.add_argument("--nd5-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    profiles = pd.read_csv(one(a.wr_r5_root, "wr_r5_individual_mechanisms.csv"), low_memory=False)
    case = pd.read_csv(one(a.nd5_root, "wr_nd5_casebook.csv"), low_memory=False)
    profiles.columns = [str(c).strip().lower() for c in profiles.columns]
    case.columns = [str(c).strip().lower() for c in case.columns]

    if len(profiles) != EXPECTED_PROFILES:
        raise RuntimeError(f"WR-R5 profile drift expected={EXPECTED_PROFILES} got={len(profiles)}")
    if len(case) != EXPECTED_ND5_ROWS:
        raise RuntimeError(f"ND5 row drift expected={EXPECTED_ND5_ROWS} got={len(case)}")
    need = {"player_clean_key", "week", "allocation_residual", "raw_target_error", "entitlement_miss_tail"} | {c for c,_ in SIGNALS.values()}
    miss = need - set(case.columns)
    if miss:
        raise RuntimeError(f"ND5 casebook missing {sorted(miss)}")
    if not {"player_key", "dominant_mechanism"}.issubset(profiles.columns):
        raise RuntimeError("WR-R5 profile missing player_key/dominant_mechanism")

    profiles["_key"] = profiles["player_key"].map(key)
    case["_key"] = case["player_clean_key"].map(key)
    target_keys = set(profiles.loc[profiles["dominant_mechanism"].astype(str).str.upper().eq("TARGETS"), "_key"])
    if len(target_keys) != EXPECTED_TARGET_PLAYERS:
        raise RuntimeError(f"TARGETS-dominant player drift expected={EXPECTED_TARGET_PLAYERS} got={len(target_keys)}")
    x = case.loc[case["_key"].isin(target_keys)].copy()
    matched_players = x["_key"].nunique()
    if matched_players != EXPECTED_TARGET_PLAYERS:
        missing = sorted(target_keys - set(x["_key"]))
        raise RuntimeError(f"TARGETS-dominant ND5 join failed players={matched_players} missing={missing[:20]}")

    rows = [score(x, name, col, mode) for name, (col, mode) in SIGNALS.items()]
    passed = [r["signal"] for r in rows if r["passed"]]
    disposition = "WR_TARGET_DOMINANT_ROLE_SIGNAL_DISCOVERY_PASS" if passed else "NO_ACTIONABLE_WR_TARGET_DOMINANT_ROLE_SIGNAL"
    result = {
        "migration": "WR_R8_TARGET_DOMINANT_ROLE_SIGNALS",
        "source_nd5_rows": int(len(case)),
        "source_wr_r5_profiles": int(len(profiles)),
        "target_dominant_players": int(len(target_keys)),
        "conditioned_rows": int(len(x)),
        "passing_signals": passed,
        "signals": {r["signal"]: {k:v for k,v in r.items() if k != "signal"} for r in rows},
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.out_dir / "wr_r8_target_dominant_casebook.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k != "gates"} for r in rows]).to_csv(a.out_dir / "wr_r8_signal_summary.csv", index=False)
    (a.out_dir / "wr_r8_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(pd.DataFrame([{k:v for k,v in r.items() if k not in {"gates"}} for r in rows]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
