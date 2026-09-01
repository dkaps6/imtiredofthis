#!/usr/bin/env python3
"""RB-STACK1 production-equivalent historical baseline audit.

No new football model is fit here except the already-canonical market-specific
nonnegative ensemble weights, learned only from prior OOS component predictions.
Sportsbook data, when supplied, is joined only after football projections are
frozen and is used for downstream benchmarking only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import fit_market_weights, apply_ensemble
from scripts.utils.canonical_names import canonicalize_player_name_safe

RB_POS = {"RB", "FB", "HB"}
MARKETS = {"rush_att", "rush_yards"}
COMPONENTS = ["mc_proj", "ml_proj", "state_proj"]


def _key(v) -> str:
    try:
        _, k = canonicalize_player_name_safe(v)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(c.lower() for c in str(v or "") if c.isalnum())


def read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing input: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def metric_row(df: pd.DataFrame, pred_col: str, arm: str, market: str, slice_name: str) -> dict:
    x = df.copy()
    y = pd.to_numeric(x.get("actual"), errors="coerce")
    p = pd.to_numeric(x.get(pred_col), errors="coerce")
    ok = y.notna() & p.notna()
    y, p = y[ok].astype(float), p[ok].astype(float)
    if len(y) == 0:
        return {"market": market, "slice": slice_name, "arm": arm, "n": 0}
    err = p - y
    return {
        "market": market, "slice": slice_name, "arm": arm, "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "corr": float(np.corrcoef(p, y)[0, 1]) if len(y) > 1 and p.std() > 0 and y.std() > 0 else np.nan,
        "actual_mean": float(y.mean()), "pred_mean": float(p.mean()),
    }


def prep_components(x: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "week", "team", "player", "position", "market", "actual", *COMPONENTS}
    miss = required - set(x.columns)
    if miss:
        raise RuntimeError(f"component predictions missing columns: {sorted(miss)}")
    out = x.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["position"] = out["position"].fillna("").astype(str).str.upper().str.strip()
    out["market"] = out["market"].fillna("").astype(str).str.lower().str.strip()
    out["player_clean_key"] = out.get("player_clean_key", out["player"]).map(_key)
    return out


def build_ensembles(c24: pd.DataFrame, c25: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Canonical market-specific all-position weights, frozen from 2024 OOS.
    w24 = fit_market_weights(c24)
    frozen = apply_ensemble(c25, weights=w24).rename(columns={
        "ensemble_proj": "ensemble_2024_frozen",
        "ensemble_status": "ensemble_2024_status",
        "ensemble_weight_mc": "ensemble_2024_w_mc",
        "ensemble_weight_ml": "ensemble_2024_w_ml",
        "ensemble_weight_state": "ensemble_2024_w_state",
    })

    # Leakage-safe weekly adaptive diagnostic: 2024 + prior 2025 weeks only.
    chunks, weight_rows = [], []
    for week in sorted(pd.to_numeric(c25["week"], errors="coerce").dropna().astype(int).unique()):
        train = pd.concat([c24, c25.loc[pd.to_numeric(c25["week"], errors="coerce") < week]], ignore_index=True)
        w = fit_market_weights(train)
        for _, r in w.iterrows():
            rec = r.to_dict(); rec["target_week"] = int(week); weight_rows.append(rec)
        target = c25.loc[pd.to_numeric(c25["week"], errors="coerce").eq(week)].copy()
        scored = apply_ensemble(target, weights=w).rename(columns={
            "ensemble_proj": "ensemble_expanding",
            "ensemble_status": "ensemble_expanding_status",
            "ensemble_weight_mc": "ensemble_expanding_w_mc",
            "ensemble_weight_ml": "ensemble_expanding_w_ml",
            "ensemble_weight_state": "ensemble_expanding_w_state",
        })
        chunks.append(scored)
    expanding = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    # Merge adaptive columns onto the frozen frame by stable row identity.
    keys = ["season", "week", "team", "player_clean_key", "market"]
    ecols = keys + [c for c in expanding.columns if c.startswith("ensemble_expanding")]
    frozen = frozen.merge(expanding[ecols].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    return frozen, w24, pd.DataFrame(weight_rows)


def add_m94c(rb: pd.DataFrame, m94: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = m94.copy()
    if "player_clean_key" not in m.columns:
        m["player_clean_key"] = m["player"].map(_key)
    m["week"] = pd.to_numeric(m["week"], errors="coerce")
    m["team"] = m["team"].astype(str)
    keys = ["week", "team", "player_clean_key"]
    cols = keys + [c for c in ["candidate_rush_att", "candidate_rush_yards", "actual_rush_att", "actual_rush_yards"] if c in m.columns]
    m = m[cols].drop_duplicates(keys)
    out = rb.merge(m, on=keys, how="left", validate="many_to_one")
    out["m94c_proj"] = np.where(out["market"].eq("rush_att"), pd.to_numeric(out.get("candidate_rush_att"), errors="coerce"), pd.to_numeric(out.get("candidate_rush_yards"), errors="coerce"))
    audit = pd.DataFrame([{
        "rb_component_rows": len(rb), "m94c_trace_rows": len(m),
        "m94c_matched_rows": int(out["m94c_proj"].notna().sum()),
        "m94c_match_rate": float(out["m94c_proj"].notna().mean()) if len(out) else np.nan,
    }])
    return out, audit


def context_coverage(rb: pd.DataFrame) -> pd.DataFrame:
    # One row per player-week; context fields repeat by market.
    keys = ["season", "week", "team", "player_clean_key"]
    x = rb.sort_values(keys).drop_duplicates(keys)
    rows = []
    fields = [
        "rules_applied", "role", "rules_role", "ctx_rush_share_available",
        "ctx_success_rate_available", "ctx_pace_available", "ctx_proe_available",
        "ctx_pressure_available", "ctx_explosive_available", "ctx_def_epa_available",
        "ctx_box_rates_available", "ctx_injury_available", "ctx_weather_available",
    ]
    for f in fields:
        if f not in x.columns:
            rows.append({"field": f, "rows": len(x), "available": 0, "coverage": 0.0, "status": "missing_column"})
            continue
        s = x[f]
        if f in {"role", "rules_role"}:
            avail = s.fillna("").astype(str).str.strip().ne("")
        elif f == "rules_applied" or f.startswith("ctx_"):
            avail = pd.to_numeric(s, errors="coerce").fillna(0).gt(0)
        else:
            avail = s.notna()
        rows.append({"field": f, "rows": len(x), "available": int(avail.sum()), "coverage": float(avail.mean()), "status": "present"})
    return pd.DataFrame(rows)


def make_slices(rb: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    week = pd.to_numeric(rb["week"], errors="coerce")
    actual_c = np.where(rb["market"].eq("rush_att"), pd.to_numeric(rb["actual"], errors="coerce"), pd.to_numeric(rb.get("actual_rush_att"), errors="coerce"))
    actual_c = pd.Series(actual_c, index=rb.index, dtype=float)
    return [
        ("all_rb", pd.Series(True, index=rb.index)),
        ("week1", week.eq(1)), ("weeks2_18", week.ge(2)),
        ("actual_carries_0_5", actual_c.le(5)),
        ("actual_carries_6_10", actual_c.between(6, 10, inclusive="both")),
        ("actual_carries_11_14", actual_c.between(11, 14, inclusive="both")),
        ("actual_carries_15_19", actual_c.between(15, 19, inclusive="both")),
        ("actual_carries_20_plus", actual_c.ge(20)),
        ("actual_carries_25_plus", actual_c.ge(25)),
    ]


def listed_market_benchmark(trace: pd.DataFrame, casebook: pd.DataFrame) -> pd.DataFrame:
    cb = casebook.copy()
    cb["player_clean_key"] = cb["player"].map(_key)
    cb["week"] = pd.to_numeric(cb["week"], errors="coerce")
    cb["team"] = cb["team"].astype(str)
    keys = ["week", "team", "player_clean_key"]
    y = trace.loc[trace["market"].eq("rush_yards")].copy()
    y = y.merge(cb[keys + ["consensus_line"]].drop_duplicates(keys), on=keys, how="inner", validate="one_to_one")
    y["vegas_consensus"] = pd.to_numeric(y["consensus_line"], errors="coerce")
    arms = {
        "MC_CANONICAL": "mc_proj", "ML_V2_M91_COMPONENT": "ml_proj", "STATE_V2": "state_proj",
        "ENSEMBLE_2024_FROZEN": "ensemble_2024_frozen", "ENSEMBLE_EXPANDING": "ensemble_expanding",
        "M94C": "m94c_proj", "VEGAS_CONSENSUS": "vegas_consensus",
    }
    return pd.DataFrame([metric_row(y, col, arm, "rush_yards", "listed_market_subset") for arm, col in arms.items()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components-2024", type=Path, required=True)
    ap.add_argument("--components-2025", type=Path, required=True)
    ap.add_argument("--m94c", type=Path, required=True)
    ap.add_argument("--market-casebook", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)

    c24, c25 = prep_components(read(args.components_2024)), prep_components(read(args.components_2025))
    if not c24["season"].eq(2024).all() or not c25["season"].eq(2025).all():
        raise RuntimeError("STACK1 temporal source season mismatch")

    scored, w24, wexp = build_ensembles(c24, c25)
    rb = scored.loc[scored["position"].isin(RB_POS) & scored["market"].isin(MARKETS)].copy()
    if rb.empty:
        raise RuntimeError("STACK1 produced zero 2025 RB/FB rows")
    rb, join_audit = add_m94c(rb, read(args.m94c))

    # Attach actual carries to rushing-yard rows for diagnostic workload slicing.
    carries = rb.loc[rb["market"].eq("rush_att"), ["week", "team", "player_clean_key", "actual"]].rename(columns={"actual": "actual_rush_att"})
    rb = rb.drop(columns=["actual_rush_att"], errors="ignore").merge(carries, on=["week", "team", "player_clean_key"], how="left", validate="many_to_one")

    arms = {
        "MC_CANONICAL": "mc_proj", "ML_V2_M91_COMPONENT": "ml_proj", "STATE_V2": "state_proj",
        "ENSEMBLE_2024_FROZEN": "ensemble_2024_frozen", "ENSEMBLE_EXPANDING": "ensemble_expanding", "M94C": "m94c_proj",
    }
    rows = []
    for market in sorted(MARKETS):
        part = rb.loc[rb["market"].eq(market)]
        for arm, col in arms.items():
            rows.append(metric_row(part, col, arm, market, "all_rb"))
    metrics = pd.DataFrame(rows)

    slice_rows = []
    for market in sorted(MARKETS):
        part = rb.loc[rb["market"].eq(market)].copy()
        for sname, mask in make_slices(part):
            sp = part.loc[mask]
            for arm, col in arms.items():
                slice_rows.append(metric_row(sp, col, arm, market, sname))
    slices = pd.DataFrame(slice_rows)

    coverage = context_coverage(rb)
    w24.to_csv(out / "stack1_weights_2024.csv", index=False)
    wexp.to_csv(out / "stack1_weights_expanding_by_week.csv", index=False)
    metrics.to_csv(out / "stack1_metrics_all.csv", index=False)
    slices.to_csv(out / "stack1_slice_metrics.csv", index=False)
    coverage.to_csv(out / "stack1_context_coverage.csv", index=False)
    join_audit.to_csv(out / "stack1_join_audit.csv", index=False)
    rb.to_csv(out / "stack1_2025_rb_trace.csv", index=False)

    listed = pd.DataFrame()
    if args.market_casebook and args.market_casebook.exists():
        listed = listed_market_benchmark(rb, read(args.market_casebook))
        listed.to_csv(out / "stack1_listed_market_metrics.csv", index=False)

    disposition = pd.DataFrame([{
        "disposition": "STACK1_BASELINE_ESTABLISHED_ADVANCE_TO_ENRICHED_M94C_ALLOCATION_INTEGRATION",
        "model_feature_search": 0, "sportsbook_input_to_football_model": 0,
        "m94c_retune": 0, "production_change": 0,
        "next": "FULL_STACK_PLUS_ENRICHED_M94C_BACKFIELD_ALLOCATION",
    }])
    disposition.to_csv(out / "stack1_disposition.csv", index=False)

    print("=== STACK1 all-RB ===")
    print(metrics.to_string(index=False))
    print("=== STACK1 context coverage ===")
    print(coverage.to_string(index=False))
    print("=== STACK1 M94C join ===")
    print(join_audit.to_string(index=False))
    if not listed.empty:
        print("=== STACK1 listed-market downstream benchmark ===")
        print(listed.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
