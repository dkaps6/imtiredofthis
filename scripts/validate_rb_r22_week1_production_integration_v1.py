#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/v3control/outputs/props_priced_clean.csv")
CANDIDATE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/props_priced_clean.csv")
TRACE = Path("data/rb_receiving_tail_production_trace.csv")
ADAPTER_AUDIT = Path("data/rb_receiving_tail_production_audit.json")
PRICING_AUDIT = Path("data/rb_receiving_tail_pricing_lineage_audit.json")
TE_AUDIT = Path("data/te_r5p_full_slate_entitlement_audit.json")
WR_AUDIT = Path("data/wr_r15_full_slate_entitlement_audit.json")
QB_AUDIT = Path("data/qb_c2_production_integration_audit.json")
OUT_JSON = Path("data/rb_r22_week1_production_integration_audit.json")
OUT_CSV = Path("data/rb_r22_week1_pricing_delta_audit.csv")

KEYS = [
    "event_id", "player_clean_key", "market", "book", "side",
    "vegas_line", "vegas_odds",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"missing R22 validation input: {path}")
    return pd.read_csv(path, low_memory=False)


def _num_equal(a: pd.Series, b: pd.Series, tol: float = 1e-10) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    both_nan = x.isna() & y.isna()
    finite = x.notna() & y.notna() & ((x - y).abs() <= tol)
    return both_nan | finite


def main() -> int:
    control = _read(CONTROL)
    candidate = _read(CANDIDATE)
    trace = _read(TRACE)
    adapter = json.loads(ADAPTER_AUDIT.read_text(encoding="utf-8"))
    pricing = json.loads(PRICING_AUDIT.read_text(encoding="utf-8"))
    te = json.loads(TE_AUDIT.read_text(encoding="utf-8"))
    wr = json.loads(WR_AUDIT.read_text(encoding="utf-8"))
    qb = json.loads(QB_AUDIT.read_text(encoding="utf-8"))

    for df, label in [(control, "control"), (candidate, "candidate")]:
        missing = [c for c in KEYS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{label} priced output missing comparison keys: {missing}")
        if df.duplicated(KEYS).any():
            dup = df.loc[df.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
            raise RuntimeError(f"{label} priced output has duplicate comparison keys: {dup}")

    c = control.copy()
    n = candidate.copy()
    c["_control_row"] = np.arange(len(c))
    n["_candidate_row"] = np.arange(len(n))
    merged = c.merge(n, on=KEYS, how="outer", suffixes=("_control", "_candidate"), indicator=True, validate="one_to_one")
    row_universe_exact = bool(len(c) == len(n) and merged["_merge"].eq("both").all())

    adapted_keys = {
        (str(r.event_id), str(r.player_clean_key))
        for r in trace.itertuples(index=False)
        if bool(r.rb_receiving_tail_applied)
    }
    merged["adapted_rb"] = [
        (str(e), str(p)) in adapted_keys
        for e, p in zip(merged["event_id"], merged["player_clean_key"])
    ]
    merged["allowed_distribution_change"] = merged["adapted_rb"] & merged["market"].astype(str).isin(["rec_yards", "rush_rec_yards"])

    numeric_checks = {}
    for col in ["model_proj", "mc_proj", "ensemble_proj", "ml_proj", "state_proj"]:
        lc, rc = f"{col}_control", f"{col}_candidate"
        if lc in merged.columns and rc in merged.columns:
            eq = _num_equal(merged[lc], merged[rc], tol=1e-8)
            numeric_checks[col] = {
                "all_equal": bool(eq.all()),
                "max_abs_delta": float(np.nanmax(np.abs(pd.to_numeric(merged[rc], errors="coerce") - pd.to_numeric(merged[lc], errors="coerce")))) if len(merged) else 0.0,
            }
            merged[f"{col}_equal"] = eq

    model_means_exact = bool(numeric_checks.get("model_proj", {}).get("all_equal", False))
    mc_means_exact = bool(numeric_checks.get("mc_proj", {}).get("all_equal", False))

    probability_cols = ["fair_prob", "fair_odds", "edge_pct", "edge_abs"]
    unexpected_prob_changes = pd.Series(False, index=merged.index)
    allowed_changed_rows = pd.Series(False, index=merged.index)
    for col in probability_cols:
        lc, rc = f"{col}_control", f"{col}_candidate"
        if lc not in merged.columns or rc not in merged.columns:
            continue
        eq = _num_equal(merged[lc], merged[rc], tol=1e-10)
        merged[f"{col}_equal"] = eq
        unexpected_prob_changes |= (~eq) & (~merged.allowed_distribution_change)
        allowed_changed_rows |= (~eq) & merged.allowed_distribution_change

    receptions_exact = True
    recmask = merged["adapted_rb"] & merged["market"].astype(str).eq("receptions")
    for col in ["model_proj", "mc_proj", *probability_cols]:
        lc, rc = f"{col}_control", f"{col}_candidate"
        if lc in merged.columns and rc in merged.columns and recmask.any():
            receptions_exact &= bool(_num_equal(merged.loc[recmask, lc], merged.loc[recmask, rc], tol=1e-10).all())

    non_rb_exact = True
    non_rb = ~merged["adapted_rb"]
    for col in ["model_proj", "mc_proj", *probability_cols]:
        lc, rc = f"{col}_control", f"{col}_candidate"
        if lc in merged.columns and rc in merged.columns and non_rb.any():
            non_rb_exact &= bool(_num_equal(merged.loc[non_rb, lc], merged.loc[non_rb, rc], tol=1e-10).all())

    existing_stack_valid = bool(
        te.get("integration_valid") is True
        and wr.get("integration_valid") is True
        and qb.get("qb_c2_integration_valid") is True
        and te.get("sportsbook_inputs_used") is False
        and wr.get("sportsbook_inputs_used") is False
        and int(qb.get("sportsbook_inputs_to_selector", 1)) == 0
    )

    gates = {
        "row_universe_exact": row_universe_exact,
        "adapter_integration_valid": adapter.get("integration_valid") is True,
        "exact_94_rb_adapted": int(adapter.get("adapted_rb_rows", -1)) == 94,
        "adapter_mean_parity": float(adapter.get("max_mean_delta", 1.0)) <= 1e-8,
        "adapter_rank_preservation": float(adapter.get("min_spearman", -1.0)) >= 0.9999,
        "adapter_non_rb_exact": adapter.get("gates", {}).get("non_rb_exact") is True,
        "adapter_receptions_exact": adapter.get("gates", {}).get("receptions_exact") is True,
        "adapter_rb_other_markets_exact": adapter.get("gates", {}).get("rb_nonreceiving_markets_exact") is True,
        "adapter_rush_rec_identity": adapter.get("gates", {}).get("rush_rec_identity") is True,
        "pricing_lineage_valid": pricing.get("integration_valid") is True,
        "priced_model_means_exact_vs_v3": model_means_exact,
        "priced_mc_means_exact_vs_v3": mc_means_exact,
        "priced_receptions_exact_vs_v3": receptions_exact,
        "priced_nonadapted_rows_exact_vs_v3": non_rb_exact,
        "no_unexpected_probability_changes": not bool(unexpected_prob_changes.any()),
        "existing_te_wr_qb_stack_valid": existing_stack_valid,
        "sportsbook_zero_to_adapter": int(adapter.get("sportsbook_inputs_added", 1)) == 0,
        "outcomes_zero_to_adapter": int(adapter.get("current_or_future_outcomes_used", 1)) == 0,
    }
    passed = bool(all(gates.values()))

    merged["unexpected_probability_change"] = unexpected_prob_changes
    merged["allowed_probability_change_observed"] = allowed_changed_rows
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep = KEYS + [
        "adapted_rb", "allowed_distribution_change", "unexpected_probability_change",
        "allowed_probability_change_observed",
    ]
    for col in ["model_proj", "mc_proj", *probability_cols]:
        for suffix in ["_control", "_candidate"]:
            name = f"{col}{suffix}"
            if name in merged.columns:
                keep.append(name)
    merged[keep].to_csv(OUT_CSV, index=False)

    payload = {
        "candidate": "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_V1",
        "disposition": "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS" if passed else "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_FAIL",
        "pass": passed,
        "control_rows": int(len(control)),
        "candidate_rows": int(len(candidate)),
        "adapted_rb_player_keys": int(len(adapted_keys)),
        "allowed_probability_change_rows": int(allowed_changed_rows.sum()),
        "unexpected_probability_change_rows": int(unexpected_prob_changes.sum()),
        "numeric_mean_checks": numeric_checks,
        "gates": gates,
        "adapter_disposition": adapter.get("disposition"),
        "pricing_lineage_disposition": pricing.get("disposition"),
        "governance_note": "PASS authorizes Week-1 RB receiving-yard distribution integration only. Receptions/target entitlement/means remain canonical; R21 prospective evidence remains active.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
