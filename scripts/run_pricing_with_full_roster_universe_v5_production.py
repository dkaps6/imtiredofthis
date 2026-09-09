#!/usr/bin/env python3
"""Week-1 Full Slate V5 production candidate with R26 RB receptions refinement.

Preserves protected V4 through QB C2 and R22, then replaces only qualified
vacancy-room RB/FB receptions Monte Carlo arrays with the R26 entitlement result.
The existing MC/ML/state ensemble remains unchanged and produces the single final
`model_proj` consumed by sportsbook comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
import scripts.run_pricing_with_full_roster_universe_v4_production as v4
from scripts.modeling.rb_r26_receptions_production_adapter_v1 import (
    AUDIT_JSON as R26_AUDIT_JSON,
    TRACE_CSV as R26_TRACE_CSV,
    VERSION as R26_VERSION,
    apply_rb_r26_receptions_production,
)

OUT = Path("outputs/props_priced_clean.csv")
R26_PRICING_AUDIT = Path("data/rb_r26_receptions_pricing_lineage_audit.json")


def _simulate_v5(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    v4_result = v4._simulate_v4(
        metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace
    )
    seasons = pd.to_numeric(metrics["season"], errors="coerce").dropna().astype(int).unique().tolist()
    weeks = pd.to_numeric(metrics["week"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"Full Slate V5 requires one season/week, got seasons={seasons} weeks={weeks}")
    adapted, _, _ = apply_rb_r26_receptions_production(
        v4_result,
        metrics,
        season=int(seasons[0]),
        week=int(weeks[0]),
        iterations=iterations,
        seed=seed,
    )
    return adapted


def _stamp_r26_pricing_lineage() -> dict:
    if not OUT.is_file() or not R26_AUDIT_JSON.is_file() or not R26_TRACE_CSV.is_file():
        raise RuntimeError("V5 R26 pricing lineage missing priced output or R26 audit/trace")
    priced = pd.read_csv(OUT, low_memory=False)
    trace = pd.read_csv(R26_TRACE_CSV, low_memory=False)
    required = {
        "team", "player", "player_clean_key", "market", "source_market",
        "model_proj", "mc_proj", "ensemble_proj", "ensemble_status", "ensemble_method",
        "ml_proj", "state_proj",
    }
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"V5 priced output missing R26 lineage columns: {sorted(missing)}")
    if trace.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("V5 R26 trace duplicate team/player identities")

    canonical = priced["player"].map(v2._suffix_safe_key)
    if canonical.astype("string").fillna("").str.strip().eq("").any():
        sample = priced.loc[canonical.astype("string").fillna("").str.strip().eq(""), ["player", "team"]].head(20).to_dict("records")
        raise RuntimeError(f"V5 R26 pricing canonical identity unresolved: {sample}")
    priced["rb_r26_receptions_canonical_player_key"] = canonical

    t = trace.copy()
    t["team"] = t["team"].astype(str).str.upper().str.strip()
    t["player_clean_key"] = t["player_clean_key"].astype(str)
    t_index = t.set_index(["team", "player_clean_key"])
    applied_keys = {
        (str(r.team).upper(), str(r.player_clean_key))
        for r in t.itertuples(index=False)
        if bool(r.rb_r26_receptions_applied)
    }
    all_trace_keys = set(zip(t.team.astype(str), t.player_clean_key.astype(str)))

    row_keys = list(zip(priced["team"].astype(str).str.upper(), canonical.astype(str)))
    is_trace_player = pd.Series([k in all_trace_keys for k in row_keys], index=priced.index)
    is_applied_player = pd.Series([k in applied_keys for k in row_keys], index=priced.index)
    is_receptions = priced["market"].astype(str).eq("receptions")
    applied = is_applied_player & is_receptions

    baseline_map = t_index["baseline_receptions_mean"].to_dict()
    final_map = t_index["final_receptions_mean"].to_dict()
    delta_map = t_index["final_minus_baseline_receptions_mean"].to_dict()

    priced["rb_r26_receptions_applied"] = applied.astype(bool)
    priced["rb_r26_receptions_version"] = np.where(applied, R26_VERSION, "")
    priced["rb_r26_baseline_mc_proj_audit"] = [
        baseline_map.get(k, np.nan) if rec else np.nan
        for k, rec in zip(row_keys, is_receptions)
    ]
    priced["rb_r26_final_mc_proj_audit"] = [
        final_map.get(k, np.nan) if rec else np.nan
        for k, rec in zip(row_keys, is_receptions)
    ]
    priced["rb_r26_mc_delta_audit"] = [
        delta_map.get(k, np.nan) if rec else np.nan
        for k, rec in zip(row_keys, is_receptions)
    ]

    applied_rows = priced.loc[applied].copy()
    if applied_rows.empty:
        raise RuntimeError("V5 R26 reached zero priced applied RB/FB receptions rows")
    mc = pd.to_numeric(applied_rows["mc_proj"], errors="coerce")
    final_mc = pd.to_numeric(applied_rows["rb_r26_final_mc_proj_audit"], errors="coerce")
    if mc.isna().any() or final_mc.isna().any() or not np.allclose(mc, final_mc, rtol=0, atol=1e-8):
        sample = applied_rows.loc[~np.isclose(mc, final_mc, rtol=0, atol=1e-8, equal_nan=False), [
            "team", "player", "mc_proj", "rb_r26_final_mc_proj_audit"
        ]].head(20).to_dict("records")
        raise RuntimeError(f"V5 R26 final MC mean did not reach priced mc_proj: {sample}")

    model_proj = pd.to_numeric(applied_rows["model_proj"], errors="coerce")
    ensemble_proj = pd.to_numeric(applied_rows["ensemble_proj"], errors="coerce")
    if model_proj.isna().any() or ensemble_proj.isna().any() or not np.allclose(model_proj, ensemble_proj, rtol=0, atol=1e-8):
        sample = applied_rows.loc[~np.isclose(model_proj, ensemble_proj, rtol=0, atol=1e-8, equal_nan=False), [
            "team", "player", "model_proj", "ensemble_proj"
        ]].head(20).to_dict("records")
        raise RuntimeError(f"V5 R26 receptions bypassed existing ensemble: {sample}")

    duplicate_offer = priced.loc[is_receptions].duplicated([
        "event_id", "player", "team", "source_market", "vegas_line", "side", "book"
    ]) if all(c in priced.columns for c in ["event_id", "player", "team", "source_market", "vegas_line", "side", "book"]) else pd.Series(False, index=priced.loc[is_receptions].index)
    if bool(duplicate_offer.any()):
        raise RuntimeError("V5 R26 produced duplicate priced reception offers")

    priced.to_csv(OUT, index=False)

    r26 = json.loads(R26_AUDIT_JSON.read_text(encoding="utf-8"))
    payload = {
        "disposition": "RB_R26_WEEK1_RECEPTIONS_PRICING_LINEAGE_PASS",
        "integration_valid": True,
        "version": R26_VERSION,
        "trace_rb_fb_players": int(len(trace)),
        "trace_applied_players": int(trace.rb_r26_receptions_applied.sum()),
        "priced_reception_rows_for_trace_players": int((is_trace_player & is_receptions).sum()),
        "priced_r26_applied_rows": int(applied.sum()),
        "max_abs_mc_trace_gap": float(np.max(np.abs(mc.to_numpy(float) - final_mc.to_numpy(float)))) if len(applied_rows) else 0.0,
        "max_abs_model_vs_ensemble_gap": float(np.max(np.abs(model_proj.to_numpy(float) - ensemble_proj.to_numpy(float)))) if len(applied_rows) else 0.0,
        "baseline_retained_for_audit_only": True,
        "existing_ml_state_inputs_rewritten_by_r26": False,
        "existing_ensemble_method_rewritten_by_r26": False,
        "sportsbook_inputs_to_r26_football": 0,
        "r26_adapter_disposition": r26.get("disposition"),
        "single_authoritative_model_proj": True,
        "note": "R26 changes the eligible RB/FB receptions MC distribution before the existing calibrated ensemble. Pricing emits one final model_proj; baseline reception means are audit-only on applied rows.",
    }
    R26_PRICING_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _clarify_r22_lineage_after_r26() -> None:
    path = v4.PRICING_AUDIT
    if not path.is_file():
        raise RuntimeError("V5 expected R22 pricing lineage audit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("disposition") != "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS":
        raise RuntimeError("V5 R22 pricing lineage lost certified disposition")
    payload["downstream_r26_receptions_adapter_present"] = True
    payload["note"] = (
        "R22 itself adapts only RB rec_yards/rush_rec_yards. Its receptions arrays are unchanged at the R22 seam; "
        "the separately qualified downstream R26 adapter may subsequently refine RB/FB receptions before pricing."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = v2._install_provider_player_aliases_and_validate
    base._build_full_universe = v3._build_with_promoted_entitlement_specialists
    base.canonical_simulate = _simulate_v5
    rc = int(base.main())
    if rc != 0:
        return rc
    r22_payload = v4._stamp_pricing_lineage()
    _clarify_r22_lineage_after_r26()
    r26_payload = _stamp_r26_pricing_lineage()
    print("[rb_receiving_tail_pricing_lineage] " + json.dumps(r22_payload, sort_keys=True))
    print("[rb_r26_receptions_pricing_lineage] " + json.dumps(r26_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
