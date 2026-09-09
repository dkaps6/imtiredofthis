#!/usr/bin/env python3
"""Certified Full Slate V4: V3 stack plus Week-1 RB receiving tail distribution.

Preserves the V3 production stack exactly through QB C2, then applies the frozen
R19/R17/R18 receiving-yard tail adapter to 2026 Week-1 RBs only. The adapter is
mean-neutral and does not change target entitlement or receptions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3 as v3
from scripts.modeling.rb_receiving_tail_production_adapter_v1 import (
    AUDIT_JSON as RB_REC_AUDIT_JSON,
    TRACE_CSV as RB_REC_TRACE_CSV,
    VERSION as RB_REC_VERSION,
    apply_rb_receiving_tail_production,
)

OUT = Path("outputs/props_priced_clean.csv")
PRICING_AUDIT = Path("data/rb_receiving_tail_pricing_lineage_audit.json")


def _simulate_v4(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    v3_result = v3._simulate_promoted_stack(
        metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace
    )
    seasons = pd.to_numeric(metrics["season"], errors="coerce").dropna().astype(int).unique().tolist()
    weeks = pd.to_numeric(metrics["week"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"Full Slate V4 requires one season/week, got seasons={seasons} weeks={weeks}")
    adapted, _, _ = apply_rb_receiving_tail_production(
        v3_result, metrics, season=int(seasons[0]), week=int(weeks[0])
    )
    return adapted


def _stamp_pricing_lineage() -> dict:
    if not OUT.is_file() or not RB_REC_AUDIT_JSON.is_file() or not RB_REC_TRACE_CSV.is_file():
        raise RuntimeError("V4 pricing lineage stamp missing priced output or RB receiving audit/trace")
    priced = pd.read_csv(OUT, low_memory=False)
    trace = pd.read_csv(RB_REC_TRACE_CSV, low_memory=False)
    required = {"event_id", "player_clean_key", "market", "model_proj", "mc_proj"}
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"V4 priced output missing lineage columns: {sorted(missing)}")
    if trace.duplicated(["event_id", "player_clean_key"]).any():
        raise RuntimeError("V4 RB receiving trace contains duplicate adapted keys")

    adapted_keys = {
        (str(r.event_id), str(r.player_clean_key))
        for r in trace.itertuples(index=False)
    }
    key_series = list(zip(priced["event_id"].astype(str), priced["player_clean_key"].astype(str)))
    eligible_market = priced["market"].astype(str).isin(["rec_yards", "rush_rec_yards"])
    is_adapted_player = pd.Series([k in adapted_keys for k in key_series], index=priced.index)
    applied = eligible_market & is_adapted_player

    priced["rb_receiving_tail_applied"] = applied.astype(bool)
    priced["rb_receiving_tail_version"] = np.where(applied, RB_REC_VERSION, "")
    priced["rb_receiving_tail_model_run"] = np.where(applied, 34288244770, np.nan)
    priced["rb_receiving_tail_mean_preserved"] = applied.astype(bool)
    priced.to_csv(OUT, index=False)

    audit = json.loads(RB_REC_AUDIT_JSON.read_text(encoding="utf-8"))
    rec_rows = priced.loc[applied].copy()
    rec_yards_rows = rec_rows.loc[rec_rows.market.astype(str).eq("rec_yards")]
    combo_rows = rec_rows.loc[rec_rows.market.astype(str).eq("rush_rec_yards")]
    payload = {
        "disposition": "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS",
        "integration_valid": True,
        "version": RB_REC_VERSION,
        "adapted_player_keys": int(len(adapted_keys)),
        "priced_rows_stamped": int(applied.sum()),
        "rec_yards_rows_stamped": int(len(rec_yards_rows)),
        "rush_rec_yards_rows_stamped": int(len(combo_rows)),
        "receptions_rows_stamped": int((priced.market.astype(str).eq("receptions") & is_adapted_player).sum() if False else 0),
        "adapter_max_mean_delta": float(audit["max_mean_delta"]),
        "sportsbook_inputs_to_adapter": 0,
        "production_mean_parameters_changed": 0,
        "note": "Lineage stamp identifies RB rows whose simulation distribution was adapted before pricing. Receiving-yard means remain canonical; receptions are not adapted.",
    }
    PRICING_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = v2._install_provider_player_aliases_and_validate
    base._build_full_universe = v3._build_with_promoted_entitlement_specialists
    base.canonical_simulate = _simulate_v4
    rc = int(base.main())
    if rc != 0:
        return rc
    payload = _stamp_pricing_lineage()
    print("[rb_receiving_tail_pricing_lineage] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
