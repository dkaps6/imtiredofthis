#!/usr/bin/env python3
"""Certified Full Slate V4 production candidate.

Preserves the certified V3 stack exactly through QB C2, then applies the frozen
R22 Week-1 RB receiving-yard distribution adapter. Uses repo-committed, hash-pinned
R19 assets so a clean checkout has no runtime Actions-artifact dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
from scripts.modeling.rb_receiving_tail_production_adapter_v1 import (
    AUDIT_JSON as RB_REC_AUDIT_JSON,
    TRACE_CSV as RB_REC_TRACE_CSV,
    VERSION as RB_REC_VERSION,
    apply_rb_receiving_tail_production,
)
from scripts.runtime_context import resolve_week

OUT = Path("outputs/props_priced_clean.csv")
PRICING_AUDIT = Path("data/rb_receiving_tail_pricing_lineage_audit.json")
MODEL = Path("data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json")
POOLS = Path("data/models/rb_r19_production_v1/rb_r19_residual_pools_v1.npz")

R22_NOT_APPLICABLE = "RB_R22_NOT_APPLICABLE_OUTSIDE_WEEK1"


def _write_r22_not_applicable_adapter_stub(*, season: int, week: int) -> None:
    payload = {
        "candidate": RB_REC_VERSION,
        "disposition": R22_NOT_APPLICABLE,
        "integration_valid": False,
        "season": int(season),
        "week": int(week),
        "adapted_rb_rows": 0,
        "football_rb_rows": 0,
        "max_mean_delta": 0.0,
        "min_spearman": 1.0,
        "gates": {},
        "sportsbook_inputs_added": 0,
        "current_or_future_outcomes_used": 0,
        "production_mean_parameters_changed": 0,
        "note": (
            "RB R22 receiving-tail adapter is Week-1-only; this week's rec_yards/"
            "rush_rec_yards distribution uses the pre-R22 (V3) result, not a silent "
            "Week-1 fallback."
        ),
    }
    RB_REC_AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _simulate_v4(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    v3_result = v3._simulate_promoted_stack(
        metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace
    )
    seasons = pd.to_numeric(metrics["season"], errors="coerce").dropna().astype(int).unique().tolist()
    weeks = pd.to_numeric(metrics["week"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"Full Slate V4 requires one season/week, got seasons={seasons} weeks={weeks}")
    if weeks[0] != 1:
        # RB R22 receiving-tail refinement is Week-1-only, same as RB P3/R26.
        # Fail closed on the R22 route only: other weeks keep the pre-R22 (V3)
        # receiving-yard result instead of aborting the rest of Full Slate.
        print(
            "[pricing] promoted RB R22 receiving-tail refinement is Week-1-only; "
            f"week={weeks[0]} keeps the pre-R22 receiving-yard result"
        )
        _write_r22_not_applicable_adapter_stub(season=int(seasons[0]), week=int(weeks[0]))
        return v3_result
    adapted, _, _ = apply_rb_receiving_tail_production(
        v3_result,
        metrics,
        season=int(seasons[0]),
        week=int(weeks[0]),
        model_path=MODEL,
        pools_path=POOLS,
    )
    return adapted


def _stamp_pricing_lineage(*, week: int) -> dict:
    if int(week) != 1:
        payload = {
            "disposition": R22_NOT_APPLICABLE,
            "integration_valid": False,
            "version": RB_REC_VERSION,
            "week": int(week),
            "note": (
                "RB R22 receiving-tail refinement is Week-1-only; this week's rec_yards/"
                "rush_rec_yards pricing uses the pre-R22 (V3) result, not a silent "
                "Week-1 fallback."
            ),
        }
        PRICING_AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    if not OUT.is_file() or not RB_REC_AUDIT_JSON.is_file() or not RB_REC_TRACE_CSV.is_file():
        raise RuntimeError("V4 pricing lineage stamp missing priced output or RB receiving audit/trace")
    priced = pd.read_csv(OUT, low_memory=False)
    trace = pd.read_csv(RB_REC_TRACE_CSV, low_memory=False)
    required = {"team", "player", "player_clean_key", "market", "model_proj", "mc_proj"}
    missing = required - set(priced.columns)
    if missing:
        raise RuntimeError(f"V4 priced output missing lineage columns: {sorted(missing)}")
    if trace.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("V4 RB receiving trace contains duplicate team/player keys")

    adapted_keys = {
        (str(r.team).upper(), str(r.player_clean_key))
        for r in trace.itertuples(index=False)
        if bool(r.rb_receiving_tail_applied)
    }

    canonical_pricing_keys = priced["player"].map(v2._suffix_safe_key)
    blank = canonical_pricing_keys.astype("string").fillna("").str.strip().eq("")
    if blank.any():
        sample = priced.loc[blank, ["player", "player_clean_key", "team"]].drop_duplicates().head(20).to_dict("records")
        raise RuntimeError(f"R22 pricing lineage suffix-safe identity unresolved: {sample}")
    priced["rb_receiving_tail_canonical_player_key"] = canonical_pricing_keys

    ambiguity = (
        priced.assign(_provider_name=priced["player"].astype("string").fillna("").str.strip())
        .groupby([priced["team"].astype(str).str.upper(), canonical_pricing_keys], dropna=False)["_provider_name"]
        .nunique(dropna=False)
    )
    bad = ambiguity.loc[ambiguity.gt(1)]
    if not bad.empty:
        raise RuntimeError(f"R22 pricing lineage suffix-safe identity ambiguous: {bad.head(20).to_dict()}")

    key_series = list(zip(priced["team"].astype(str).str.upper(), canonical_pricing_keys.astype(str)))
    is_adapted_player = pd.Series([k in adapted_keys for k in key_series], index=priced.index)
    eligible_market = priced["market"].astype(str).isin(["rec_yards", "rush_rec_yards"])
    applied = eligible_market & is_adapted_player

    priced["rb_receiving_tail_applied"] = applied.astype(bool)
    priced["rb_receiving_tail_version"] = np.where(applied, RB_REC_VERSION, "")
    priced["rb_receiving_tail_model_run"] = np.where(applied, 34288244770, np.nan)
    priced["rb_receiving_tail_mean_preserved"] = applied.astype(bool)
    priced.to_csv(OUT, index=False)

    audit = json.loads(RB_REC_AUDIT_JSON.read_text(encoding="utf-8"))
    rec_rows = priced.loc[applied & priced.market.astype(str).eq("rec_yards")]
    combo_rows = priced.loc[applied & priced.market.astype(str).eq("rush_rec_yards")]
    reception_rows = priced.loc[is_adapted_player & priced.market.astype(str).eq("receptions")]
    provider_alias_rows = int((priced["player_clean_key"].astype(str) != canonical_pricing_keys.astype(str)).sum())
    adapted_provider_alias_rows = int((is_adapted_player & (priced["player_clean_key"].astype(str) != canonical_pricing_keys.astype(str))).sum())
    payload = {
        "disposition": "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS",
        "integration_valid": True,
        "version": RB_REC_VERSION,
        "adapted_player_keys": int(len(adapted_keys)),
        "priced_rows_stamped": int(applied.sum()),
        "rec_yards_rows_stamped": int(len(rec_rows)),
        "rush_rec_yards_rows_stamped": int(len(combo_rows)),
        "receptions_rows_present_unadapted": int(len(reception_rows)),
        "adapter_max_mean_delta": float(audit["max_mean_delta"]),
        "sportsbook_inputs_to_adapter": 0,
        "production_mean_parameters_changed": 0,
        "provider_alias_rows": provider_alias_rows,
        "adapted_provider_alias_rows": adapted_provider_alias_rows,
        "provider_identity_note": "Pricing lineage uses the same governed suffix-safe player identity as the post-simulation provider alias installer; provider event/player aliases remain lookup-only and do not enter football generation.",
        "note": "Only RB rec_yards/rush_rec_yards distribution rows are stamped. Receptions remain unadapted and receiving means remain canonical.",
    }
    if payload["rec_yards_rows_stamped"] <= 0:
        raise RuntimeError("R22 reached zero priced RB receiving-yard rows")
    if payload["receptions_rows_present_unadapted"] <= 0:
        raise RuntimeError("R22 audit unexpectedly found zero RB receptions rows for unchanged-market protection")
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
    payload = _stamp_pricing_lineage(week=int(resolve_week()))
    print("[rb_receiving_tail_pricing_lineage] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
