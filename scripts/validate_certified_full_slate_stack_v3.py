#!/usr/bin/env python3
"""Fail-closed Full Slate certification including Week-1 RB-R22.

Runs the established V2 certification first, then requires the exact R22
mean-preserving RB receiving-yard production adapter/pricing lineage. This does
not claim RB receptions or receiving entitlement are solved.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import validate_certified_full_slate_stack_v2_core as v2

DATA = Path("data")
OUT = DATA / "certified_full_slate_stack_audit.json"
ADAPTER = DATA / "rb_receiving_tail_production_audit.json"
PRICING = DATA / "rb_receiving_tail_pricing_lineage_audit.json"
LINEAGE = DATA / "market_model_lineage_current.json"

VERSION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1"


def _load(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"certified stack V3 artifact missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise RuntimeError(message)


def main() -> int:
    rc = int(v2.main())
    if rc != 0:
        return rc

    adapter = _load(ADAPTER)
    pricing = _load(PRICING)
    lineage = _load(LINEAGE)
    payload = _load(OUT)

    _require(adapter.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS", "R22 adapter not certified")
    _require(adapter.get("integration_valid") is True, "R22 adapter integration invalid")
    _require(int(adapter.get("season", 0)) == 2026 and int(adapter.get("week", 0)) == 1, "R22 route is not qualified Week 1")
    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 adapted-RB coverage drift")
    _require(float(adapter.get("max_mean_delta", 1.0)) <= 1e-8, "R22 receiving mean drift")
    _require(float(adapter.get("min_spearman", -1.0)) >= 0.9999, "R22 rank-preservation drift")
    for gate in (
        "all_rb_adapted", "r19_assets_exact", "strict_prior_history",
        "sportsbook_inputs_upstream_zero", "current_or_future_outcomes_zero",
        "deterministic_replay", "finite_nonnegative_rec_yards", "mean_parity",
        "rank_preservation", "r9_shadow_pool_conservation", "non_rb_exact",
        "fb_exact", "receptions_exact", "rb_nonreceiving_markets_exact",
        "rush_rec_identity", "qualified_week",
    ):
        _require(adapter.get("gates", {}).get(gate) is True, f"R22 adapter gate failed/drifted: {gate}")

    _require(pricing.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS", "R22 pricing lineage not certified")
    _require(pricing.get("integration_valid") is True, "R22 pricing lineage invalid")
    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing RB coverage drift")
    _require(int(pricing.get("rec_yards_rows_stamped", 0)) > 0, "R22 does not reach receiving-yard pricing")
    _require(int(pricing.get("rush_rec_yards_rows_stamped", 0)) > 0, "R22 does not reach rush+rec pricing")
    _require(int(pricing.get("receptions_rows_present_unadapted", 0)) > 0, "R22 receptions protection absent")
    _require(int(pricing.get("sportsbook_inputs_to_adapter", 1)) == 0, "sportsbook leaked into R22")
    _require(int(pricing.get("production_mean_parameters_changed", 1)) == 0, "R22 changed mean parameters")

    _require(lineage.get("rb_r22_receiving_tail_consumed") is True, "market lineage does not record R22")
    _require(lineage.get("rb_r22_version") == VERSION, "market lineage R22 version drift")
    _require(lineage.get("rb_r22_receiving_mean_preserved") is True, "market lineage does not preserve R22 mean")
    _require(lineage.get("rb_r22_receptions_preserved") is True, "market lineage does not preserve receptions")
    _require(lineage.get("rb_r22_sportsbook_inputs_used") is False, "market lineage reports R22 sportsbook input")

    payload.update({
        "disposition": "FULL_SLATE_CERTIFIED_STACK_READY_M38_R15_TE_R5P_C2_R22_WEEK1_WITH_DECLARED_SCIENCE_LIMITATIONS",
        "rb_r22_receiving_tail_active": True,
        "rb_r22_version": VERSION,
        "rb_r22_week1_only": True,
        "rb_r22_adapted_rb_rows": 94,
        "rb_r22_max_receiving_mean_delta": float(adapter.get("max_mean_delta")),
        "rb_r22_min_rank_spearman": float(adapter.get("min_spearman")),
        "rb_r22_receptions_unchanged": True,
        "rb_r22_non_rb_unchanged": True,
        "rb_r22_sportsbook_inputs_used": False,
        "remaining_science_lanes": [
            "RB receiving entitlement/receptions and receiving-yard mean refinement (R22 covers Week-1 tail shape only)",
            "WR/TE receiving efficiency and distribution calibration",
            "shared QB-receiver C2 conservation",
            "dedicated anytime-TD probability calibration",
            "game-level moneyline/spread/total model built from certified joint football state",
        ],
    })
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[certified_full_slate_stack_v3] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
