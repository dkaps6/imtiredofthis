"""Production-candidate wrapper for the frozen QB C2 selector.

The selector, starter authority, features, thresholds, and audit logic remain the
existing production implementation. Only the C2 simulation generator is swapped
for the semantic-RNG isolation candidate during this call.
"""
from __future__ import annotations

import pandas as pd

import scripts.modeling.qb_c2_production_adapter_v1 as production
from scripts.simulation_c2_qb_candidate import StateSimulationResult
from scripts.simulation_c2_rng_isolation_v1 import apply_c2 as isolated_apply_c2


def apply_qb_c2_selector(
    base_state: StateSimulationResult,
    metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
):
    original = production.apply_c2
    production.apply_c2 = isolated_apply_c2
    try:
        selected, audit, payload = production.apply_qb_c2_selector(
            base_state,
            metrics,
            season=int(season),
            week=int(week),
        )
    finally:
        production.apply_c2 = original

    payload = dict(payload)
    payload["rng_isolation_candidate"] = True
    payload["rng_isolation_version"] = "SPECIALIST_RNG_ISOLATION_PRODUCTION_REPAIR_V1"
    payload["sportsbook_inputs_to_rng_routing"] = 0
    selected.qb_distribution_audit = payload
    return selected, audit, payload
