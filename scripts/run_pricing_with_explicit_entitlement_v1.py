#!/usr/bin/env python3
"""Price Full Slate with explicit, projection-neutral target entitlement.

This hardening layer sits on top of the certified sportsbook-independent full
roster universe. It makes the target probabilities that the legacy simulator was
already using explicit before Monte Carlo, and proves the refactor is neutral
under a matched seed before allowing pricing to continue.

No TE-R5P/WR/RB receiving research is promoted here. That comes only after this
architectural seam is certified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as identity
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.simulation_v2 import simulate as legacy_simulate
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

DATA = Path("data")
TRACE = DATA / "target_entitlement_v1_trace.csv"
AUDIT = DATA / "target_entitlement_v1_audit.json"
INVARIANCE = DATA / "target_entitlement_v1_projection_invariance.csv"

_ORIGINAL_BUILD = base._build_full_universe


def _build_with_explicit_entitlement(pricing_metrics: pd.DataFrame):
    universe, aliases, audit = _ORIGINAL_BUILD(pricing_metrics)
    explicit, trace = materialize_target_entitlement(universe)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(TRACE, index=False)
    team = trace.drop_duplicates(["event_id", "team"])
    payload = {
        "disposition": "EXPLICIT_TARGET_ENTITLEMENT_MATERIALIZED",
        "version": "TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL",
        "football_players": int(len(explicit)),
        "teams": int(team["team"].nunique()),
        "games": int(team["event_id"].nunique()),
        "raw_team_sum_min": float(team["raw_team_sum"].min()),
        "raw_team_sum_median": float(team["raw_team_sum"].median()),
        "raw_team_sum_max": float(team["raw_team_sum"].max()),
        "explicit_modeled_sum_min": float(team["modeled_player_sum"].min()),
        "explicit_modeled_sum_max": float(team["modeled_player_sum"].max()),
        "residual_min": float(team["residual_share"].min()),
        "residual_max": float(team["residual_share"].max()),
        "m38_applied_before_entitlement": True,
        "sportsbook_inputs_used": False,
        "new_scientific_parameters_introduced": False,
        "trace": str(TRACE),
    }
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.update({
        "explicit_target_entitlement_version": payload["version"],
        "explicit_target_entitlement_materialized": True,
    })
    return explicit, aliases, audit


def _projection_neutral_simulate(metrics: pd.DataFrame, *, iterations=None, seed=None, allocation_trace=None):
    # Recreate the pre-refactor input by dropping explicit fields. All other
    # football inputs/order are identical. Both simulations receive the same seed.
    legacy_input = metrics.drop(
        columns=[c for c in metrics.columns if c.startswith("entitlement_")],
        errors="ignore",
    )
    legacy = legacy_simulate(legacy_input, iterations=iterations, seed=seed)
    explicit = explicit_simulate(metrics, iterations=iterations, seed=seed, allocation_trace=allocation_trace)

    legacy_keys = set(legacy.values)
    explicit_keys = set(explicit.values)
    if legacy_keys != explicit_keys:
        raise RuntimeError(
            "explicit entitlement changed simulation key universe; "
            f"missing={list(legacy_keys-explicit_keys)[:20]} extra={list(explicit_keys-legacy_keys)[:20]}"
        )
    rows = []
    max_mean_gap = 0.0
    max_element_gap = 0.0
    changed_arrays = 0
    for key in sorted(legacy_keys):
        a = np.asarray(legacy.values[key], dtype=float)
        b = np.asarray(explicit.values[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"explicit entitlement invalid simulation arrays key={key}")
        mean_gap = abs(float(a.mean()) - float(b.mean()))
        element_gap = float(np.max(np.abs(a-b))) if len(a) else 0.0
        if element_gap > 0:
            changed_arrays += 1
        max_mean_gap = max(max_mean_gap, mean_gap)
        max_element_gap = max(max_element_gap, element_gap)
        rows.append({
            "event_id": key[0], "player_clean_key": key[1], "market": key[2],
            "legacy_mean": float(a.mean()) if len(a) else np.nan,
            "explicit_mean": float(b.mean()) if len(b) else np.nan,
            "mean_gap": mean_gap, "max_element_gap": element_gap,
        })
    pd.DataFrame(rows).to_csv(INVARIANCE, index=False)

    # The nextafter guard should make the refactor effectively identical. Allow
    # only sub-statistical Monte Carlo drift; anything player-relevant is fatal.
    if max_mean_gap > 0.005:
        raise RuntimeError(
            f"explicit target entitlement is not projection-neutral: max_mean_gap={max_mean_gap}"
        )
    status = json.loads(AUDIT.read_text(encoding="utf-8"))
    status.update({
        "projection_invariance_keys": int(len(rows)),
        "projection_invariance_changed_arrays": int(changed_arrays),
        "projection_invariance_max_mean_gap": max_mean_gap,
        "projection_invariance_max_element_gap": max_element_gap,
        "projection_neutral_gate": "PASS",
        "invariance_audit": str(INVARIANCE),
    })
    AUDIT.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[target_entitlement_v1] " + json.dumps(status, sort_keys=True))
    return explicit


def main() -> int:
    base._identity_frame = identity._canonical_identity_frame
    base._validate_priced_distribution_coverage = identity._install_provider_player_aliases_and_validate
    base._build_full_universe = _build_with_explicit_entitlement
    base.canonical_simulate = _projection_neutral_simulate
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
