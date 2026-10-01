#!/usr/bin/env python3
"""Full Slate production repair candidate with semantic RNG isolation.

This entry point intentionally does not replace the canonical Full Slate stack.
It reuses the exact current football universe, entitlement specialists, pricing
authorities and selector logic while substituting only the candidate randomness
routing proven by the frozen research chain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as core
from scripts.modeling.qb_c2_production_adapter_rng_isolation_v1 import (
    apply_qb_c2_selector,
)
from scripts.modeling.qb_c2_production_adapter_v1 import AUDIT_JSON as QB_C2_AUDIT_JSON
from scripts.simulation_explicit_entitlement_v1 import simulate as reference_explicit_simulate
from scripts.simulation_rng_isolation_v1 import (
    simulate as isolated_simulate,
    simulate_with_states as isolated_simulate_with_states,
)
from scripts.simulation_v2 import simulate as legacy_simulate

DATA = Path("data")
CANDIDATE_AUDIT = DATA / "specialist_rng_isolation_production_candidate_audit.json"
CANDIDATE_TE_DELTA = DATA / "rng_isolation_te_r5p_simulation_delta.csv"
CANDIDATE_WR_DELTA = DATA / "rng_isolation_wr_r15_simulation_delta.csv"
CANDIDATE_BASELINE_AUTHORITY = DATA / "rng_isolation_baseline_authority_parity.csv"
CANDIDATE_STATE_PARITY = DATA / "rng_isolation_state_capture_parity.csv"
VERSION = "SPECIALIST_RNG_ISOLATION_PRODUCTION_REPAIR_V1"


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def _exact_scope_audit(
    left,
    right,
    metrics: pd.DataFrame,
    *,
    changed_scope_col: str,
    label: str,
) -> dict:
    if set(left.values) != set(right.values):
        raise RuntimeError(f"{label} changed simulation key universe")

    frame = metrics.copy()
    frame["player_clean_key"] = frame["player_clean_key"].astype(str)
    scope = _bool_series(frame[changed_scope_col])
    changed_players = {
        (str(e), str(p))
        for e, p in zip(
            frame.loc[scope, "event_id"],
            frame.loc[scope, "player_clean_key"],
        )
    }
    protected_players = {
        (str(e), str(p))
        for e, p in zip(
            frame.loc[~scope, "event_id"],
            frame.loc[~scope, "player_clean_key"],
        )
    }

    protected_keys = 0
    protected_drift = 0
    max_protected_mean_gap = 0.0
    max_protected_element_gap = 0.0
    intentional_receiving_arrays = 0
    intentional_receiving_changed = 0

    for key in sorted(left.values):
        a = np.asarray(left.values[key], dtype=float)
        b = np.asarray(right.values[key], dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"{label} invalid simulation arrays key={key}")
        ep = (str(key[0]), str(key[1]))
        mean_gap = abs(float(b.mean() - a.mean())) if len(a) else 0.0
        element_gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if ep in protected_players and str(key[2]) in {
            "pass_yards", "rush_att", "rush_yards",
            "receptions", "rec_yards", "rush_rec_yards",
        }:
            protected_keys += 1
            if mean_gap > 1e-12 or element_gap > 1e-12:
                protected_drift += 1
            max_protected_mean_gap = max(max_protected_mean_gap, mean_gap)
            max_protected_element_gap = max(max_protected_element_gap, element_gap)

        if ep in changed_players and str(key[2]) in {"receptions", "rec_yards"}:
            intentional_receiving_arrays += 1
            intentional_receiving_changed += int(element_gap > 1e-12)

    if protected_keys <= 0:
        raise RuntimeError(f"{label} checked zero protected keys")
    if protected_drift != 0:
        raise RuntimeError(
            f"{label} protected RNG isolation failed drift={protected_drift} "
            f"max_mean={max_protected_mean_gap} max_element={max_protected_element_gap}"
        )
    if intentional_receiving_arrays <= 0 or intentional_receiving_changed <= 0:
        raise RuntimeError(f"{label} accidentally froze intended specialist movement")

    return {
        "label": label,
        "protected_keys": int(protected_keys),
        "protected_drift_keys": int(protected_drift),
        "max_protected_mean_gap": float(max_protected_mean_gap),
        "max_protected_element_gap": float(max_protected_element_gap),
        "intentional_receiving_arrays": int(intentional_receiving_arrays),
        "intentional_receiving_changed": int(intentional_receiving_changed),
    }


def _simulate_promoted_stack(
    metrics: pd.DataFrame,
    *,
    iterations=None,
    seed=None,
    allocation_trace=None,
):
    required = {
        "baseline_entitlement_tgt_share",
        "te_only_entitlement_tgt_share",
        "wr_r15_baseline_entitlement_tgt_share",
        "te_r5p_applied",
        "wr_r15_applied",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(
            f"RNG isolation production candidate missing columns: {sorted(missing)}"
        )

    # Preserve the existing exact M38 explicit-entitlement authority proof using
    # the unchanged legacy/current-production paths. The new RNG architecture is
    # not used for this seam proof because a different finite random sample is
    # expected by design.
    legacy_input = metrics.drop(
        columns=[
            c
            for c in metrics.columns
            if c.startswith("entitlement_")
            or c.startswith("baseline_entitlement_")
            or c.startswith("te_r5p_")
            or c.startswith("te_only_")
            or c.startswith("wr_r15_")
        ],
        errors="ignore",
    )
    baseline_metrics = metrics.copy()
    baseline_metrics["entitlement_tgt_share"] = pd.to_numeric(
        baseline_metrics["baseline_entitlement_tgt_share"], errors="raise"
    ).astype(float)

    legacy = legacy_simulate(legacy_input, iterations=iterations, seed=seed)
    baseline_reference = reference_explicit_simulate(
        baseline_metrics, iterations=iterations, seed=seed
    )
    neutral = v2._compare_results_exact(
        legacy,
        baseline_reference,
        label="RNG isolation candidate M38 authority parity",
    )
    authority_rows = []
    for key in sorted(legacy.values):
        a = np.asarray(legacy.values[key], dtype=float)
        b = np.asarray(baseline_reference.values[key], dtype=float)
        authority_rows.append(
            {
                "event_id": key[0],
                "player_clean_key": key[1],
                "market": key[2],
                "legacy_mean": float(a.mean()) if len(a) else np.nan,
                "explicit_reference_mean": float(b.mean()) if len(b) else np.nan,
                "mean_gap": abs(float(a.mean()) - float(b.mean())),
                "max_element_gap": float(np.max(np.abs(a - b))) if len(a) else 0.0,
            }
        )
    pd.DataFrame(authority_rows).to_csv(CANDIDATE_BASELINE_AUTHORITY, index=False)
    if (
        neutral["changed_arrays"] != 0
        or neutral["max_mean_gap"] > 1e-12
        or neutral["max_element_gap"] > 1e-12
    ):
        raise RuntimeError(
            f"RNG isolation candidate changed M38 entitlement authority: {neutral}"
        )

    # Candidate finite-MC stages.
    baseline_isolated = isolated_simulate(
        baseline_metrics, iterations=iterations, seed=seed
    )

    te_metrics = metrics.copy()
    te_metrics["entitlement_tgt_share"] = pd.to_numeric(
        te_metrics["te_only_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    te_isolated = isolated_simulate(te_metrics, iterations=iterations, seed=seed)

    final_isolated = isolated_simulate(
        metrics,
        iterations=iterations,
        seed=seed,
        allocation_trace=allocation_trace,
    )

    te_scope = _exact_scope_audit(
        baseline_isolated,
        te_isolated,
        metrics,
        changed_scope_col="te_r5p_applied",
        label="M38_TO_TE_R5P_RNG_ISOLATION",
    )
    wr_scope = _exact_scope_audit(
        te_isolated,
        final_isolated,
        metrics,
        changed_scope_col="wr_r15_applied",
        label="TE_R5P_TO_WR_R15_RNG_ISOLATION",
    )

    core._write_delta(
        baseline_isolated,
        te_isolated,
        metrics,
        CANDIDATE_TE_DELTA,
        "m38_baseline",
        "te_r5p",
    )
    core._write_delta(
        te_isolated,
        final_isolated,
        metrics,
        CANDIDATE_WR_DELTA,
        "te_r5p",
        "wr_r15",
    )

    # State capture must be exactly the same candidate sample, not a second RNG path.
    stateful = isolated_simulate_with_states(
        metrics,
        iterations=iterations,
        seed=seed,
    )
    state_parity = v2._compare_results_exact(
        final_isolated,
        stateful,
        label="RNG isolation candidate state-capture seam",
        out_path=CANDIDATE_STATE_PARITY,
    )
    if (
        state_parity["changed_arrays"] != 0
        or state_parity["max_mean_gap"] > 1e-12
        or state_parity["max_element_gap"] > 1e-12
    ):
        raise RuntimeError(
            f"RNG isolation candidate state capture is not exact: {state_parity}"
        )

    seasons = (
        pd.to_numeric(metrics["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    weeks = (
        pd.to_numeric(metrics["week"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(
            f"RNG isolation candidate requires one season/week, got "
            f"seasons={seasons} weeks={weeks}"
        )

    selected, _, qb_payload = apply_qb_c2_selector(
        stateful,
        metrics,
        season=int(seasons[0]),
        week=int(weeks[0]),
    )
    qb_payload.update({
        "state_capture_parity_keys": state_parity["keys"],
        "state_capture_changed_arrays": state_parity["changed_arrays"],
        "state_capture_max_mean_gap": state_parity["max_mean_gap"],
        "state_capture_max_element_gap": state_parity["max_element_gap"],
        "state_capture_parity_audit": str(CANDIDATE_STATE_PARITY),
        "te_r5p_consumed_before_c2": True,
        "wr_r15_consumed_before_c2": True,
        "wr_r15_model_version": "WR_R15_PRODUCTION_MODEL_V1",
        "explicit_entitlement_consumed_before_c2": True,
    })
    QB_C2_AUDIT_JSON.write_text(
        json.dumps(qb_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected.qb_distribution_audit = qb_payload

    payload = {
        "disposition": "RNG_ISOLATION_PRODUCTION_CANDIDATE_SIMULATION_READY",
        "version": VERSION,
        "week3_outcomes_used": False,
        "sportsbook_inputs_to_rng_routing": 0,
        "football_parameters_changed": False,
        "legacy_m38_authority_exact": True,
        "legacy_m38_authority_keys": int(neutral["keys"]),
        "te_scope": te_scope,
        "wr_scope": wr_scope,
        "state_capture_parity": {
            "keys": int(state_parity["keys"]),
            "changed_arrays": int(state_parity["changed_arrays"]),
            "max_mean_gap": float(state_parity["max_mean_gap"]),
            "max_element_gap": float(state_parity["max_element_gap"]),
        },
        "qb_c2": qb_payload,
        "candidate_te_delta": str(CANDIDATE_TE_DELTA),
        "candidate_wr_delta": str(CANDIDATE_WR_DELTA),
        "baseline_authority_parity": str(CANDIDATE_BASELINE_AUTHORITY),
        "state_capture_parity_audit": str(CANDIDATE_STATE_PARITY),
        "production_merge_authorized": False,
    }
    CANDIDATE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_AUDIT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Preserve the normal entitlement audit and add candidate lineage without
    # replacing any specialist-science authority.
    if core.ENTITLEMENT_AUDIT.exists():
        ent = json.loads(core.ENTITLEMENT_AUDIT.read_text(encoding="utf-8"))
        ent.update(
            {
                "rng_isolation_candidate": True,
                "rng_isolation_candidate_version": VERSION,
                "rng_isolation_candidate_audit": str(CANDIDATE_AUDIT),
                "rng_isolation_te_protected_drift_keys": 0,
                "rng_isolation_wr_protected_drift_keys": 0,
                "rng_isolation_sportsbook_inputs_used": False,
            }
        )
        core.ENTITLEMENT_AUDIT.write_text(
            json.dumps(ent, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("[specialist_rng_isolation_production_candidate] " + json.dumps(payload, sort_keys=True))
    return selected


def main() -> int:
    base._identity_frame = v2._canonical_identity_frame
    base._validate_priced_distribution_coverage = (
        v2._install_provider_player_aliases_and_validate
    )
    base._build_full_universe = core._build_with_promoted_entitlement_specialists
    base.canonical_simulate = _simulate_promoted_stack
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
