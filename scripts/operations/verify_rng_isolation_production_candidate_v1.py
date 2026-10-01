#!/usr/bin/env python3
"""Verify the production RNG-isolation candidate against frozen Week-3 research.

No Week-3 outcomes are loaded. No sportsbook acquisition occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import load_weights
from scripts.modeling.qb_pass_synthesis_v1 import (
    load_artifact as load_qb_synthesis_artifact,
    load_player_logs as load_qb_player_logs,
    load_team_context as load_qb_team_context,
)
from scripts.operations.grade_market_track_record_v1 import _ev_roi
from scripts.operations.rng_isolation_candidate_verify_helpers_v1 import (
    SUPPORTED,
    compare_boards,
    install_provider_aliases,
    price_stage,
    provider_aliases,
    provider_identity_aliases,
    read_csv,
    representative_rule_rows,
)
from scripts.modeling.qb_c2_production_adapter_v1 import AUDIT_JSON as QB_C2_AUDIT_JSON
from scripts.run_pricing_with_full_roster_universe_v3_rng_isolation_candidate import (
    CANDIDATE_AUDIT,
    _simulate_promoted_stack,
)

ITERATIONS = 25000
TOL = 1e-12

EXPECTED = {
    "SHAPE_ONLY_FIXED_FINAL_MEAN": {
        "p99_abs_prob_delta": 0.01616000000000002,
        "max_abs_prob_delta": 0.03588000000000002,
        "p99_abs_ev_delta": 0.031741616000000014,
        "best_snapshot_bet_pass_flips": 6,
        "best_snapshot_identity_changes": 1,
        "best_ev_spearman": 0.9984915688405269,
        "top10_turnover": 0,
        "top25_turnover": 0,
    },
    "FULL_DOWNSTREAM_PROPAGATION": {
        "p99_abs_prob_delta": 0.010273600000000119,
        "max_abs_prob_delta": 0.015880000000000005,
        "p99_abs_ev_delta": 0.019283169750603693,
        "best_snapshot_bet_pass_flips": 4,
        "best_snapshot_identity_changes": 1,
        "best_ev_spearman": 0.9989391731762268,
        "top10_turnover": 0,
        "top25_turnover": 0,
    },
}


def _build_candidate_metrics(root: Path) -> pd.DataFrame:
    universe = read_csv(
        root / "data/football_simulation_universe.csv",
        "football simulation universe",
    )
    target = read_csv(
        root / "data/target_entitlement_v1_trace.csv",
        "target entitlement trace",
    )
    te = read_csv(
        root / "data/te_r5p_full_slate_entitlement_trace.csv",
        "TE-R5P trace",
    )

    keys = ["event_id", "team", "player_clean_key"]
    for frame, label in [(universe, "universe"), (target, "target"), (te, "TE")]:
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label} has duplicate candidate keys")

    required_target = {
        "m38_explicit_entitlement_tgt_share",
        "entitlement_tgt_share",
        "te_r5p_applied",
        "wr_r15_applied",
        "wr_r15_anchor",
    }
    missing = sorted(required_target - set(target.columns))
    if missing:
        raise RuntimeError(f"target trace missing candidate columns: {missing}")
    if "te_r5p_entitlement_tgt_share" not in te.columns:
        raise RuntimeError("TE trace missing te_r5p_entitlement_tgt_share")

    meta = target[
        keys
        + [
            "m38_explicit_entitlement_tgt_share",
            "entitlement_tgt_share",
            "te_r5p_applied",
            "wr_r15_applied",
            "wr_r15_anchor",
        ]
    ].merge(
        te[keys + ["te_r5p_entitlement_tgt_share"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    meta["baseline_entitlement_tgt_share"] = pd.to_numeric(
        meta["m38_explicit_entitlement_tgt_share"], errors="raise"
    ).astype(float)
    meta["te_only_entitlement_tgt_share"] = pd.to_numeric(
        meta["te_r5p_entitlement_tgt_share"], errors="coerce"
    ).combine_first(meta["baseline_entitlement_tgt_share"]).astype(float)
    meta["wr_r15_baseline_entitlement_tgt_share"] = meta[
        "te_only_entitlement_tgt_share"
    ].astype(float)
    meta["entitlement_tgt_share"] = pd.to_numeric(
        meta["entitlement_tgt_share"], errors="raise"
    ).astype(float)

    out = universe.merge(
        meta[
            keys
            + [
                "baseline_entitlement_tgt_share",
                "te_only_entitlement_tgt_share",
                "wr_r15_baseline_entitlement_tgt_share",
                "entitlement_tgt_share",
                "te_r5p_applied",
                "wr_r15_applied",
                "wr_r15_anchor",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    required_no_null = [
        "baseline_entitlement_tgt_share",
        "te_only_entitlement_tgt_share",
        "wr_r15_baseline_entitlement_tgt_share",
        "entitlement_tgt_share",
        "te_r5p_applied",
        "wr_r15_applied",
    ]
    if out[required_no_null].isna().any().any():
        raise RuntimeError("candidate metric reconstruction has null specialist authority")
    return out


def _paid_reference(paid: pd.DataFrame) -> pd.DataFrame:
    out = paid.copy()
    out["stage_ev_roi"] = [
        _ev_roi(p, o)
        for p, o in zip(
            pd.to_numeric(out["fair_prob"], errors="raise"),
            pd.to_numeric(out["vegas_odds"], errors="raise"),
        )
    ]
    return out


def _all_player_keys(board: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(e), str(p))
        for e, p in zip(board["event_id"], board["player_clean_key"])
    }


def _check_expected(summary: dict, surface: str) -> dict:
    expected = EXPECTED[surface]
    checks = {}
    for metric, want in expected.items():
        got = float(summary[metric])
        if isinstance(want, int):
            passed = int(round(got)) == int(want)
        else:
            passed = abs(got - float(want)) <= TOL
        checks[metric] = {
            "got": got,
            "expected": float(want),
            "pass": bool(passed),
        }
    checks["all_pass"] = all(v["pass"] for k, v in checks.items() if k != "all_pass")
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-run-id", default="36293274478")
    ap.add_argument("--source-artifact-id", default="10923570170")
    ap.add_argument(
        "--source-artifact-digest",
        default="sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _build_candidate_metrics(args.root)
    selected = _simulate_promoted_stack(metrics, iterations=ITERATIONS, seed=42)

    if not CANDIDATE_AUDIT.exists():
        raise RuntimeError("candidate simulation did not emit candidate audit")
    candidate_audit = json.loads(CANDIDATE_AUDIT.read_text(encoding="utf-8"))

    te_scope = candidate_audit["te_scope"]
    wr_scope = candidate_audit["wr_scope"]
    qb = candidate_audit["qb_c2"]

    frozen_scope_pass = bool(
        int(te_scope["protected_keys"]) == 1750
        and int(te_scope["protected_drift_keys"]) == 0
        and int(te_scope["intentional_receiving_arrays"]) == 172
        and int(te_scope["intentional_receiving_changed"]) == 172
        and int(wr_scope["protected_keys"]) == 1525
        and int(wr_scope["protected_drift_keys"]) == 0
        and int(wr_scope["intentional_receiving_arrays"]) == 262
        and int(wr_scope["intentional_receiving_changed"]) == 262
    )
    frozen_c2_pass = bool(
        int(qb["selected_qb_rows"]) == 30
        and int(qb["changed_simulation_keys"]) == 30
        and bool(qb["all_changed_keys_are_selected_qb_pass_yards"])
        and float(qb["max_raw_qb_mean_gap"]) <= 1e-10
        and float(qb["max_nonselected_element_gap"]) == 0.0
        and int(qb.get("sportsbook_inputs_to_rng_routing", 0)) == 0
    )
    if not QB_C2_AUDIT_JSON.exists():
        raise RuntimeError("candidate did not persist canonical QB C2 integration audit")
    canonical_qb_audit = json.loads(QB_C2_AUDIT_JSON.read_text(encoding="utf-8"))
    canonical_qb_audit_pass = bool(
        int(canonical_qb_audit.get("state_capture_changed_arrays", -1)) == 0
        and float(canonical_qb_audit.get("state_capture_max_mean_gap", np.inf)) <= 1e-12
        and float(canonical_qb_audit.get("state_capture_max_element_gap", np.inf)) <= 1e-12
        and bool(canonical_qb_audit.get("te_r5p_consumed_before_c2"))
        and bool(canonical_qb_audit.get("wr_r15_consumed_before_c2"))
        and bool(canonical_qb_audit.get("explicit_entitlement_consumed_before_c2"))
        and str(canonical_qb_audit.get("wr_r15_model_version", "")) == "WR_R15_PRODUCTION_MODEL_V1"
    )

    paid = read_csv(args.root / "outputs/props_priced_clean.csv", "paid priced board")
    paid = paid.loc[paid["market"].isin(sorted(SUPPORTED))].copy().reset_index(drop=True)
    paid["paid_row_id"] = np.arange(len(paid), dtype=int)
    paid_ref = _paid_reference(paid)

    aliases = provider_aliases(paid)
    identity_aliases = provider_identity_aliases(paid, aliases)
    install_provider_aliases(selected, aliases, identity_aliases)

    rule_rows = representative_rule_rows(args.root)
    weights = load_weights(Path("data/model_ensemble_weights.csv"))
    if weights.empty:
        raise RuntimeError("candidate verification missing model ensemble weights")
    qb_bundle = {
        "artifact": load_qb_synthesis_artifact(),
        "team_context": load_qb_team_context(),
        "player_logs": load_qb_player_logs(),
        "weather": pd.read_csv("data/weather_week.csv", low_memory=False)
        if Path("data/weather_week.csv").exists()
        else pd.DataFrame(),
    }
    boards = price_stage(selected, paid, rule_rows, weights, qb_bundle)

    summaries = []
    equivalence = {}
    keys = _all_player_keys(paid_ref)
    for surface in ("SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"):
        summary = compare_boards(
            paid_ref,
            boards[surface],
            keys,
            comparison="PAID_TO_PRODUCTION_RNG_ISOLATION_CANDIDATE",
            surface=surface,
            kind="PRODUCTION_CANDIDATE",
        )
        summaries.append(summary)
        equivalence[surface] = _check_expected(summary, surface)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "candidate_vs_paid_summary.csv", index=False)

    equivalence_pass = all(v["all_pass"] for v in equivalence.values())
    all_pass = bool(frozen_scope_pass and frozen_c2_pass and canonical_qb_audit_pass and equivalence_pass)
    disposition = (
        "RNG_ISOLATION_PRODUCTION_CANDIDATE_PASS"
        if all_pass
        else "RNG_ISOLATION_PRODUCTION_CANDIDATE_FAIL"
    )
    payload = {
        "version": "SPECIALIST_RNG_ISOLATION_PRODUCTION_REPAIR_V1",
        "disposition": disposition,
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_main_changed": False,
        "frozen_scope_equivalence_pass": frozen_scope_pass,
        "frozen_c2_equivalence_pass": frozen_c2_pass,
        "canonical_qb_audit_pass": canonical_qb_audit_pass,
        "canonical_qb_audit": canonical_qb_audit,
        "paid_board_research_equivalence_pass": equivalence_pass,
        "paid_board_equivalence": equivalence,
        "candidate_audit": candidate_audit,
        "merge_authorized": False,
    }
    (args.out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    print(summary_df.to_string(index=False))
    if not all_pass:
        raise RuntimeError(disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
