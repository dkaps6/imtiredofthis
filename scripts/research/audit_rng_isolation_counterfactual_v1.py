#!/usr/bin/env python3
"""Paid-board counterfactual for Specialist RNG Isolation V1.

No outcomes. No OddsAPI. No production mutation.

Compares the isolated final Week-3 simulation/C2 candidate to the actual preserved
paid board, then benchmarks that movement against ordinary alternate-seed
resampling of the unchanged current simulator.
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
from scripts.research.audit_qb_c2_rng_isolation_extension_v1 import (
    _anchor_map,
    apply_c2_isolated,
)
from scripts.research.audit_specialist_mc_downstream_materiality_v1 import (
    ALT_SEEDS,
    ITERATIONS,
    MEAN_REPLAY_TOL,
    PROB_REPLAY_TOL,
    SUPPORTED,
    TARGET_REPLAY_TOL,
    _build_entitlement_state,
    _compare_boards,
    _install_provider_aliases,
    _price_stage,
    _provider_aliases,
    _provider_identity_aliases,
    _read_csv,
    _representative_rule_rows,
    _simulate_stage as _simulate_current_stage,
    _stage_metrics,
)
from scripts.research.audit_specialist_rng_isolation_v1 import (
    _build_group_plan,
    simulate_isolated,
)

SURFACES = ("SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION")
GATE_METRICS = (
    "p99_abs_prob_delta",
    "p99_abs_ev_delta",
    "best_snapshot_bet_pass_flips",
    "best_snapshot_identity_changes",
    "top10_turnover",
    "top25_turnover",
)


def _paid_reference(paid: pd.DataFrame) -> pd.DataFrame:
    out = paid.copy()
    out["stage_ev_roi"] = [
        _ev_roi(p, o)
        for p, o in zip(
            pd.to_numeric(out["fair_prob"], errors="coerce"),
            pd.to_numeric(out["vegas_odds"], errors="coerce"),
        )
    ]
    if not np.isfinite(pd.to_numeric(out["stage_ev_roi"], errors="coerce")).all():
        raise RuntimeError("paid reference contains non-finite stage EV")
    return out


def _population_keys(board: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(e), str(p))
        for e, p in zip(board["event_id"], board["player_clean_key"])
    }


def _summary_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    kind: str,
    comparison: str,
    surface: str,
    seed: int | None = None,
) -> list[dict]:
    protected = _population_keys(left) | _population_keys(right)
    rows = []
    summary, _ = _compare_boards(
        left,
        right,
        protected,
        comparison=comparison,
        surface=surface,
        kind=kind,
        seed=seed,
    )
    summary["market_scope"] = "ALL_SUPPORTED"
    rows.append(summary)
    markets = sorted(set(left["market"].astype(str)) | set(right["market"].astype(str)))
    for market in markets:
        l = left.loc[left["market"].astype(str).eq(market)].copy()
        r = right.loc[right["market"].astype(str).eq(market)].copy()
        if l.empty or r.empty:
            raise RuntimeError(f"market population missing market={market}")
        mkeys = _population_keys(l) | _population_keys(r)
        s, _ = _compare_boards(
            l,
            r,
            mkeys,
            comparison=comparison,
            surface=surface,
            kind=kind,
            seed=seed,
        )
        s["market_scope"] = market
        rows.append(s)
    return rows


def _envelope(resampling: pd.DataFrame, surface: str) -> dict[str, float]:
    g = resampling.loc[
        resampling["surface"].eq(surface)
        & resampling["market_scope"].eq("ALL_SUPPORTED")
    ].copy()
    if len(g) != len(ALT_SEEDS):
        raise RuntimeError(
            f"ordinary resampling envelope expected {len(ALT_SEEDS)} rows "
            f"surface={surface}, got {len(g)}"
        )
    metrics = list(GATE_METRICS) + [
        "max_abs_prob_delta",
        "max_abs_ev_delta",
        "quote_preferred_side_flips",
        "quote_has_edge_pass_flips",
        "best_snapshot_side_flips",
        "mean_abs_best_ev_delta",
        "max_abs_best_ev_delta",
    ]
    return {
        m: float(pd.to_numeric(g[m], errors="raise").max())
        for m in metrics
    }


def _validate_reconstruction(
    paid: pd.DataFrame,
    reconstructed: dict[str, pd.DataFrame],
) -> dict:
    p = paid.set_index("paid_row_id")
    payload = {}
    for surface in SURFACES:
        r = reconstructed[surface].set_index("paid_row_id")
        if set(p.index) != set(r.index):
            raise RuntimeError(f"paid/reconstructed row-id mismatch surface={surface}")
        fair_gap = float(
            (
                pd.to_numeric(r["fair_prob"], errors="raise")
                - pd.to_numeric(p["fair_prob"], errors="raise")
            ).abs().max()
        )
        model_gap = float(
            (
                pd.to_numeric(r["stage_model_proj"], errors="raise")
                - pd.to_numeric(p["model_proj"], errors="raise")
            ).abs().max()
        )
        mc_gap = float(
            (
                pd.to_numeric(r["stage_mc_proj"], errors="raise")
                - pd.to_numeric(p["mc_proj"], errors="raise")
            ).abs().max()
        )
        payload[surface] = {
            "max_fair_prob_gap": fair_gap,
            "max_model_proj_gap": model_gap,
            "max_mc_proj_gap": mc_gap,
            "pass": bool(
                fair_gap <= PROB_REPLAY_TOL
                and model_gap <= TARGET_REPLAY_TOL
                and mc_gap <= MEAN_REPLAY_TOL
            ),
        }
    return payload


def _candidate_gate(
    candidate: pd.DataFrame,
    envelopes: dict[str, dict[str, float]],
) -> tuple[bool, dict]:
    results = {}
    all_pass = True
    for surface in SURFACES:
        row = candidate.loc[
            candidate["surface"].eq(surface)
            & candidate["market_scope"].eq("ALL_SUPPORTED")
        ]
        if len(row) != 1:
            raise RuntimeError(f"candidate summary missing unique surface={surface}")
        r = row.iloc[0]
        env = envelopes[surface]
        checks = {
            "best_ev_spearman_ge_0_99": bool(
                np.isfinite(float(r["best_ev_spearman"]))
                and float(r["best_ev_spearman"]) >= 0.99
            ),
            "top10_turnover_within_resampling": bool(
                float(r["top10_turnover"]) <= env["top10_turnover"]
            ),
            "top25_turnover_within_resampling": bool(
                float(r["top25_turnover"]) <= env["top25_turnover"]
            ),
            "best_snapshot_bet_pass_within_resampling": bool(
                float(r["best_snapshot_bet_pass_flips"])
                <= env["best_snapshot_bet_pass_flips"]
            ),
            "best_snapshot_identity_within_resampling": bool(
                float(r["best_snapshot_identity_changes"])
                <= env["best_snapshot_identity_changes"]
            ),
            "p99_prob_within_resampling": bool(
                float(r["p99_abs_prob_delta"]) <= env["p99_abs_prob_delta"] + 1e-15
            ),
            "p99_ev_within_resampling": bool(
                float(r["p99_abs_ev_delta"]) <= env["p99_abs_ev_delta"] + 1e-15
            ),
            "max_prob_le_5pct": bool(float(r["max_abs_prob_delta"]) <= 0.05 + 1e-15),
        }
        passed = all(checks.values())
        checks["surface_pass"] = passed
        results[surface] = {
            "checks": checks,
            "candidate": {
                k: float(r[k])
                for k in [
                    "p99_abs_prob_delta",
                    "max_abs_prob_delta",
                    "p99_abs_ev_delta",
                    "max_abs_ev_delta",
                    "best_snapshot_bet_pass_flips",
                    "best_snapshot_identity_changes",
                    "top10_turnover",
                    "top25_turnover",
                    "best_ev_spearman",
                ]
            },
            "ordinary_resampling_envelope": env,
        }
        all_pass = all_pass and passed
    return all_pass, results


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
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _build_entitlement_state(args.root)
    universe = _read_csv(
        args.root / "data/football_simulation_universe.csv", "football universe"
    )
    starters = _read_csv(
        args.root / "data/qb_c2_production_starter_audit.csv", "QB C2 starter audit"
    )
    if len(starters) != 30:
        raise RuntimeError(f"expected frozen 30 primary QBs, got {len(starters)}")

    paid = _read_csv(args.root / "outputs/props_priced_clean.csv", "paid priced board")
    paid = paid.loc[paid["market"].isin(sorted(SUPPORTED))].copy().reset_index(drop=True)
    paid["paid_row_id"] = np.arange(len(paid), dtype=int)
    if paid.empty:
        raise RuntimeError("no supported paid board rows")
    paid_ref = _paid_reference(paid)

    event_aliases = _provider_aliases(paid)
    identity_aliases = _provider_identity_aliases(paid, event_aliases)

    rule_rows = _representative_rule_rows(args.root)
    weights = load_weights(Path("data/model_ensemble_weights.csv"))
    if weights.empty:
        raise RuntimeError("model ensemble weights unavailable")
    qb_bundle = {
        "artifact": load_qb_synthesis_artifact(),
        "team_context": load_qb_team_context(),
        "player_logs": load_qb_player_logs(),
        "weather": pd.read_csv("data/weather_week.csv", low_memory=False)
        if Path("data/weather_week.csv").exists()
        else pd.DataFrame(),
    }

    final_metrics = _stage_metrics(
        universe, state, "final_entitlement", starters
    )

    # Current production-seed reconstruction is the integrity bridge to the
    # preserved paid board and is also the ordinary-resampling reference.
    _, current_seed42, _ = _simulate_current_stage(
        final_metrics, starters, seed=42
    )
    _install_provider_aliases(current_seed42, event_aliases, identity_aliases)
    current_boards = _price_stage(
        current_seed42, paid, rule_rows, weights, qb_bundle
    )
    reconstruction = _validate_reconstruction(paid, current_boards)
    reconstruction_pass = all(v["pass"] for v in reconstruction.values())

    # Isolated mechanical candidate.
    plan = _build_group_plan(state)
    isolated_base, isolated_core_meta = simulate_isolated(
        final_metrics, plan=plan, iterations=ITERATIONS, seed=42
    )
    isolated_c2, isolated_c2_diag, isolated_c2_meta = apply_c2_isolated(
        isolated_base,
        final_metrics,
        plan=plan,
        anchor_map=_anchor_map(isolated_base, starters),
    )
    _install_provider_aliases(isolated_c2, event_aliases, identity_aliases)
    candidate_boards = _price_stage(
        isolated_c2, paid, rule_rows, weights, qb_bundle
    )
    isolated_c2_diag.to_csv(out_dir / "isolated_c2_diag.csv", index=False)

    candidate_rows = []
    for surface in SURFACES:
        candidate_rows.extend(
            _summary_rows(
                paid_ref,
                candidate_boards[surface],
                kind="ISOLATED_CANDIDATE_VS_PAID",
                comparison="PAID_TO_ISOLATED_CANDIDATE",
                surface=surface,
            )
        )
    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary.to_csv(out_dir / "candidate_vs_paid_summary.csv", index=False)

    # Ordinary current-simulator finite-MC envelope.
    resampling_rows = []
    for seed in ALT_SEEDS:
        _, alt_current, _ = _simulate_current_stage(
            final_metrics, starters, seed=int(seed)
        )
        _install_provider_aliases(alt_current, event_aliases, identity_aliases)
        alt_boards = _price_stage(
            alt_current, paid, rule_rows, weights, qb_bundle
        )
        for surface in SURFACES:
            resampling_rows.extend(
                _summary_rows(
                    current_boards[surface],
                    alt_boards[surface],
                    kind="CURRENT_SIMULATOR_RESAMPLING",
                    comparison="SEED42_TO_ALT_SEED",
                    surface=surface,
                    seed=int(seed),
                )
            )
    resampling_summary = pd.DataFrame(resampling_rows)
    resampling_summary.to_csv(out_dir / "ordinary_resampling_summary.csv", index=False)

    envelopes = {surface: _envelope(resampling_summary, surface) for surface in SURFACES}
    counterfactual_pass, gate_detail = _candidate_gate(candidate_summary, envelopes)

    c2_integrity = bool(
        int(isolated_c2_meta["primary_qb_rows"]) == 30
        and int(isolated_c2_meta["changed_simulation_keys"]) == 30
        and bool(isolated_c2_meta["all_changed_keys_are_primary_qb_pass_yards"])
        and float(isolated_c2_meta["max_raw_mean_gap"]) <= 1e-10
        and float(isolated_c2_meta["max_nonselected_element_gap"]) == 0.0
        and not bool(isolated_c2_meta["sportsbook_inputs_used"])
    )
    core_integrity = bool(
        isolated_core_meta["target_total_violations"] == 0
        and isolated_core_meta["rush_total_violations"] == 0
        and isolated_core_meta["nonfinite_values"] == 0
    )
    integrity_pass = bool(reconstruction_pass and c2_integrity and core_integrity)

    if not integrity_pass:
        disposition = "RNG_ISOLATION_COUNTERFACTUAL_INTEGRITY_FAILURE"
    elif counterfactual_pass:
        disposition = "RNG_ISOLATION_COUNTERFACTUAL_WITHIN_ORDINARY_MC_ENVELOPE"
    else:
        disposition = "RNG_ISOLATION_COUNTERFACTUAL_EXCEEDS_ORDINARY_MC_ENVELOPE"

    payload = {
        "version": "SPECIALIST_RNG_ISOLATION_COUNTERFACTUAL_V1",
        "disposition": disposition,
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "integrity": {
            "paid_reconstruction": reconstruction,
            "paid_reconstruction_pass": reconstruction_pass,
            "isolated_core_integrity_pass": core_integrity,
            "isolated_c2_integrity_pass": c2_integrity,
            "all_integrity_pass": integrity_pass,
        },
        "candidate_gate_pass": counterfactual_pass,
        "candidate_gate_detail": gate_detail,
        "ordinary_resampling_seeds": ALT_SEEDS,
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_changed": False,
        "production_repair_candidate_authorized": bool(
            disposition == "RNG_ISOLATION_COUNTERFACTUAL_WITHIN_ORDINARY_MC_ENVELOPE"
        ),
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(payload, sort_keys=True))
    print("=== CANDIDATE VS PAID ===")
    print(
        candidate_summary.loc[
            candidate_summary["market_scope"].eq("ALL_SUPPORTED")
        ].to_string(index=False)
    )
    print("=== ORDINARY RESAMPLING ENVELOPES ===")
    print(json.dumps(envelopes, indent=2, sort_keys=True))

    if disposition == "RNG_ISOLATION_COUNTERFACTUAL_INTEGRITY_FAILURE":
        raise RuntimeError(disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
