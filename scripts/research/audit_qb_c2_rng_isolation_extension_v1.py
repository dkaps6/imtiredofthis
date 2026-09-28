#!/usr/bin/env python3
"""QB C2 RNG Isolation Extension V1.

Research-only continuation of SPECIALIST_RNG_ISOLATION_CORE_PASS.
No Week-3 outcomes. No OddsAPI acquisition. No production mutation.
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
from scripts.research.audit_specialist_mc_downstream_materiality_v1 import (
    C2_SEED,
    ITERATIONS,
    SUPPORTED,
    _build_entitlement_state,
    _compare_boards,
    _install_provider_aliases,
    _price_stage,
    _protected_sets,
    _provider_aliases,
    _provider_identity_aliases,
    _read_csv,
    _representative_rule_rows,
    _stage_metrics,
)
from scripts.research.audit_specialist_rng_isolation_v1 import (
    DIST_SEEDS,
    _build_group_plan,
    _hierarchical_targets,
    _rng,
    simulate_isolated,
)
from scripts.simulation_c2_qb_candidate import (
    C2_RESIDUAL_CATCH_RATE,
    C2_RESIDUAL_YPT,
    C2_YPR_MAX,
    C2_YPR_MIN,
    PASS_CATCHER_POSITIONS,
    StateSimulationResult,
    _primary_qb_row,
    apply_c2,
    simulate_with_states,
)
from scripts.simulation_v2 import _clip_prob, _num, _player_key

MEAN_TOL = 1e-10


def apply_c2_isolated(
    base: StateSimulationResult,
    metrics: pd.DataFrame,
    *,
    plan: dict,
    anchor_map: dict[tuple[str, str], float],
    seed: int = C2_SEED,
) -> tuple[StateSimulationResult, pd.DataFrame, dict]:
    values = {k: np.asarray(v, dtype=float).copy() for k, v in base.values.items()}
    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)

    game_key = "event_id" if "event_id" in frame.columns and frame["event_id"].notna().any() else None
    if game_key is None:
        frame["_game_key"] = frame.apply(
            lambda r: "|".join(sorted([str(r.get("team", "")), str(r.get("opponent", ""))])),
            axis=1,
        )
        game_key = "_game_key"

    players = frame.sort_values([game_key, "team", "player_clean_key"]).drop_duplicates(
        [game_key, "team", "player_clean_key"], keep="last"
    )
    rows = []
    allowed_changed = set()

    for game, gdf in players.groupby(game_key, dropna=False, sort=True):
        gs = str(game)
        for team, tdf0 in gdf.groupby("team", dropna=False, sort=True):
            if pd.isna(team) or not str(team).strip():
                continue
            ts = str(team)
            tdf = tdf0.sort_values("player_clean_key", kind="mergesort").reset_index(drop=True)
            pass_att = np.asarray(base.team_states[(gs, ts, "pass_att")], dtype=int)
            pass_eff = np.asarray(base.team_states[(gs, ts, "pass_eff_shock")], dtype=float)

            positions = (
                tdf.get("position", pd.Series("", index=tdf.index))
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )
            catcher_mask = positions.isin(PASS_CATCHER_POSITIONS).to_numpy()
            catcher_df = tdf.loc[catcher_mask].reset_index(drop=True)
            if catcher_df.empty:
                raise RuntimeError(f"C2 isolated has no pass catchers game={gs} team={ts}")
            shares = pd.to_numeric(
                catcher_df["entitlement_tgt_share"], errors="raise"
            ).to_numpy(float)

            targets, _ = _hierarchical_targets(
                game=gs,
                team=ts,
                team_df=catcher_df,
                pass_att=pass_att,
                stage_shares=shares,
                plan=plan,
                base_seed=int(seed),
            )
            residual_targets = np.maximum(0, pass_att - targets.sum(axis=1))

            receiver_yards = []
            for j, (_, row) in enumerate(catcher_df.iterrows()):
                pkey = _player_key(row)
                if not pkey:
                    continue
                catch = _clip_prob(
                    _num(
                        row,
                        "rules_catch_rate",
                        "bayes_receptions_per_target",
                        "receptions_per_target",
                        "catch_rate",
                        default=C2_RESIDUAL_CATCH_RATE,
                    ),
                    C2_RESIDUAL_CATCH_RATE,
                )
                recs = _rng(seed, "C2_CATCH", gs, ts, pkey).binomial(targets[:, j], catch)
                ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
                ypt = C2_RESIDUAL_YPT if not np.isfinite(ypt) or ypt <= 0 else float(ypt)
                ypr = float(np.clip(ypt / catch, C2_YPR_MIN, C2_YPR_MAX))
                vol = float(np.clip(_num(row, "rules_volatility_mult", default=1.0), 0.75, 1.50))
                mu = recs.astype(float) * ypr * pass_eff
                sd = np.maximum(3.0, np.sqrt(np.maximum(recs, 1)) * ypr * 0.55) * vol
                y = np.clip(
                    _rng(seed, "C2_REC_YARDS", gs, ts, pkey).normal(mu, sd),
                    0.0,
                    None,
                )
                y = np.where(recs > 0, y, 0.0)
                receiver_yards.append(y)

            residual_catch_rng = _rng(seed, "C2_RESIDUAL_CATCH", gs, ts)
            residual_recs = residual_catch_rng.binomial(
                residual_targets, C2_RESIDUAL_CATCH_RATE
            )
            residual_ypr = C2_RESIDUAL_YPT / C2_RESIDUAL_CATCH_RATE
            residual_mu = residual_recs.astype(float) * residual_ypr * pass_eff
            residual_sd = np.maximum(
                3.0,
                np.sqrt(np.maximum(residual_recs, 1)) * residual_ypr * 0.55,
            )
            residual_yards = np.where(
                residual_recs > 0,
                np.clip(
                    _rng(seed, "C2_RESIDUAL_YARDS", gs, ts).normal(
                        residual_mu, residual_sd
                    ),
                    0.0,
                    None,
                ),
                0.0,
            )

            raw_total = (
                np.sum(np.vstack(receiver_yards), axis=0)
                if receiver_yards
                else np.zeros(base.iterations)
            ) + residual_yards

            anchor = float(anchor_map.get((gs, ts), np.nan))
            raw_mean = float(np.mean(raw_total)) if len(raw_total) else np.nan
            if not np.isfinite(anchor) or anchor <= 0:
                raise RuntimeError(f"invalid isolated C2 anchor game={gs} team={ts} anchor={anchor}")
            if not np.isfinite(raw_mean) or raw_mean <= 0:
                raise RuntimeError(f"invalid isolated C2 raw mean game={gs} team={ts} mean={raw_mean}")

            qb_row = _primary_qb_row(tdf)
            if qb_row is None:
                raise RuntimeError(f"isolated C2 primary QB missing game={gs} team={ts}")
            qb_key = _player_key(qb_row)
            if not qb_key:
                raise RuntimeError(f"isolated C2 blank primary QB key game={gs} team={ts}")

            scale = anchor / raw_mean
            qb_yards = raw_total * scale
            key = (gs, qb_key, "pass_yards")
            canonical = np.asarray(base.values.get(key), dtype=float)
            if canonical.shape != qb_yards.shape:
                raise RuntimeError(f"isolated C2 QB array shape mismatch key={key}")
            if not np.isfinite(qb_yards).all():
                raise RuntimeError(f"isolated C2 non-finite QB array key={key}")
            mean_gap = float(qb_yards.mean() - canonical.mean())
            if abs(mean_gap) > MEAN_TOL:
                raise RuntimeError(
                    f"isolated C2 mean neutrality failed key={key} gap={mean_gap}"
                )
            values[key] = qb_yards
            allowed_changed.add(key)
            rows.append(
                {
                    "event_id": gs,
                    "team": ts,
                    "player_clean_key": qb_key,
                    "canonical_raw_mean": float(canonical.mean()),
                    "isolated_c2_raw_mean": float(qb_yards.mean()),
                    "raw_mean_gap": mean_gap,
                    "canonical_raw_sd": float(np.std(canonical, ddof=1)),
                    "isolated_c2_raw_sd": float(np.std(qb_yards, ddof=1)),
                    "canonical_p10": float(np.quantile(canonical, 0.10)),
                    "canonical_p50": float(np.quantile(canonical, 0.50)),
                    "canonical_p90": float(np.quantile(canonical, 0.90)),
                    "isolated_c2_p10": float(np.quantile(qb_yards, 0.10)),
                    "isolated_c2_p50": float(np.quantile(qb_yards, 0.50)),
                    "isolated_c2_p90": float(np.quantile(qb_yards, 0.90)),
                }
            )

    changed = set()
    max_nonselected_gap = 0.0
    for key, arr in base.values.items():
        a = np.asarray(arr, dtype=float)
        b = np.asarray(values[key], dtype=float)
        gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
        if gap > 0:
            changed.add(key)
        if key not in allowed_changed:
            max_nonselected_gap = max(max_nonselected_gap, gap)
    illegal = changed - allowed_changed
    if illegal or max_nonselected_gap > 0:
        raise RuntimeError(
            f"isolated C2 changed non-QB/non-primary arrays illegal={list(illegal)[:20]} "
            f"max_nonselected_gap={max_nonselected_gap}"
        )

    diag = pd.DataFrame(rows).sort_values(["event_id", "team"]).reset_index(drop=True)
    payload = {
        "primary_qb_rows": int(len(diag)),
        "changed_simulation_keys": int(len(changed)),
        "all_changed_keys_are_primary_qb_pass_yards": bool(not illegal),
        "max_raw_mean_gap": float(diag["raw_mean_gap"].abs().max()) if len(diag) else np.nan,
        "max_nonselected_element_gap": max_nonselected_gap,
        "sportsbook_inputs_used": False,
    }
    return StateSimulationResult(values, base.iterations, base.team_states), diag, payload


def _anchor_map(
    base: StateSimulationResult,
    starters: pd.DataFrame,
) -> dict[tuple[str, str], float]:
    out = {}
    for r in starters.itertuples(index=False):
        key = (str(r.event_id), str(r.primary_player_clean_key), "pass_yards")
        arr = base.values.get(key)
        if arr is None or len(arr) != base.iterations:
            raise RuntimeError(f"missing primary QB base array key={key}")
        mean = float(np.mean(np.asarray(arr, dtype=float)))
        if not np.isfinite(mean) or mean <= 0:
            raise RuntimeError(f"invalid primary QB base mean key={key} mean={mean}")
        out[(str(r.event_id), str(r.team))] = mean
    return out


def _simulate_stage(
    metrics: pd.DataFrame,
    starters: pd.DataFrame,
    plan: dict,
    *,
    seed: int,
) -> tuple[StateSimulationResult, StateSimulationResult, pd.DataFrame, dict]:
    base, core_meta = simulate_isolated(
        metrics, plan=plan, iterations=ITERATIONS, seed=int(seed)
    )
    selected, diag, c2_meta = apply_c2_isolated(
        base,
        metrics,
        plan=plan,
        anchor_map=_anchor_map(base, starters),
        seed=C2_SEED,
    )
    c2_meta["core_meta"] = core_meta
    return base, selected, diag, c2_meta


def _zero_nonqb_gate(summary: dict) -> bool:
    exact_zero = [
        "max_abs_prob_delta",
        "max_abs_ev_delta",
        "quote_preferred_side_flips",
        "quote_has_edge_pass_flips",
        "best_snapshot_bet_pass_flips",
        "best_snapshot_side_flips",
        "best_snapshot_identity_changes",
        "top10_turnover",
        "top25_turnover",
        "max_abs_best_ev_delta",
    ]
    for key in exact_zero:
        value = float(summary.get(key, np.nan))
        if not np.isfinite(value) or abs(value) > 1e-15:
            return False
    return True


def _current_vs_isolated_c2_compatibility(
    metrics: pd.DataFrame,
    starters: pd.DataFrame,
    plan: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in DIST_SEEDS:
        current_base = simulate_with_states(metrics, iterations=ITERATIONS, seed=int(seed))
        current = apply_c2(
            current_base,
            metrics,
            anchor_map=_anchor_map(current_base, starters),
            seed=C2_SEED,
        )
        isolated_base, _ = simulate_isolated(
            metrics, plan=plan, iterations=ITERATIONS, seed=int(seed)
        )
        isolated, _, _ = apply_c2_isolated(
            isolated_base,
            metrics,
            plan=plan,
            anchor_map=_anchor_map(isolated_base, starters),
            seed=C2_SEED,
        )
        for r in starters.itertuples(index=False):
            key = (str(r.event_id), str(r.primary_player_clean_key), "pass_yards")
            a = np.asarray(current.values[key], dtype=float)
            b = np.asarray(isolated.values[key], dtype=float)
            rows.append(
                {
                    "seed": int(seed),
                    "event_id": str(r.event_id),
                    "team": str(r.team),
                    "player_clean_key": str(r.primary_player_clean_key),
                    "abs_mean_diff": abs(float(a.mean() - b.mean())),
                    "abs_sd_diff": abs(float(np.std(a, ddof=1) - np.std(b, ddof=1))),
                    "abs_p10_diff": abs(float(np.quantile(a, .10) - np.quantile(b, .10))),
                    "abs_p50_diff": abs(float(np.quantile(a, .50) - np.quantile(b, .50))),
                    "abs_p90_diff": abs(float(np.quantile(a, .90) - np.quantile(b, .90))),
                }
            )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "rows": int(len(detail)),
                "mean_abs_mean_diff": float(detail["abs_mean_diff"].mean()),
                "p95_abs_mean_diff": float(detail["abs_mean_diff"].quantile(.95)),
                "max_abs_mean_diff": float(detail["abs_mean_diff"].max()),
                "mean_abs_sd_diff": float(detail["abs_sd_diff"].mean()),
                "p95_abs_sd_diff": float(detail["abs_sd_diff"].quantile(.95)),
                "max_abs_sd_diff": float(detail["abs_sd_diff"].max()),
                "mean_abs_p10_diff": float(detail["abs_p10_diff"].mean()),
                "mean_abs_p50_diff": float(detail["abs_p50_diff"].mean()),
                "mean_abs_p90_diff": float(detail["abs_p90_diff"].mean()),
                "max_abs_p10_diff": float(detail["abs_p10_diff"].max()),
                "max_abs_p50_diff": float(detail["abs_p50_diff"].max()),
                "max_abs_p90_diff": float(detail["abs_p90_diff"].max()),
            }
        ]
    )
    return detail, summary


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
    universe = _read_csv(args.root / "data/football_simulation_universe.csv", "football universe")
    starters = _read_csv(args.root / "data/qb_c2_production_starter_audit.csv", "QB C2 starter audit")
    if len(starters) != 30:
        raise RuntimeError(f"expected frozen 30 primary QBs, got {len(starters)}")

    paid = _read_csv(args.root / "outputs/props_priced_clean.csv", "paid priced board")
    paid = paid.loc[paid["market"].isin(sorted(SUPPORTED))].copy().reset_index(drop=True)
    paid["paid_row_id"] = np.arange(len(paid), dtype=int)
    if paid.empty:
        raise RuntimeError("no supported paid rows")

    event_aliases = _provider_aliases(paid)
    identity_aliases = _provider_identity_aliases(paid, event_aliases)
    protected = _protected_sets(state, event_aliases, identity_aliases)
    plan = _build_group_plan(state)

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

    metrics = {
        "m38": _stage_metrics(universe, state, "m38_entitlement", starters),
        "te": _stage_metrics(universe, state, "te_entitlement", starters),
        "wr": _stage_metrics(universe, state, "final_entitlement", starters),
    }

    stage_boards = {}
    c2_meta = {}
    c2_diag = {}
    for name in ("m38", "te", "wr"):
        _, selected, diag, meta = _simulate_stage(
            metrics[name], starters, plan, seed=42
        )
        _install_provider_aliases(selected, event_aliases, identity_aliases)
        stage_boards[name] = _price_stage(
            selected, paid, rule_rows, weights, qb_bundle
        )
        c2_meta[name] = meta
        c2_diag[name] = diag
        diag.to_csv(out_dir / f"c2_{name}_diag.csv", index=False)

    comparisons = [
        ("M38_TO_TE_R5P", "m38", "te", "TE_R5P_PROTECTED"),
        ("TE_R5P_TO_WR_R15", "te", "wr", "WR_R15_PROTECTED"),
    ]
    full_rows = []
    nonqb_rows = []
    qb_rows = []
    for cname, left, right, scope in comparisons:
        for surface in ("SHAPE_ONLY_FIXED_FINAL_MEAN", "FULL_DOWNSTREAM_PROPAGATION"):
            s, _ = _compare_boards(
                stage_boards[left][surface],
                stage_boards[right][surface],
                protected[scope],
                comparison=cname,
                surface=surface,
                kind="ISOLATED_FULL_PROTECTED",
            )
            s["protected_scope"] = scope
            full_rows.append(s)

            lb = stage_boards[left][surface]
            rb = stage_boards[right][surface]
            l_nonqb = lb.loc[lb["market"].ne("pass_yards")].copy()
            r_nonqb = rb.loc[rb["market"].ne("pass_yards")].copy()
            ns, _ = _compare_boards(
                l_nonqb,
                r_nonqb,
                protected[scope],
                comparison=cname,
                surface=surface,
                kind="ISOLATED_NONQB_PROTECTED",
            )
            ns["protected_scope"] = scope
            nonqb_rows.append(ns)

            l_qb = lb.loc[lb["market"].eq("pass_yards")].copy()
            r_qb = rb.loc[rb["market"].eq("pass_yards")].copy()
            qs, _ = _compare_boards(
                l_qb,
                r_qb,
                protected[scope],
                comparison=cname,
                surface=surface,
                kind="ISOLATED_QB_PASS",
            )
            qs["protected_scope"] = scope
            qb_rows.append(qs)

    full_summary = pd.DataFrame(full_rows)
    nonqb_summary = pd.DataFrame(nonqb_rows)
    qb_summary = pd.DataFrame(qb_rows)
    full_summary.to_csv(out_dir / "full_protected_board_summary.csv", index=False)
    nonqb_summary.to_csv(out_dir / "nonqb_protected_board_summary.csv", index=False)
    qb_summary.to_csv(out_dir / "qb_pass_board_summary.csv", index=False)

    compat_detail, compat_summary = _current_vs_isolated_c2_compatibility(
        metrics["wr"], starters, plan
    )
    compat_detail.to_csv(out_dir / "c2_distribution_compatibility_detail.csv", index=False)
    compat_summary.to_csv(out_dir / "c2_distribution_compatibility_summary.csv", index=False)

    c2_integrity = all(
        int(c2_meta[name]["primary_qb_rows"]) == 30
        and int(c2_meta[name]["changed_simulation_keys"]) == 30
        and bool(c2_meta[name]["all_changed_keys_are_primary_qb_pass_yards"])
        and float(c2_meta[name]["max_raw_mean_gap"]) <= MEAN_TOL
        and float(c2_meta[name]["max_nonselected_element_gap"]) == 0.0
        and not bool(c2_meta[name]["sportsbook_inputs_used"])
        for name in ("m38", "te", "wr")
    )
    nonqb_exact = all(_zero_nonqb_gate(row._asdict()) for row in nonqb_summary.itertuples(index=False))
    disposition = (
        "QB_C2_RNG_ISOLATION_DOWNSTREAM_PASS"
        if c2_integrity and nonqb_exact
        else "QB_C2_RNG_ISOLATION_DOWNSTREAM_FAIL"
    )

    payload = {
        "version": "QB_C2_RNG_ISOLATION_EXTENSION_V1",
        "disposition": disposition,
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "c2_integrity_pass": c2_integrity,
        "nonqb_protected_downstream_exact_pass": nonqb_exact,
        "c2_stage_meta": c2_meta,
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_changed": False,
        "production_repair_authorized": False,
        "next_if_pass": "FREEZE_SPECIALIST_RNG_ISOLATION_PRODUCTION_REPAIR_CANDIDATE_PLAN",
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    print("=== NON-QB PROTECTED ===")
    print(nonqb_summary.to_string(index=False))
    print("=== QB PASS ===")
    print(qb_summary.to_string(index=False))
    print("=== C2 COMPATIBILITY ===")
    print(compat_summary.to_string(index=False))

    if disposition != "QB_C2_RNG_ISOLATION_DOWNSTREAM_PASS":
        raise RuntimeError(disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
