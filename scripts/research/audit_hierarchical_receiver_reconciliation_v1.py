#!/usr/bin/env python3
"""Read-only audit for hierarchical receiver reconciliation V1.

No target-game outcomes are read. No production arrays are modified.

For current C2-selected teams, this audit measures whether the QB/team passing
mean can be made coherent with canonical named receiver means by assigning the
difference to the existing residual receiving bucket first. Only when named
receiver means alone exceed the QB mean does it compute a research-only
nonnegative weighted projection using existing Bayesian pregame uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_pricing_with_full_roster_universe_v1 as base
import scripts.run_pricing_with_full_roster_universe_v2 as v2
import scripts.run_pricing_with_full_roster_universe_v3_core as v3
from scripts._opponent_map import canon_team
from scripts.modeling.qb_c2_production_adapter_v1 import (
    annotate_primary_qbs,
    apply_qb_c2_selector,
)
from scripts.research.audit_shared_pass_state_coherence_v1 import (
    _build_synthetic_pricing_metrics,
    _capture_c2_shadow,
)
from scripts.simulation_c2_qb_candidate import PASS_CATCHER_POSITIONS
from scripts.simulation_c2_qb_candidate import simulate_with_states

OUT = Path("data/research/hierarchical_receiver_reconciliation_v1")
SUMMARY = OUT / "summary.json"
TEAM_CSV = OUT / "team_reconciliation_diagnostic.csv"
PLAYER_CSV = OUT / "player_reconciliation_diagnostic.csv"

ITERATIONS = 5000
BASE_SEED = 42
TOL = 1e-10


def _position_family(value: object) -> str:
    p = str(value or "").upper().strip()
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("QB"):
        return "QB"
    return p or "OTHER"


def _finite_float(value: object, label: str) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise RuntimeError(f"non-finite {label}: {value}")
    return x


def _weighted_nonnegative_projection(
    base_means: np.ndarray,
    variances: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    """Weighted Euclidean projection onto x>=0 and sum(x)=target_sum.

    Objective: sum((x_i-b_i)^2 / V_i).
    Movement is therefore proportional to V_i among the currently active set.
    """
    b = np.asarray(base_means, dtype=float)
    v = np.asarray(variances, dtype=float)
    target = float(target_sum)

    if not np.isfinite(b).all() or (b < -TOL).any():
        raise RuntimeError("invalid base means for reconciliation")
    if not np.isfinite(v).all() or (v <= 0).any():
        raise RuntimeError("weighted reconciliation requires strictly positive finite variances")
    if not np.isfinite(target) or target < -TOL or target > float(b.sum()) + TOL:
        raise RuntimeError(
            f"invalid downward-reconciliation target target={target} base_sum={float(b.sum())}"
        )
    if abs(target - float(b.sum())) <= TOL:
        return b.copy()
    if target <= TOL:
        return np.zeros_like(b)

    x = np.zeros_like(b)
    active = np.ones(len(b), dtype=bool)
    remaining_target = target

    while active.any():
        idx = np.flatnonzero(active)
        ba = b[idx]
        va = v[idx]
        reduction = float(ba.sum() - remaining_target)
        if reduction < -TOL:
            raise RuntimeError("active-set target exceeds active base sum")
        trial = ba - reduction * va / float(va.sum())
        negative = trial < 0
        if not negative.any():
            x[idx] = np.maximum(trial, 0.0)
            break
        # Variables that would cross below zero bind at zero. Since fixed
        # variables contribute zero, the same remaining target is allocated
        # across the reduced active set.
        bind = idx[negative]
        x[bind] = 0.0
        active[bind] = False
        if not active.any() and remaining_target > TOL:
            raise RuntimeError("active-set exhausted before satisfying target")

    gap = float(x.sum() - target)
    if abs(gap) > TOL:
        raise RuntimeError(f"weighted projection identity failed gap={gap}")
    if (x < -TOL).any():
        raise RuntimeError("weighted projection produced negative means")
    return np.maximum(x, 0.0)


def _spearman(df: pd.DataFrame, x: str, y: str) -> float | None:
    z = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 3 or z[x].nunique() < 2 or z[y].nunique() < 2:
        return None
    val = z[x].corr(z[y], method="spearman")
    return float(val) if pd.notna(val) else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    synthetic = _build_synthetic_pricing_metrics()
    base._identity_frame = v2._canonical_identity_frame
    universe, _, universe_audit = v3._build_with_promoted_entitlement_specialists(synthetic)

    seasons = sorted(pd.to_numeric(universe["season"], errors="coerce").dropna().astype(int).unique())
    weeks = sorted(pd.to_numeric(universe["week"], errors="coerce").dropna().astype(int).unique())
    if len(seasons) != 1 or len(weeks) != 1:
        raise RuntimeError(f"expected one current season/week, got seasons={seasons} weeks={weeks}")
    season, week = seasons[0], weeks[0]

    required_uncertainty = {
        "bayes_tgt_share_sd",
        "bayes_tgt_share_effective_n",
        "bayes_ypt",
        "bayes_ypt_sd",
        "bayes_ypt_effective_n",
        "bayes_evidence_state",
        "entitlement_tgt_share",
    }
    missing = sorted(required_uncertainty - set(universe.columns))
    if missing:
        raise RuntimeError(f"current football universe missing uncertainty fields: {missing}")

    stateful = simulate_with_states(universe, iterations=ITERATIONS, seed=BASE_SEED)
    selected_state, selector_audit, selector_payload = apply_qb_c2_selector(
        stateful, universe, season=season, week=week
    )
    annotated, _ = annotate_primary_qbs(universe, season=season, week=week)
    primary = annotated.loc[
        pd.to_numeric(annotated["qb_projection_eligible"], errors="coerce").eq(1)
    ].copy()
    primary_by_team = {canon_team(r.team): r for r in primary.itertuples(index=False)}

    anchors: dict[tuple[str, str], float] = {}
    for team, row in primary_by_team.items():
        game = str(row.event_id)
        pkey = str(row.player_clean_key)
        arr = np.asarray(stateful.values[(game, pkey, "pass_yards")], dtype=float)
        anchors[(game, team)] = float(arr.mean())
    shadow = _capture_c2_shadow(stateful, annotated, anchors)

    selected_teams = set(
        selector_audit.loc[selector_audit["selector_c2_selected"].eq(1), "team"]
        .astype(str)
        .map(canon_team)
    )
    if not selected_teams:
        raise RuntimeError("current C2 selector selected zero teams")

    frame = universe.copy()
    frame["team"] = frame["team"].map(canon_team)
    frame["position_family"] = frame["position"].map(_position_family)

    team_rows: list[dict] = []
    player_rows: list[dict] = []
    max_weighted_identity_gap = 0.0

    for team in sorted(selected_teams):
        qrow = primary_by_team.get(team)
        if qrow is None:
            raise RuntimeError(f"selected team missing primary QB: {team}")
        game = str(qrow.event_id)
        qpk = str(qrow.player_clean_key)
        qb_arr = np.asarray(selected_state.values[(game, qpk, "pass_yards")], dtype=float)
        q_mean = float(qb_arr.mean())
        if not np.isfinite(q_mean) or q_mean <= 0:
            raise RuntimeError(f"invalid selected QB mean team={team}: {q_mean}")

        part = frame.loc[
            frame["team"].eq(team)
            & frame["event_id"].astype(str).eq(game)
            & frame["position"].astype(str).str.upper().isin(PASS_CATCHER_POSITIONS)
        ].copy()
        if part.empty:
            raise RuntimeError(f"selected team has no pass catchers: {team}")

        means = []
        variances = []
        row_meta = []
        pass_att_mean = float(
            np.mean(np.asarray(stateful.team_states[(game, team, "pass_att")], dtype=float))
        )
        if not np.isfinite(pass_att_mean) or pass_att_mean <= 0:
            raise RuntimeError(f"invalid pass-attempt mean team={team}")

        for _, row in part.iterrows():
            pk = str(row["player_clean_key"])
            arr = stateful.values.get((game, pk, "rec_yards"))
            if arr is None:
                raise RuntimeError(f"missing canonical receiver array team={team} player={pk}")
            bmean = float(np.mean(np.asarray(arr, dtype=float)))
            share = _finite_float(row["entitlement_tgt_share"], f"{team}/{pk}/entitlement")
            share_sd = _finite_float(row["bayes_tgt_share_sd"], f"{team}/{pk}/tgt_share_sd")
            ypt = _finite_float(row["bayes_ypt"], f"{team}/{pk}/bayes_ypt")
            ypt_sd = _finite_float(row["bayes_ypt_sd"], f"{team}/{pk}/ypt_sd")
            tgt_n = _finite_float(
                row["bayes_tgt_share_effective_n"], f"{team}/{pk}/tgt_effective_n"
            )
            ypt_n = _finite_float(row["bayes_ypt_effective_n"], f"{team}/{pk}/ypt_effective_n")
            if share > 0 and (share_sd <= 0 or ypt_sd <= 0 or tgt_n <= 0 or ypt_n <= 0):
                raise RuntimeError(
                    f"invalid positive-entitlement uncertainty team={team} player={pk}"
                )

            variance = (
                (pass_att_mean * ypt * share_sd) ** 2
                + (pass_att_mean * share * ypt_sd) ** 2
            )
            # Zero-entitlement / zero-variance depth rows carry no modeled
            # receiving mean and do not need adjustment capacity. Give them a
            # numerically tiny positive variance only if their base mean is zero.
            if variance <= 0:
                if bmean > TOL:
                    raise RuntimeError(
                        f"positive canonical mean has zero propagated variance team={team} player={pk}"
                    )
                variance = 1e-18

            means.append(bmean)
            variances.append(float(variance))
            row_meta.append(
                {
                    "season": season,
                    "week": week,
                    "event_id": game,
                    "team": team,
                    "player": str(row["player"]),
                    "player_clean_key": pk,
                    "position_family": _position_family(row["position"]),
                    "canonical_mean": bmean,
                    "entitlement_tgt_share": share,
                    "bayes_tgt_share_sd": share_sd,
                    "bayes_tgt_share_effective_n": tgt_n,
                    "bayes_ypt": ypt,
                    "bayes_ypt_sd": ypt_sd,
                    "bayes_ypt_effective_n": ypt_n,
                    "bayes_evidence_state": str(row["bayes_evidence_state"]),
                    "propagated_variance": float(variance),
                    "pass_att_mean": pass_att_mean,
                }
            )

        b = np.asarray(means, dtype=float)
        v = np.asarray(variances, dtype=float)
        named_sum = float(b.sum())
        gap = float(q_mean - named_sum)
        residual_only = gap >= -TOL

        if residual_only:
            reconciled = b.copy()
            implied_residual = max(0.0, gap)
            required_named_reduction = 0.0
        else:
            implied_residual = 0.0
            required_named_reduction = float(named_sum - q_mean)
            reconciled = _weighted_nonnegative_projection(b, v, q_mean)
            ident = float(reconciled.sum() - q_mean)
            max_weighted_identity_gap = max(max_weighted_identity_gap, abs(ident))

        sh = shadow[(game, team)]
        existing_residual = float(np.mean(np.asarray(sh["residual_yards_scaled"], dtype=float)))

        team_rows.append(
            {
                "season": season,
                "week": week,
                "event_id": game,
                "team": team,
                "qb_player": str(qrow.player),
                "qb_mean": q_mean,
                "canonical_named_receiver_sum": named_sum,
                "qb_minus_named_gap": gap,
                "named_to_qb_ratio": float(named_sum / q_mean),
                "existing_c2_residual_mean": existing_residual,
                "existing_c2_residual_share": float(existing_residual / q_mean),
                "implied_residual_first_mean": implied_residual,
                "implied_residual_first_share": float(implied_residual / q_mean),
                "residual_only_feasible": bool(residual_only),
                "required_named_reduction_yards": required_named_reduction,
                "required_named_reduction_pct": (
                    float(required_named_reduction / named_sum) if named_sum > 0 else 0.0
                ),
                "weighted_reconciled_named_sum": float(reconciled.sum()),
                "weighted_identity_gap": (
                    float(reconciled.sum() - q_mean) if not residual_only else 0.0
                ),
            }
        )

        for meta, new_mean in zip(row_meta, reconciled):
            old = float(meta["canonical_mean"])
            adjustment = float(new_mean - old)
            pct = float(adjustment / old) if old > TOL else 0.0
            meta.update(
                {
                    "reconciled_mean": float(new_mean),
                    "adjustment_yards": adjustment,
                    "abs_adjustment_yards": abs(adjustment),
                    "adjustment_pct": pct,
                    "abs_adjustment_pct": abs(pct),
                    "residual_only_feasible_team": bool(residual_only),
                    "team_qb_mean": q_mean,
                    "team_named_sum_before": named_sum,
                    "team_required_named_reduction_yards": required_named_reduction,
                }
            )
            player_rows.append(meta)

    team_df = pd.DataFrame(team_rows)
    player_df = pd.DataFrame(player_rows)
    if team_df.empty or player_df.empty:
        raise RuntimeError("hierarchical reconciliation audit produced empty outputs")

    # Entitlement quartiles are descriptive only.
    ranked = player_df["entitlement_tgt_share"].rank(method="first")
    player_df["entitlement_quartile"] = pd.qcut(
        ranked,
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    ).astype(str)

    by_position = {}
    for pos, p in player_df.groupby("position_family"):
        by_position[str(pos)] = {
            "players": int(len(p)),
            "median_abs_adjustment_yards": float(p["abs_adjustment_yards"].median()),
            "median_abs_adjustment_pct": float(p["abs_adjustment_pct"].median()),
            "median_propagated_variance": float(p["propagated_variance"].median()),
            "median_tgt_effective_n": float(p["bayes_tgt_share_effective_n"].median()),
            "median_ypt_effective_n": float(p["bayes_ypt_effective_n"].median()),
        }

    by_entitlement = {}
    for q, p in player_df.groupby("entitlement_quartile"):
        by_entitlement[str(q)] = {
            "players": int(len(p)),
            "median_abs_adjustment_yards": float(p["abs_adjustment_yards"].median()),
            "median_abs_adjustment_pct": float(p["abs_adjustment_pct"].median()),
            "median_propagated_variance": float(p["propagated_variance"].median()),
            "median_tgt_effective_n": float(p["bayes_tgt_share_effective_n"].median()),
            "median_ypt_effective_n": float(p["bayes_ypt_effective_n"].median()),
        }

    payload = {
        "study": "HIERARCHICAL_RECEIVER_RECONCILIATION_V1",
        "status": "READ_ONLY_DIAGNOSTIC_COMPLETE",
        "season": int(season),
        "week": int(week),
        "iterations": ITERATIONS,
        "sportsbook_inputs_used": False,
        "target_game_outcomes_used": False,
        "production_changed": False,
        "selected_teams": int(len(team_df)),
        "receiver_player_rows": int(len(player_df)),
        "residual_only_feasible_teams": int(team_df["residual_only_feasible"].sum()),
        "residual_only_feasible_rate": float(team_df["residual_only_feasible"].mean()),
        "named_reduction_required_teams": int((~team_df["residual_only_feasible"]).sum()),
        "median_abs_qb_minus_named_gap": float(team_df["qb_minus_named_gap"].abs().median()),
        "median_implied_residual_share": float(team_df["implied_residual_first_share"].median()),
        "implied_residual_share_gt_05_rate": float(
            team_df["implied_residual_first_share"].gt(0.05).mean()
        ),
        "implied_residual_share_gt_10_rate": float(
            team_df["implied_residual_first_share"].gt(0.10).mean()
        ),
        "implied_residual_share_gt_15_rate": float(
            team_df["implied_residual_first_share"].gt(0.15).mean()
        ),
        "implied_residual_share_gt_20_rate": float(
            team_df["implied_residual_first_share"].gt(0.20).mean()
        ),
        "median_required_named_reduction_pct_among_required": (
            float(
                team_df.loc[
                    ~team_df["residual_only_feasible"], "required_named_reduction_pct"
                ].median()
            )
            if (~team_df["residual_only_feasible"]).any()
            else 0.0
        ),
        "max_weighted_identity_gap": float(max_weighted_identity_gap),
        "by_position": by_position,
        "by_entitlement_quartile": by_entitlement,
        "spearman_abs_adjustment_vs_entitlement": _spearman(
            player_df, "abs_adjustment_yards", "entitlement_tgt_share"
        ),
        "spearman_abs_adjustment_vs_tgt_effective_n": _spearman(
            player_df, "abs_adjustment_yards", "bayes_tgt_share_effective_n"
        ),
        "spearman_abs_adjustment_vs_ypt_effective_n": _spearman(
            player_df, "abs_adjustment_yards", "bayes_ypt_effective_n"
        ),
        "spearman_abs_adjustment_vs_propagated_variance": _spearman(
            player_df, "abs_adjustment_yards", "propagated_variance"
        ),
        "selector_payload": selector_payload,
        "universe_audit": {
            "football_player_rows": int(universe_audit["football_player_rows"]),
            "football_teams": int(universe_audit["football_teams"]),
            "canonical_games": int(universe_audit["canonical_games"]),
            "sportsbook_rows_used_to_define_player_universe": int(
                universe_audit["sportsbook_rows_used_to_define_player_universe"]
            ),
        },
        "team_output": str(TEAM_CSV),
        "player_output": str(PLAYER_CSV),
    }

    if float(payload["max_weighted_identity_gap"]) > TOL:
        raise RuntimeError("weighted reconciliation identity gate failed")
    if payload["sportsbook_inputs_used"] or payload["target_game_outcomes_used"]:
        raise RuntimeError("forbidden information reached reconciliation audit")

    team_df.to_csv(TEAM_CSV, index=False)
    player_df.to_csv(PLAYER_CSV, index=False)
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[hierarchical_receiver_reconciliation_v1] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
