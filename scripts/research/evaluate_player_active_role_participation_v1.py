#!/usr/bin/env python3
"""Player Active-Role Participation V1 — frozen mechanism evaluator.

Scores one pre-frozen named-player participation probability against one
strictly-prior position-level baseline.  The pregame universe is reconstructed
through the already-qualified source-audit code.

This script is research-only.  It does not mutate production projections.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.historical_inputs import build_schedule_history
from scripts.research import audit_player_active_role_participation_v1 as source

VERSION = "PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM"
FREEZE_VERSION = "PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE"
SOURCE_HEAD = "28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c"
SOURCE_RUN = 38895205962
SOURCE_ARTIFACT = 12044155086
SEASONS = (2024, 2025)
FAMILIES = ("QB", "RB", "WR", "TE")
PLAYER_WINDOW = 4
PLAYER_MIN = 3
BASELINE_MIN = 100
PRIOR_STRENGTH = 2.0
BOOT_REPS = 5000
BOOT_SEED = 20261016

SUPPORT = {
    "QB": {"rows": 350, "players": 45, "positive": 100, "zero": 100, "eval_players": 30},
    "RB": {"rows": 800, "players": 80, "positive": 500, "zero": 75, "eval_players": 55},
    "WR": {"rows": 1200, "players": 140, "positive": 900, "zero": 50, "eval_players": 90},
    "TE": {"rows": 700, "players": 75, "positive": 550, "zero": 25, "eval_players": 50},
}


def baseline_probability(k_pos: int, n_pos: int) -> float:
    return (float(k_pos) + 0.5) / (float(n_pos) + 1.0)


def candidate_probability(k_player: int, n_player: int, p_baseline: float) -> float:
    return (float(k_player) + PRIOR_STRENGTH * float(p_baseline)) / (
        float(n_player) + PRIOR_STRENGTH
    )


def bernoulli_logloss(y: np.ndarray, p: np.ndarray) -> float:
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise RuntimeError("frozen probabilities must be interior to (0,1)")
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def score_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        raise RuntimeError("cannot score empty frame")
    y = df["y"].to_numpy(dtype=float)
    p0 = df["p_baseline"].to_numpy(dtype=float)
    p1 = df["p_candidate"].to_numpy(dtype=float)
    b0 = float(np.mean((p0 - y) ** 2))
    b1 = float(np.mean((p1 - y) ** 2))
    l0 = bernoulli_logloss(y, p0)
    l1 = bernoulli_logloss(y, p1)
    obs = float(np.mean(y))
    return {
        "rows": int(len(df)),
        "players": int(df["player_id"].nunique()),
        "positive": int(y.sum()),
        "zero": int(len(y) - y.sum()),
        "observed_rate": obs,
        "mean_p_baseline": float(np.mean(p0)),
        "mean_p_candidate": float(np.mean(p1)),
        "baseline_brier": b0,
        "candidate_brier": b1,
        "delta_brier": b0 - b1,
        "baseline_logloss": l0,
        "candidate_logloss": l1,
        "delta_logloss": l0 - l1,
        "baseline_calibration_in_large_abs": abs(float(np.mean(p0)) - obs),
        "candidate_calibration_in_large_abs": abs(float(np.mean(p1)) - obs),
    }


def per_player_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("player_id", sort=True):
        if len(g) < 5:
            continue
        s = score_frame(g)
        rows.append({
            "player_id": str(pid),
            "player_name_key": str(g["player_name_key"].iloc[0]),
            "season": int(g["season"].iloc[0]) if g["season"].nunique() == 1 else 0,
            "family": str(g["family"].iloc[0]),
            "rows": int(len(g)),
            "teams": "|".join(sorted(set(g["team"].astype(str)))),
            "observed_rate": s["observed_rate"],
            "baseline_brier": s["baseline_brier"],
            "candidate_brier": s["candidate_brier"],
            "player_delta_brier": s["delta_brier"],
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_delta(df: pd.DataFrame, cluster_col: str) -> dict:
    keys = np.array(sorted(df[cluster_col].astype(str).unique()), dtype=object)
    if len(keys) < 2:
        raise RuntimeError(f"insufficient bootstrap clusters for {cluster_col}")
    groups = {k: df.loc[df[cluster_col].astype(str).eq(k)] for k in keys}
    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        total_base = 0.0
        total_cand = 0.0
        total_n = 0
        for key in sampled:
            g = groups[str(key)]
            y = g["y"].to_numpy(dtype=float)
            p0 = g["p_baseline"].to_numpy(dtype=float)
            p1 = g["p_candidate"].to_numpy(dtype=float)
            total_base += float(np.sum((p0 - y) ** 2))
            total_cand += float(np.sum((p1 - y) ** 2))
            total_n += len(g)
        vals[i] = (total_base - total_cand) / max(total_n, 1)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {
        "clusters": int(len(keys)),
        "replicates": BOOT_REPS,
        "seed": BOOT_SEED,
        "mean": float(np.mean(vals)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def equal_player_bootstrap(player_df: pd.DataFrame) -> dict:
    vals = player_df["player_delta_brier"].to_numpy(dtype=float)
    if len(vals) < 2:
        raise RuntimeError("insufficient equal-player bootstrap support")
    rng = np.random.default_rng(BOOT_SEED)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        out[i] = float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {
        "players": int(len(vals)),
        "replicates": BOOT_REPS,
        "seed": BOOT_SEED,
        "mean": float(np.mean(out)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def attach_game_keys(universe: pd.DataFrame) -> pd.DataFrame:
    sched = build_schedule_history(SEASONS)
    keep = sched[["season", "week", "team", "opponent", "game_id"]].copy()
    x = universe.merge(
        keep,
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if x["opponent"].isna().any():
        miss = x.loc[x["opponent"].isna(), ["season", "week", "team"]].drop_duplicates()
        raise RuntimeError(f"schedule bridge missing {len(miss)} team-weeks")
    game = x["game_id"].astype("string").fillna("").str.strip()
    fallback = [
        f"{int(s)}_{int(w):02d}_{'_'.join(sorted([str(t), str(o)]))}"
        for s, w, t, o in zip(x["season"], x["week"], x["team"], x["opponent"])
    ]
    x["game_key"] = np.where(game.ne(""), game.astype(str), fallback)
    return x


def build_scored_rows(inactives: Path) -> tuple[pd.DataFrame, list[dict]]:
    teamweeks, inactive_rows = source.parse_inactives(inactives)
    source_meta: list[dict] = []
    roster = source.load_rosters(source_meta)
    universe = source.build_universe(roster, inactive_rows, teamweeks)
    snaps = source.load_snaps(source_meta)
    universe, _ = source.attach_snap_source(universe, snaps)
    universe = source.attach_prior_universe_support(universe)
    universe = attach_game_keys(universe)

    u = universe.loc[
        universe["season"].isin(SEASONS)
        & universe["family"].isin(FAMILIES)
        & universe["stable_roster_identity"].eq(1)
        & universe["teamweek_snap_source_valid"].eq(1)
        & universe["player_id"].astype(str).str.strip().ne("")
    ].copy()
    u["y"] = u["positive_offensive_participation"].astype(int)
    u = u.sort_values(["season", "week", "family", "team", "player_id"]).reset_index(drop=True)

    rows = []
    for season in SEASONS:
        us = u.loc[u["season"].eq(season)].copy()
        for fam in FAMILIES:
            uf = us.loc[us["family"].eq(fam)].copy()
            for r in uf.itertuples(index=False):
                prior_pos = uf.loc[uf["week"].lt(int(r.week))]
                n_pos = int(len(prior_pos))
                if n_pos < BASELINE_MIN:
                    continue
                k_pos = int(prior_pos["y"].sum())
                p0 = baseline_probability(k_pos, n_pos)

                prior_player = uf.loc[
                    uf["week"].lt(int(r.week))
                    & uf["player_id"].astype(str).eq(str(r.player_id))
                    & uf["team"].astype(str).eq(str(r.team))
                ].sort_values(["week"]).tail(PLAYER_WINDOW)
                n_player = int(len(prior_player))
                if n_player < PLAYER_MIN:
                    continue
                k_player = int(prior_player["y"].sum())
                p1 = candidate_probability(k_player, n_player, p0)

                if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
                    raise RuntimeError("probability mapping escaped open interval")

                rows.append({
                    "season": int(r.season),
                    "week": int(r.week),
                    "team": str(r.team),
                    "opponent": str(r.opponent),
                    "game_key": str(r.game_key),
                    "family": str(r.family),
                    "player_id": str(r.player_id),
                    "player_name_key": str(r.player_name_key),
                    "y": int(r.y),
                    "n_pos_prior": n_pos,
                    "k_pos_prior": k_pos,
                    "p_baseline": p0,
                    "n_player_prior": n_player,
                    "k_player_prior": k_player,
                    "p_candidate": p1,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("frozen mechanism produced zero scored rows")
    if out.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate scored player-game rows")
    return out, source_meta


def support_gates(fam: str, metrics: dict, eval_players: int) -> dict:
    req = SUPPORT[fam]
    return {
        "rows": metrics["rows"] >= req["rows"],
        "players": metrics["players"] >= req["players"],
        "positive": metrics["positive"] >= req["positive"],
        "zero": metrics["zero"] >= req["zero"],
        "evaluable_players": eval_players >= req["eval_players"],
    }


def score_slice(df: pd.DataFrame, pooled: bool = False) -> tuple[dict, pd.DataFrame]:
    m = score_frame(df)
    p = per_player_frame(df)
    m["evaluable_players_ge5"] = int(len(p))
    if len(p):
        vals = p["player_delta_brier"].to_numpy(dtype=float)
        m["equal_player_mean_delta_brier"] = float(np.mean(vals))
        m["median_player_delta_brier"] = float(np.median(vals))
        m["share_players_positive_delta_brier"] = float(np.mean(vals > 0))
    else:
        m["equal_player_mean_delta_brier"] = math.nan
        m["median_player_delta_brier"] = math.nan
        m["share_players_positive_delta_brier"] = math.nan

    q = df.copy()
    if pooled:
        q["player_cluster_key"] = q["season"].astype(str) + "|" + q["player_id"].astype(str)
        q["game_cluster_key"] = q["season"].astype(str) + "|" + q["game_key"].astype(str)
    else:
        q["player_cluster_key"] = q["player_id"].astype(str)
        q["game_cluster_key"] = q["game_key"].astype(str)

    m["player_cluster_bootstrap"] = cluster_bootstrap_delta(q, "player_cluster_key")
    m["game_cluster_bootstrap"] = cluster_bootstrap_delta(q, "game_cluster_key")
    m["equal_player_bootstrap"] = equal_player_bootstrap(p) if len(p) >= 2 else None
    return m, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows, source_meta = build_scored_rows(args.inactives)
        all_player = []
        season_metrics: dict[str, dict] = {}
        pooled_metrics: dict[str, dict] = {}
        position_dispositions: dict[str, str] = {}
        gate_details: dict[str, dict] = {}

        for fam in FAMILIES:
            season_metrics[fam] = {}
            support_ok = True
            for season in SEASONS:
                q = rows.loc[rows["family"].eq(fam) & rows["season"].eq(season)].copy()
                m, p = score_slice(q, pooled=False)
                p["scope"] = str(season)
                all_player.append(p)
                sg = support_gates(fam, m, m["evaluable_players_ge5"])
                m["support_gates"] = sg
                season_metrics[fam][str(season)] = m
                support_ok = support_ok and all(sg.values())

            pooled = rows.loc[rows["family"].eq(fam)].copy()
            pm, pp = score_slice(pooled, pooled=True)
            pp["scope"] = "pooled"
            all_player.append(pp)
            pooled_metrics[fam] = pm

            m24 = season_metrics[fam]["2024"]
            m25 = season_metrics[fam]["2025"]
            gates = {
                "support_2024": all(m24["support_gates"].values()),
                "support_2025": all(m25["support_gates"].values()),
                "development_2024_delta_brier_gt_0": m24["delta_brier"] > 0,
                "development_2024_delta_logloss_ge_0": m24["delta_logloss"] >= 0,
                "confirm_2025_delta_brier_gt_0": m25["delta_brier"] > 0,
                "confirm_2025_delta_logloss_ge_0": m25["delta_logloss"] >= 0,
                "confirm_2025_player_cluster_ci_low_gt_0": m25["player_cluster_bootstrap"]["ci95_low"] > 0,
                "confirm_2025_game_cluster_ci_low_gt_0": m25["game_cluster_bootstrap"]["ci95_low"] > 0,
                "confirm_2025_equal_player_mean_gt_0": m25["equal_player_mean_delta_brier"] > 0,
                "confirm_2025_equal_player_ci_low_gt_0": (
                    m25["equal_player_bootstrap"] is not None
                    and m25["equal_player_bootstrap"]["ci95_low"] > 0
                ),
                "pooled_delta_brier_gt_0": pm["delta_brier"] > 0,
                "pooled_player_cluster_ci_low_gt_0": pm["player_cluster_bootstrap"]["ci95_low"] > 0,
                "pooled_game_cluster_ci_low_gt_0": pm["game_cluster_bootstrap"]["ci95_low"] > 0,
            }
            gate_details[fam] = gates

            if not support_ok:
                position_dispositions[fam] = f"{fam}_ACTIVE_ROLE_PARTICIPATION_MECHANISM_SOURCE_BLOCKED"
            elif all(gates.values()):
                position_dispositions[fam] = f"{fam}_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED"
            else:
                position_dispositions[fam] = f"{fam}_ACTIVE_ROLE_PARTICIPATION_MECHANISM_CLOSED"

        n_qualified = sum(v.endswith("_MECHANISM_QUALIFIED") for v in position_dispositions.values())
        n_blocked = sum(v.endswith("_MECHANISM_SOURCE_BLOCKED") for v in position_dispositions.values())
        if n_qualified == len(FAMILIES):
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL"
        elif n_qualified:
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_PARTIAL"
        elif n_blocked == len(FAMILIES):
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_SOURCE_BLOCKED_ALL"
        else:
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_CLOSED"

        player_df = pd.concat(all_player, ignore_index=True) if all_player else pd.DataFrame()
        rows.to_csv(args.out_dir / "player_active_role_mechanism_scored_rows.csv", index=False)
        player_df.to_csv(args.out_dir / "player_active_role_mechanism_player_diagnostics.csv", index=False)
        pd.DataFrame(source_meta).to_csv(
            args.out_dir / "player_active_role_mechanism_source_snapshots.csv", index=False
        )

        result = {
            "version": VERSION,
            "freeze_version": FREEZE_VERSION,
            "source_head": SOURCE_HEAD,
            "source_run": SOURCE_RUN,
            "source_artifact": SOURCE_ARTIFACT,
            "disposition": overall,
            "position_dispositions": position_dispositions,
            "season_metrics": season_metrics,
            "pooled_metrics": pooled_metrics,
            "qualification_gates": gate_details,
            "frozen_parameters": {
                "seasons": list(SEASONS),
                "player_window": PLAYER_WINDOW,
                "player_min": PLAYER_MIN,
                "baseline_min": BASELINE_MIN,
                "position_baseline_jeffreys_alpha": 0.5,
                "position_baseline_jeffreys_beta": 0.5,
                "candidate_position_prior_strength": PRIOR_STRENGTH,
                "bootstrap_replicates": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
            },
            "target_game_used_for_universe_membership": 0,
            "target_game_snap_used_as_predictor": 0,
            "stage2_opportunity_inputs_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_version": FREEZE_VERSION,
            "source_head": SOURCE_HEAD,
            "source_run": SOURCE_RUN,
            "source_artifact": SOURCE_ARTIFACT,
            "disposition": "PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_INTEGRITY_FAILURE",
            "error": str(exc),
            "target_game_used_for_universe_membership": 0,
            "target_game_snap_used_as_predictor": 0,
            "stage2_opportunity_inputs_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }
        (args.out_dir / "player_active_role_participation_mechanism_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "player_active_role_participation_mechanism_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
