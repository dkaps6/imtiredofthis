#!/usr/bin/env python3
"""Shadow Specialist RNG Isolation V1.

Research-only. No Week-3 outcomes. No OddsAPI. No production mutation.

The candidate replaces order-coupled global RNG consumption with deterministic
semantic substreams and a hierarchical target multinomial at the already-
conserved TE-R5P / WR-R15 specialist rooms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.audit_specialist_mc_downstream_materiality_v1 import (
    ITERATIONS,
    TOL,
    _build_entitlement_state,
    _read_csv,
    _stage_metrics,
)
from scripts.simulation_c2_qb_candidate import StateSimulationResult, simulate_with_states
from scripts.simulation_v2 import (
    _clip_prob,
    _num,
    _player_key,
    _team_inputs,
    _top_n_shares,
)

PRIMARY_MARKETS = {
    "pass_yards",
    "rush_att",
    "rush_yards",
    "receptions",
    "rec_yards",
    "rush_rec_yards",
}
DIST_SEEDS = [1042, 2042, 3042, 4042, 5042, 6042]


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"NFLRNGV1").digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng(base_seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(base_seed, *parts))


def _allocate_counts(
    rng: np.random.Generator,
    totals: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    n_iter = len(totals)
    n_players = len(shares)
    if n_players == 0:
        return np.empty((n_iter, 0), dtype=int)
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    total_share = float(clean.sum())
    if total_share > 0.95:
        clean *= 0.95 / total_share
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs = probs / probs.sum()
    out = np.empty((n_iter, n_players), dtype=int)
    for i, total in enumerate(np.asarray(totals, dtype=int)):
        out[i] = rng.multinomial(max(0, int(total)), probs)[:n_players]
    return out


def _allocate_room(
    rng: np.random.Generator,
    totals: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    n_iter = len(totals)
    n_players = len(shares)
    if n_players == 0:
        return np.empty((n_iter, 0), dtype=int)
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, None)
    mass = float(clean.sum())
    if not np.isfinite(mass) or mass <= 0:
        if np.asarray(totals, dtype=int).max(initial=0) > 0:
            raise RuntimeError("positive room total with zero room probability mass")
        return np.zeros((n_iter, n_players), dtype=int)
    probs = clean / mass
    out = np.empty((n_iter, n_players), dtype=int)
    for i, total in enumerate(np.asarray(totals, dtype=int)):
        out[i] = rng.multinomial(max(0, int(total)), probs)
    return out


def _state_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row["event_id"]), str(row["team"]), str(row["player_clean_key"]))


def _build_group_plan(state: pd.DataFrame) -> dict:
    s = state.copy()
    te_changed = {
        (str(r.event_id), str(r.team), str(r.player_clean_key))
        for r in s.loc[s["te_delta"].abs().gt(TOL)].itertuples(index=False)
    }
    wr_changed = {
        (str(r.event_id), str(r.team), str(r.player_clean_key))
        for r in s.loc[s["wr_delta"].abs().gt(TOL)].itertuples(index=False)
    }
    overlap = te_changed & wr_changed
    if overlap:
        raise RuntimeError(f"specialist mutable-room overlap is unsupported: {list(sorted(overlap))[:20]}")

    authority = {}
    for r in s.itertuples(index=False):
        authority[(str(r.event_id), str(r.team), str(r.player_clean_key))] = float(r.m38_entitlement)

    # Prove the specialist contracts required by hierarchical factorization.
    for cols, label in [
        (("m38_entitlement", "te_entitlement"), "TE"),
        (("te_entitlement", "final_entitlement"), "WR"),
    ]:
        left, right = cols
        changed = te_changed if label == "TE" else wr_changed
        if not changed:
            raise RuntimeError(f"{label} mutable room is empty")
        mask = [
            (str(e), str(t), str(p)) in changed
            for e, t, p in zip(s["event_id"], s["team"], s["player_clean_key"])
        ]
        g = s.loc[mask].copy()
        by_team = g.groupby(["event_id", "team"], dropna=False)[[left, right]].sum()
        max_gap = float((by_team[left] - by_team[right]).abs().max()) if len(by_team) else 0.0
        if max_gap > TOL:
            raise RuntimeError(f"{label} mutable room mass not conserved max_gap={max_gap}")

    return {
        "te_changed": te_changed,
        "wr_changed": wr_changed,
        "authority": authority,
    }


def _group_label(key: tuple[str, str, str], plan: dict) -> str:
    if key in plan["te_changed"]:
        return "TE_CHANGED_ROOM"
    if key in plan["wr_changed"]:
        return "WR_CHANGED_ROOM"
    return f"PLAYER::{key[2]}"


def _hierarchical_targets(
    *,
    game: str,
    team: str,
    team_df: pd.DataFrame,
    pass_att: np.ndarray,
    stage_shares: np.ndarray,
    plan: dict,
    base_seed: int,
) -> tuple[np.ndarray, dict]:
    keys = [
        (str(game), str(team), str(_player_key(row)))
        for _, row in team_df.iterrows()
    ]
    labels = [_group_label(k, plan) for k in keys]

    # Stable top-level authority comes from M38 individual shares with mutable
    # specialist rows collapsed into their already-conserved rooms.
    top_mass: dict[str, float] = {}
    for key, label in zip(keys, labels):
        if key not in plan["authority"]:
            raise RuntimeError(f"missing top-level authority key={key}")
        top_mass[label] = top_mass.get(label, 0.0) + float(plan["authority"][key])

    ordered = sorted(top_mass)
    probs = np.array([top_mass[x] for x in ordered], dtype=float)
    if not np.isfinite(probs).all() or (probs < -TOL).any():
        raise RuntimeError("invalid top-level target room probabilities")
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total > 0.950000000001:
        raise RuntimeError(f"isolated target authority exceeds 0.95 game={game} team={team} sum={total}")
    residual = max(0.0, 1.0 - total)
    top_probs = np.append(probs, residual)
    top_probs = top_probs / top_probs.sum()

    top_rng = _rng(base_seed, "TARGET_TOP", game, team)
    top_counts = np.empty((len(pass_att), len(ordered)), dtype=int)
    for i, n in enumerate(np.asarray(pass_att, dtype=int)):
        top_counts[i] = top_rng.multinomial(max(0, int(n)), top_probs)[:len(ordered)]

    targets = np.zeros((len(pass_att), len(keys)), dtype=int)
    for gi, label in enumerate(ordered):
        idx = [j for j, x in enumerate(labels) if x == label]
        room_totals = top_counts[:, gi]
        if len(idx) == 1:
            targets[:, idx[0]] = room_totals
            continue
        room_shares = np.asarray([stage_shares[j] for j in idx], dtype=float)
        room_rng = _rng(base_seed, "TARGET_ROOM", game, team, label)
        alloc = _allocate_room(room_rng, room_totals, room_shares)
        targets[:, idx] = alloc

    if (targets < 0).any():
        raise RuntimeError("negative target allocation")
    if np.any(targets.sum(axis=1) > np.asarray(pass_att, dtype=int)):
        raise RuntimeError("target allocation exceeds pass attempts")

    return targets, {
        "top_probability_sum": total,
        "top_groups": len(ordered),
        "te_room_members": int(sum(x == "TE_CHANGED_ROOM" for x in labels)),
        "wr_room_members": int(sum(x == "WR_CHANGED_ROOM" for x in labels)),
    }


def simulate_isolated(
    metrics: pd.DataFrame,
    *,
    plan: dict,
    iterations: int = ITERATIONS,
    seed: int = 42,
) -> tuple[StateSimulationResult, dict]:
    values = {}
    states = {}
    counters = {
        "teams": 0,
        "target_total_violations": 0,
        "rush_total_violations": 0,
        "nonfinite_values": 0,
        "max_top_probability_sum": 0.0,
    }
    if metrics.empty:
        return StateSimulationResult(values, int(iterations), states), counters

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

    for game, game_df in players.groupby(game_key, dropna=False, sort=True):
        gs = str(game)
        game_pace_shock = _rng(seed, "GAME_PACE", gs).normal(0.0, 2.0, int(iterations))
        for team, team_df in game_df.groupby("team", dropna=False, sort=True):
            if pd.isna(team) or not str(team).strip():
                continue
            ts = str(team)
            counters["teams"] += 1
            team_df = team_df.sort_values("player_clean_key", kind="mergesort").reset_index(drop=True)
            plays_mean, pass_rate_mean = _team_inputs(team_df)

            volume_rng = _rng(seed, "TEAM_VOLUME", gs, ts)
            plays = np.rint(
                np.clip(volume_rng.normal(plays_mean, 3.5, int(iterations)) + game_pace_shock, 45, 85)
            ).astype(int)
            pass_rate = np.clip(
                volume_rng.normal(pass_rate_mean, 0.035, int(iterations)), 0.25, 0.82
            )
            pass_att = volume_rng.binomial(plays, pass_rate)
            rush_att = plays - pass_att

            pass_eff = np.clip(
                _rng(seed, "PASS_EFF", gs, ts).normal(1.0, 0.09, int(iterations)),
                0.65,
                1.35,
            )
            rush_eff = np.clip(
                _rng(seed, "RUSH_EFF", gs, ts).normal(1.0, 0.10, int(iterations)),
                0.60,
                1.40,
            )

            states[(gs, ts, "plays")] = plays.copy()
            states[(gs, ts, "pass_rate")] = pass_rate.copy()
            states[(gs, ts, "pass_att")] = pass_att.copy()
            states[(gs, ts, "rush_att")] = rush_att.copy()
            states[(gs, ts, "pass_eff_shock")] = pass_eff.copy()
            states[(gs, ts, "rush_eff_shock")] = rush_eff.copy()

            tshares = pd.to_numeric(team_df["entitlement_tgt_share"], errors="raise").to_numpy(float)
            targets, tmeta = _hierarchical_targets(
                game=gs,
                team=ts,
                team_df=team_df,
                pass_att=pass_att,
                stage_shares=tshares,
                plan=plan,
                base_seed=seed,
            )
            counters["max_top_probability_sum"] = max(
                counters["max_top_probability_sum"], float(tmeta["top_probability_sum"])
            )

            raw_rush = np.array(
                [
                    _num(r, "rules_rush_share", "bayes_rush_share", "rush_share", default=0.0)
                    for _, r in team_df.iterrows()
                ]
            )
            rush_shares = _top_n_shares(raw_rush, 5)
            carries = _allocate_counts(
                _rng(seed, "RUSH_ALLOC", gs, ts),
                rush_att,
                rush_shares,
            )
            if np.any(carries.sum(axis=1) > rush_att):
                counters["rush_total_violations"] += 1

            for j, (_, row) in enumerate(team_df.iterrows()):
                pkey = _player_key(row)
                if not pkey:
                    continue
                role = str(row.get("model_role", row.get("role", "")) or "").upper()
                pos = str(row.get("position", "") or "").upper()
                catch = _clip_prob(
                    _num(
                        row,
                        "rules_catch_rate",
                        "bayes_receptions_per_target",
                        "receptions_per_target",
                        "catch_rate",
                        default=0.64,
                    ),
                    0.64,
                )
                recs = _rng(seed, "CATCH", gs, ts, pkey).binomial(targets[:, j], catch)
                vol = float(np.clip(_num(row, "rules_volatility_mult", default=1.0), 0.75, 1.50))

                ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
                ypt = 7.5 if not np.isfinite(ypt) or ypt <= 0 else ypt
                rec_mu = targets[:, j] * ypt * pass_eff
                rec_sd = np.maximum(6.0, np.sqrt(np.maximum(targets[:, j], 1)) * ypt * 0.55) * vol
                rec_yards = np.clip(
                    _rng(seed, "REC_YARDS", gs, ts, pkey).normal(rec_mu, rec_sd),
                    0.0,
                    None,
                )

                ypc = _num(row, "rules_ypc", "bayes_ypc", "ypc")
                ypc = 4.2 if not np.isfinite(ypc) or ypc <= 0 else ypc
                rush_mu = carries[:, j] * ypc * rush_eff
                rush_sd = np.maximum(3.0, np.sqrt(np.maximum(carries[:, j], 1)) * ypc * 0.65) * vol
                rush_yards = np.clip(
                    _rng(seed, "RUSH_YARDS", gs, ts, pkey).normal(rush_mu, rush_sd),
                    0.0,
                    None,
                )

                values[(gs, pkey, "receptions")] = recs.astype(float)
                values[(gs, pkey, "rec_yards")] = rec_yards
                values[(gs, pkey, "rush_att")] = carries[:, j].astype(float)
                values[(gs, pkey, "rush_yards")] = rush_yards
                values[(gs, pkey, "rush_rec_yards")] = rush_yards + rec_yards

                if pos == "QB" or role.startswith("QB"):
                    ypa = _num(row, "rules_ypa", "bayes_ypa", "ypa", "ypa_prior")
                    ypa = 7.0 if not np.isfinite(ypa) or ypa <= 0 else ypa
                    qb_noise = np.clip(
                        _rng(seed, "QB_PASS_NOISE", gs, ts, pkey).normal(
                            1.0, 0.07 * vol, int(iterations)
                        ),
                        0.72,
                        1.28,
                    )
                    values[(gs, pkey, "pass_yards")] = np.clip(
                        pass_att * ypa * pass_eff * qb_noise, 0.0, None
                    )

                td_rate = _num(row, "offensive_td_rate")
                if np.isfinite(td_rate) and td_rate >= 0:
                    rz = _num(row, "rz_share", default=np.nan)
                    rz_mult = float(np.clip(0.75 + rz, 0.75, 1.35)) if np.isfinite(rz) else 1.0
                    wp = _num(row, "team_wp")
                    script_mult = 1.0 + (0.08 * (wp - 0.5) if np.isfinite(wp) else 0.0)
                    lam = max(0.0, td_rate * rz_mult * script_mult)
                    trng = _rng(seed, "TD", gs, ts, pkey)
                    shock = np.clip(trng.normal(1.0, 0.12, int(iterations)), 0.65, 1.35)
                    p_iter = np.clip(1.0 - np.exp(-lam * shock), 0.001, 0.98)
                    values[(gs, pkey, "anytime_td")] = trng.binomial(1, p_iter).astype(float)

    for arr in values.values():
        if not np.isfinite(np.asarray(arr, dtype=float)).all():
            counters["nonfinite_values"] += 1

    return StateSimulationResult(values, int(iterations), states), counters


def _protected_keys(state: pd.DataFrame, column: str) -> set[tuple[str, str]]:
    return {
        (str(r.event_id), str(r.player_clean_key))
        for r in state.loc[state[column]].itertuples(index=False)
    }


def _compare_exact(
    left: StateSimulationResult,
    right: StateSimulationResult,
    protected: set[tuple[str, str]],
    label: str,
) -> dict:
    keys = sorted(set(left.values) & set(right.values))
    checked = drift = 0
    max_mean = 0.0
    max_elem = 0.0
    by_market: dict[str, dict[str, float]] = {}
    for key in keys:
        event, player, market = map(str, key)
        if market not in PRIMARY_MARKETS or (event, player) not in protected:
            continue
        a = np.asarray(left.values[key], dtype=float)
        b = np.asarray(right.values[key], dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"{label} shape mismatch key={key}")
        checked += 1
        mean_gap = abs(float(b.mean() - a.mean()))
        elem_gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
        max_mean = max(max_mean, mean_gap)
        max_elem = max(max_elem, elem_gap)
        if mean_gap > TOL or elem_gap > TOL:
            drift += 1
        m = by_market.setdefault(market, {"keys": 0, "drift_keys": 0, "max_mean_gap": 0.0, "max_element_gap": 0.0})
        m["keys"] += 1
        m["drift_keys"] += int(mean_gap > TOL or elem_gap > TOL)
        m["max_mean_gap"] = max(m["max_mean_gap"], mean_gap)
        m["max_element_gap"] = max(m["max_element_gap"], elem_gap)
    if checked == 0:
        raise RuntimeError(f"{label} checked zero protected keys")
    return {
        "label": label,
        "checked_keys": checked,
        "drift_keys": drift,
        "max_abs_mean_gap": max_mean,
        "max_element_gap": max_elem,
        "by_market": by_market,
    }


def _intentional_changes(
    left: StateSimulationResult,
    right: StateSimulationResult,
    changed: set[tuple[str, str, str]],
    label: str,
) -> dict:
    changed_arrays = 0
    checked = 0
    top = []
    for event, team, player in sorted(changed):
        for market in ("receptions", "rec_yards"):
            key = (str(event), str(player), market)
            if key not in left.values or key not in right.values:
                continue
            checked += 1
            a = np.asarray(left.values[key], dtype=float)
            b = np.asarray(right.values[key], dtype=float)
            gap = float(np.max(np.abs(a - b))) if len(a) else 0.0
            if gap > TOL:
                changed_arrays += 1
            top.append((gap, event, team, player, market))
    top.sort(reverse=True)
    return {
        "label": label,
        "checked_arrays": checked,
        "changed_arrays": changed_arrays,
        "top_changes": [
            {"max_element_gap": g, "event_id": e, "team": t, "player_clean_key": p, "market": m}
            for g, e, t, p, m in top[:20]
        ],
    }


def _distribution_compatibility(
    final_metrics: pd.DataFrame,
    plan: dict,
) -> pd.DataFrame:
    rows = []
    for seed in DIST_SEEDS:
        canonical = simulate_with_states(final_metrics, iterations=ITERATIONS, seed=seed)
        isolated, _ = simulate_isolated(final_metrics, plan=plan, iterations=ITERATIONS, seed=seed)
        for key in sorted(set(canonical.values) & set(isolated.values)):
            market = str(key[2])
            if market not in PRIMARY_MARKETS:
                continue
            a = np.asarray(canonical.values[key], dtype=float)
            b = np.asarray(isolated.values[key], dtype=float)
            rows.append({
                "seed": int(seed),
                "event_id": str(key[0]),
                "player_clean_key": str(key[1]),
                "market": market,
                "abs_mean_diff": abs(float(a.mean() - b.mean())),
                "abs_sd_diff": abs(float(np.std(a, ddof=1) - np.std(b, ddof=1))),
            })
    return pd.DataFrame(rows)


def _distribution_summary(detail: pd.DataFrame) -> pd.DataFrame:
    out = []
    for market, g in detail.groupby("market", sort=True):
        out.append({
            "market": market,
            "rows": int(len(g)),
            "mean_abs_mean_diff": float(g["abs_mean_diff"].mean()),
            "p95_abs_mean_diff": float(g["abs_mean_diff"].quantile(.95)),
            "max_abs_mean_diff": float(g["abs_mean_diff"].max()),
            "mean_abs_sd_diff": float(g["abs_sd_diff"].mean()),
            "p95_abs_sd_diff": float(g["abs_sd_diff"].quantile(.95)),
            "max_abs_sd_diff": float(g["abs_sd_diff"].max()),
        })
    return pd.DataFrame(out)


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
    starters = _read_csv(args.root / "data/qb_c2_production_starter_audit.csv", "QB starter audit")
    if len(starters) != 30:
        raise RuntimeError(f"expected frozen 30-QB Week-3 starter authority, got {len(starters)}")

    plan = _build_group_plan(state)
    metrics = {
        "m38": _stage_metrics(universe, state, "m38_entitlement", starters),
        "te": _stage_metrics(universe, state, "te_entitlement", starters),
        "wr": _stage_metrics(universe, state, "final_entitlement", starters),
    }

    sims = {}
    counters = {}
    for name in ("m38", "te", "wr"):
        sims[name], counters[name] = simulate_isolated(
            metrics[name], plan=plan, iterations=ITERATIONS, seed=42
        )

    te_exact = _compare_exact(
        sims["m38"], sims["te"], _protected_keys(state, "te_protected"), "M38_TO_TE_R5P"
    )
    wr_exact = _compare_exact(
        sims["te"], sims["wr"], _protected_keys(state, "wr_protected"), "TE_R5P_TO_WR_R15"
    )
    te_intent = _intentional_changes(
        sims["m38"], sims["te"], plan["te_changed"], "TE_R5P_INTENTIONAL"
    )
    wr_intent = _intentional_changes(
        sims["te"], sims["wr"], plan["wr_changed"], "WR_R15_INTENTIONAL"
    )

    integrity = {
        "source_run_id": str(args.source_run_id),
        "source_artifact_id": str(args.source_artifact_id),
        "source_artifact_digest": str(args.source_artifact_digest),
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_changed": False,
        "iterations": ITERATIONS,
        "production_seed": 42,
        "te_changed_players": len(plan["te_changed"]),
        "wr_changed_players": len(plan["wr_changed"]),
        "stage_counters": counters,
    }
    conservation_ok = all(
        c["target_total_violations"] == 0
        and c["rush_total_violations"] == 0
        and c["nonfinite_values"] == 0
        for c in counters.values()
    )
    exact_ok = te_exact["drift_keys"] == 0 and wr_exact["drift_keys"] == 0
    intent_ok = te_intent["changed_arrays"] > 0 and wr_intent["changed_arrays"] > 0

    if not intent_ok:
        disposition = "SPECIALIST_RNG_ISOLATION_INVALID_FREEZE"
    elif exact_ok and conservation_ok:
        disposition = "SPECIALIST_RNG_ISOLATION_CORE_PASS"
    else:
        disposition = "SPECIALIST_RNG_ISOLATION_CORE_FAIL"

    dist_detail = _distribution_compatibility(metrics["wr"], plan)
    dist_summary = _distribution_summary(dist_detail)
    dist_detail.to_csv(out_dir / "distribution_compatibility_detail.csv", index=False)
    dist_summary.to_csv(out_dir / "distribution_compatibility_summary.csv", index=False)

    payload = {
        "version": "SPECIALIST_RNG_ISOLATION_V1",
        "disposition": disposition,
        "integrity": integrity,
        "exact_isolation": {
            "m38_to_te_r5p": te_exact,
            "te_r5p_to_wr_r15": wr_exact,
        },
        "intentional_specialist_change": {
            "te_r5p": te_intent,
            "wr_r15": wr_intent,
        },
        "conservation_pass": conservation_ok,
        "exact_isolation_pass": exact_ok,
        "intentional_change_pass": intent_ok,
        "production_repair_authorized": False,
        "next_if_core_pass": "QB_C2_RNG_ISOLATION_EXTENSION_V1",
        "week3_outcomes_used": False,
        "odds_refetch_performed": False,
        "production_changed": False,
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    print(dist_summary.to_string(index=False))
    if disposition != "SPECIALIST_RNG_ISOLATION_CORE_PASS":
        raise RuntimeError(disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
