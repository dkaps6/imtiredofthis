"""Semantic RNG isolation candidate for the canonical joint NFL simulator.

This module changes random-number routing only. It consumes the already-certified
explicit target entitlement and specialist scope metadata, preserving football
means/inputs while preventing an unrelated specialist redistribution from
changing protected players merely because NumPy consumed a different global RNG
path earlier in the simulation.

Production status: candidate only until separately promoted.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from scripts.config import MC
from scripts.simulation_c2_qb_candidate import StateSimulationResult
from scripts.simulation_v2 import (
    SimulationResult,
    _allocate_counts,
    _clip_prob,
    _num,
    _player_key,
    _team_inputs,
    _top_n_shares,
)

TOL = 1e-12
# These literal room labels are part of the frozen deterministic RNG keyspace.
# Preserve the exact research V1 labels; renaming them changes the substream.
TE_ROOM = "TE_CHANGED_ROOM"
WR_ROOM = "WR_CHANGED_ROOM"


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *[str(p) for p in parts]]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"NFLRNGV1").digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng(base_seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(base_seed, *parts))


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


def _prepare(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if metrics is None or metrics.empty:
        return pd.DataFrame(), {
            "te_members": set(),
            "wr_members": set(),
            "baseline": {},
        }

    frame = metrics.copy()
    frame["player_clean_key"] = frame.apply(_player_key, axis=1)
    required = {
        "event_id",
        "team",
        "player_clean_key",
        "baseline_entitlement_tgt_share",
        "entitlement_tgt_share",
        "te_r5p_applied",
        "wr_r15_applied",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"RNG isolation candidate missing required columns: {missing}")

    baseline = pd.to_numeric(frame["baseline_entitlement_tgt_share"], errors="coerce")
    current = pd.to_numeric(frame["entitlement_tgt_share"], errors="coerce")
    if (
        baseline.isna().any()
        or current.isna().any()
        or not np.isfinite(baseline.to_numpy(float)).all()
        or not np.isfinite(current.to_numpy(float)).all()
        or baseline.lt(0).any()
        or current.lt(0).any()
    ):
        raise RuntimeError("RNG isolation candidate received invalid entitlement")

    frame["baseline_entitlement_tgt_share"] = baseline.astype(float)
    frame["entitlement_tgt_share"] = current.astype(float)
    frame["_te_scope"] = _bool_series(frame["te_r5p_applied"])
    frame["_wr_scope"] = _bool_series(frame["wr_r15_applied"])
    if (frame["_te_scope"] & frame["_wr_scope"]).any():
        bad = frame.loc[
            frame["_te_scope"] & frame["_wr_scope"],
            ["event_id", "team", "player_clean_key", "position"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"RNG isolation specialist scopes overlap: {bad}")

    key_cols = ["event_id", "team", "player_clean_key"]
    if frame.duplicated(key_cols).any():
        # The canonical simulator itself reduces to one row per player/team/game.
        # Perform that reduction before deriving semantic room membership.
        frame = (
            frame.sort_values(key_cols, kind="mergesort")
            .drop_duplicates(key_cols, keep="last")
            .reset_index(drop=True)
        )

    te_members = {
        (str(r.event_id), str(r.team), str(r.player_clean_key))
        for r in frame.loc[frame["_te_scope"]].itertuples(index=False)
    }
    wr_members = {
        (str(r.event_id), str(r.team), str(r.player_clean_key))
        for r in frame.loc[frame["_wr_scope"]].itertuples(index=False)
    }
    baseline_map = {
        (str(r.event_id), str(r.team), str(r.player_clean_key)):
            float(r.baseline_entitlement_tgt_share)
        for r in frame.itertuples(index=False)
    }

    # The room factorization is valid only because the promoted specialists
    # conserve their authorized room mass. Prove that on every call/stage.
    for room_col, label in [("_te_scope", TE_ROOM), ("_wr_scope", WR_ROOM)]:
        room = frame.loc[frame[room_col]].copy()
        if room.empty:
            raise RuntimeError(f"RNG isolation candidate found empty {label}")
        before = room.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].sum()
        after = room.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
        gap = (after - before).abs()
        max_gap = float(gap.max()) if len(gap) else 0.0
        if max_gap > TOL:
            raise RuntimeError(f"{label} mass not conserved max_gap={max_gap}")

    outside = ~(frame["_te_scope"] | frame["_wr_scope"])
    outside_gap = (
        frame.loc[outside, "entitlement_tgt_share"]
        - frame.loc[outside, "baseline_entitlement_tgt_share"]
    ).abs()
    max_outside = float(outside_gap.max()) if len(outside_gap) else 0.0
    if max_outside > TOL:
        raise RuntimeError(
            "RNG isolation candidate found entitlement change outside explicit specialist scopes "
            f"max_gap={max_outside}"
        )

    return frame, {
        "te_members": te_members,
        "wr_members": wr_members,
        "baseline": baseline_map,
    }


def _group_label(key: tuple[str, str, str], plan: dict) -> str:
    if key in plan["te_members"]:
        return TE_ROOM
    if key in plan["wr_members"]:
        return WR_ROOM
    return f"PLAYER::{key[2]}"


def _allocate_room(
    rng: np.random.Generator,
    totals: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    n_iter = len(totals)
    n_players = len(shares)
    if n_players == 0:
        return np.empty((n_iter, 0), dtype=int)
    clean = np.nan_to_num(
        np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    clean = np.clip(clean, 0.0, None)
    mass = float(clean.sum())
    max_total = int(np.max(totals)) if len(totals) else 0
    if mass <= 0:
        if max_total > 0:
            raise RuntimeError("positive semantic-room allocation with zero room probability mass")
        return np.zeros((n_iter, n_players), dtype=int)
    probs = clean / mass
    out = np.empty((n_iter, n_players), dtype=int)
    for i, total in enumerate(np.asarray(totals, dtype=int)):
        out[i] = rng.multinomial(max(0, int(total)), probs)
    return out


def _hierarchical_targets(
    *,
    game: str,
    team: str,
    team_df: pd.DataFrame,
    pass_att: np.ndarray,
    plan: dict,
    base_seed: int,
) -> np.ndarray:
    keys = [
        (str(game), str(team), str(_player_key(row)))
        for _, row in team_df.iterrows()
    ]
    labels = [_group_label(k, plan) for k in keys]

    top_mass: dict[str, float] = {}
    for key, label in zip(keys, labels):
        if key not in plan["baseline"]:
            raise RuntimeError(f"RNG isolation missing baseline authority key={key}")
        top_mass[label] = top_mass.get(label, 0.0) + float(plan["baseline"][key])

    ordered = sorted(top_mass)
    probs = np.asarray([top_mass[x] for x in ordered], dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise RuntimeError("RNG isolation invalid top-level target probabilities")
    total = float(probs.sum())
    if total > 0.950000000001:
        raise RuntimeError(
            f"RNG isolation modeled target mass exceeds 0.95 game={game} team={team} sum={total}"
        )

    residual = max(0.0, 1.0 - total)
    top_probs = np.append(probs, residual)
    top_probs = top_probs / top_probs.sum()
    top_rng = _rng(base_seed, "TARGET_TOP", game, team)
    top_counts = np.empty((len(pass_att), len(ordered)), dtype=int)
    for i, n in enumerate(np.asarray(pass_att, dtype=int)):
        top_counts[i] = top_rng.multinomial(max(0, int(n)), top_probs)[: len(ordered)]

    stage_shares = pd.to_numeric(
        team_df["entitlement_tgt_share"], errors="raise"
    ).to_numpy(float)
    targets = np.zeros((len(pass_att), len(keys)), dtype=int)
    for group_idx, label in enumerate(ordered):
        idx = [j for j, value in enumerate(labels) if value == label]
        totals = top_counts[:, group_idx]
        if len(idx) == 1:
            targets[:, idx[0]] = totals
            continue
        room_shares = stage_shares[idx]
        baseline_mass = float(sum(plan["baseline"][keys[j]] for j in idx))
        stage_mass = float(room_shares.sum())
        if abs(stage_mass - baseline_mass) > TOL:
            raise RuntimeError(
                f"RNG isolation room mass drift game={game} team={team} room={label} "
                f"baseline={baseline_mass} stage={stage_mass}"
            )
        targets[:, idx] = _allocate_room(
            _rng(base_seed, "TARGET_ROOM", game, team, label),
            totals,
            room_shares,
        )

    if (targets < 0).any():
        raise RuntimeError("RNG isolation generated negative target counts")
    if np.any(targets.sum(axis=1) > np.asarray(pass_att, dtype=int)):
        raise RuntimeError("RNG isolation modeled targets exceed pass attempts")
    return targets


def _simulate_internal(
    metrics: pd.DataFrame,
    *,
    iterations: int | None = None,
    seed: int | None = None,
    allocation_trace: list[dict] | None = None,
) -> tuple[dict, dict, int]:
    iterations = int(iterations or MC.get("iterations", 25000))
    seed = int(MC.get("seed", 42) if seed is None else seed)
    prepared, plan = _prepare(metrics)
    values: dict = {}
    states: dict = {}
    if prepared.empty:
        return values, states, iterations

    game_key = (
        "event_id"
        if "event_id" in prepared.columns and prepared["event_id"].notna().any()
        else None
    )
    if game_key is None:
        prepared["_game_key"] = prepared.apply(
            lambda r: "|".join(
                sorted([str(r.get("team", "")), str(r.get("opponent", ""))])
            ),
            axis=1,
        )
        game_key = "_game_key"

    players = (
        prepared.sort_values([game_key, "team", "player_clean_key"], kind="mergesort")
        .drop_duplicates([game_key, "team", "player_clean_key"], keep="last")
    )

    for game, game_df in players.groupby(game_key, dropna=False, sort=True):
        gs = str(game)
        game_pace_shock = _rng(seed, "GAME_PACE", gs).normal(
            0.0, 2.0, iterations
        )
        for team, team_df in game_df.groupby("team", dropna=False, sort=True):
            if pd.isna(team) or not str(team).strip():
                continue
            ts = str(team)
            team_df = team_df.sort_values("player_clean_key", kind="mergesort").reset_index(drop=True)

            plays_mean, pass_rate_mean = _team_inputs(team_df)
            volume_rng = _rng(seed, "TEAM_VOLUME", gs, ts)
            plays = np.rint(
                np.clip(
                    volume_rng.normal(plays_mean, 3.5, iterations) + game_pace_shock,
                    45,
                    85,
                )
            ).astype(int)
            pass_rate = np.clip(
                volume_rng.normal(pass_rate_mean, 0.035, iterations),
                0.25,
                0.82,
            )
            pass_att = volume_rng.binomial(plays, pass_rate)
            rush_att = plays - pass_att

            pass_eff = np.clip(
                _rng(seed, "PASS_EFF", gs, ts).normal(1.0, 0.09, iterations),
                0.65,
                1.35,
            )
            rush_eff = np.clip(
                _rng(seed, "RUSH_EFF", gs, ts).normal(1.0, 0.10, iterations),
                0.60,
                1.40,
            )

            states[(gs, ts, "plays")] = plays.copy()
            states[(gs, ts, "pass_rate")] = pass_rate.copy()
            states[(gs, ts, "pass_att")] = pass_att.copy()
            states[(gs, ts, "rush_att")] = rush_att.copy()
            states[(gs, ts, "pass_eff_shock")] = pass_eff.copy()
            states[(gs, ts, "rush_eff_shock")] = rush_eff.copy()

            targets = _hierarchical_targets(
                game=gs,
                team=ts,
                team_df=team_df,
                pass_att=pass_att,
                plan=plan,
                base_seed=seed,
            )

            raw_rush = np.asarray(
                [
                    _num(
                        row,
                        "rules_rush_share",
                        "bayes_rush_share",
                        "rush_share",
                        default=0.0,
                    )
                    for _, row in team_df.iterrows()
                ],
                dtype=float,
            )
            rush_shares = _top_n_shares(raw_rush, 5)
            carries = _allocate_counts(
                _rng(seed, "RUSH_ALLOC", gs, ts),
                rush_att,
                rush_shares,
            )
            if np.any(carries.sum(axis=1) > rush_att):
                raise RuntimeError("RNG isolation modeled carries exceed rush attempts")

            if allocation_trace is not None:
                clean = np.clip(
                    np.nan_to_num(
                        rush_shares.astype(float),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                    0.0,
                    0.95,
                )
                raw_sum = float(clean.sum())
                used = clean.copy()
                if raw_sum > 0.95:
                    used *= 0.95 / raw_sum
                residual = max(0.0, 1.0 - float(used.sum()))
                probs_trace = np.append(used, residual)
                probs_trace = probs_trace / probs_trace.sum()
                team_rush_mean = (
                    float(np.mean(rush_att)) if len(rush_att) else np.nan
                )
                for j, (_, trace_row) in enumerate(team_df.iterrows()):
                    allocation_trace.append(
                        {
                            "event_id": gs,
                            "team": ts,
                            "player_clean_key": _player_key(trace_row),
                            "sim_selected_market": str(trace_row.get("market", "")),
                            "raw_player_rush_share": float(clean[j]),
                            "raw_team_rush_share_sum": raw_sum,
                            "final_player_probability": float(probs_trace[j]),
                            "residual_probability": float(probs_trace[-1]),
                            "team_rush_total_mean": team_rush_mean,
                            "expected_carries_from_final_probability": team_rush_mean
                            * float(probs_trace[j]),
                            "realized_multinomial_mean_carries": float(
                                carries[:, j].mean()
                            ),
                        }
                    )

            for j, (_, row) in enumerate(team_df.iterrows()):
                pkey = _player_key(row)
                if not pkey:
                    continue
                role = str(row.get("model_role", row.get("role", "")) or "").upper()
                position = str(row.get("position", "") or "").upper()
                catch_rate = _clip_prob(
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
                receptions = _rng(seed, "CATCH", gs, ts, pkey).binomial(
                    targets[:, j], catch_rate
                )
                vol_mult = float(
                    np.clip(_num(row, "rules_volatility_mult", default=1.0), 0.75, 1.50)
                )

                ypt = _num(row, "rules_ypt", "bayes_ypt", "ypt")
                ypt = 7.5 if not np.isfinite(ypt) or ypt <= 0 else ypt
                rec_mu = targets[:, j] * ypt * pass_eff
                rec_sd = (
                    np.maximum(
                        6.0,
                        np.sqrt(np.maximum(targets[:, j], 1)) * ypt * 0.55,
                    )
                    * vol_mult
                )
                rec_yards = np.clip(
                    _rng(seed, "REC_YARDS", gs, ts, pkey).normal(rec_mu, rec_sd),
                    0.0,
                    None,
                )

                ypc = _num(row, "rules_ypc", "bayes_ypc", "ypc")
                ypc = 4.2 if not np.isfinite(ypc) or ypc <= 0 else ypc
                rush_mu = carries[:, j] * ypc * rush_eff
                rush_sd = (
                    np.maximum(
                        3.0,
                        np.sqrt(np.maximum(carries[:, j], 1)) * ypc * 0.65,
                    )
                    * vol_mult
                )
                rush_yards = np.clip(
                    _rng(seed, "RUSH_YARDS", gs, ts, pkey).normal(rush_mu, rush_sd),
                    0.0,
                    None,
                )

                values[(gs, pkey, "receptions")] = receptions.astype(float)
                values[(gs, pkey, "rec_yards")] = rec_yards
                values[(gs, pkey, "rush_att")] = carries[:, j].astype(float)
                values[(gs, pkey, "rush_yards")] = rush_yards
                values[(gs, pkey, "rush_rec_yards")] = rush_yards + rec_yards

                if position == "QB" or role.startswith("QB"):
                    ypa = _num(row, "rules_ypa", "bayes_ypa", "ypa", "ypa_prior")
                    ypa = 7.0 if not np.isfinite(ypa) or ypa <= 0 else ypa
                    qb_noise = np.clip(
                        _rng(seed, "QB_PASS_NOISE", gs, ts, pkey).normal(
                            1.0, 0.07 * vol_mult, iterations
                        ),
                        0.72,
                        1.28,
                    )
                    values[(gs, pkey, "pass_yards")] = np.clip(
                        pass_att * ypa * pass_eff * qb_noise,
                        0.0,
                        None,
                    )

                td_rate = _num(row, "offensive_td_rate")
                if np.isfinite(td_rate) and td_rate >= 0:
                    rz = _num(row, "rz_share", default=np.nan)
                    rz_mult = (
                        float(np.clip(0.75 + rz, 0.75, 1.35))
                        if np.isfinite(rz)
                        else 1.0
                    )
                    wp = _num(row, "team_wp")
                    script_mult = 1.0 + (
                        0.08 * (wp - 0.5) if np.isfinite(wp) else 0.0
                    )
                    lam = max(0.0, td_rate * rz_mult * script_mult)
                    td_rng = _rng(seed, "TD", gs, ts, pkey)
                    scoring_shock = np.clip(
                        td_rng.normal(1.0, 0.12, iterations),
                        0.65,
                        1.35,
                    )
                    p_iter = np.clip(
                        1.0 - np.exp(-lam * scoring_shock),
                        0.001,
                        0.98,
                    )
                    values[(gs, pkey, "anytime_td")] = td_rng.binomial(
                        1, p_iter
                    ).astype(float)

    for key, arr in values.items():
        if not np.isfinite(np.asarray(arr, dtype=float)).all():
            raise RuntimeError(f"RNG isolation produced non-finite simulation array key={key}")

    return values, states, iterations


def simulate(
    metrics: pd.DataFrame,
    *,
    iterations: int | None = None,
    seed: int | None = None,
    allocation_trace: list[dict] | None = None,
) -> SimulationResult:
    values, _, n = _simulate_internal(
        metrics,
        iterations=iterations,
        seed=seed,
        allocation_trace=allocation_trace,
    )
    return SimulationResult(values, n)


def simulate_with_states(
    metrics: pd.DataFrame,
    *,
    iterations: int | None = None,
    seed: int | None = None,
) -> StateSimulationResult:
    values, states, n = _simulate_internal(
        metrics,
        iterations=iterations,
        seed=seed,
        allocation_trace=None,
    )
    return StateSimulationResult(values, n, states)
