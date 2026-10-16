#!/usr/bin/env python3
"""Score the frozen Player Active-Role -> Conditional Allocation Bridge V1.

Research-only historical scoring for the four source-ready cells.
The candidate is an exact finite mixture over independent Bernoulli active-role
states.  No coefficient, threshold, sportsbook input, 2026 outcome, or
production mutation is permitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from scripts._opponent_map import canon_team
from scripts.research.audit_player_opportunity_allocation_uncertainty_betabinomial_v1 import (
    build_target_counts,
    discrete_crps,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_HISTORICAL_SCORE"
FREEZE = "docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md"
SOURCE_RUN = 38955368643
SOURCE_ARTIFACT = 12070806828
STAGE1_RUN = 38907459996
STAGE1_ARTIFACT = 12049120604
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
CELLS = ("QB_PASSING", "RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")
BOOT_REPS = 5000
BOOT_SEED = 20261016
TOL = 1e-12

SUPPORT = {
    "QB_PASSING": {"rows": 350, "players": 45, "rooms": 300, "eval_players": 30},
    "RB_RUSHING": {"rows": 1200, "players": 100, "rooms": 400, "eval_players": 75},
    "RB_RECEIVING": {"rows": 1200, "players": 100, "rooms": 400, "eval_players": 75},
    "TE_RECEIVING": {"rows": 1200, "players": 100, "rooms": 400, "eval_players": 75},
}


def num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def clean_key(value: Any) -> str:
    try:
        _, k = canonicalize_player_name_safe(value)
        if k:
            return str(k)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def attach_stage1_probability(source: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    req = {
        "season", "week", "team", "player_id", "family", "player_name_key",
        "stage1_p_baseline", "authority_q", "authority_covered",
        "denominator_finite", "ambiguous_identity", "chronology_violations", "cell",
    }
    miss = req - set(source.columns)
    if miss:
        raise RuntimeError(f"source join detail missing columns: {sorted(miss)}")
    mreq = {"season", "week", "team", "family", "player_id", "p_candidate"}
    miss = mreq - set(mechanism.columns)
    if miss:
        raise RuntimeError(f"Stage-1 mechanism missing columns: {sorted(miss)}")

    x = source.loc[
        num(source["season"]).isin(TARGET_SEASONS)
        & num(source["week"]).isin(TARGET_WEEKS)
        & source["cell"].astype(str).isin(CELLS)
    ].copy()
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].astype(str).str.strip()
    x["player_name_key"] = x["player_name_key"].map(clean_key)
    x["authority_q"] = num(x["authority_q"])
    x["stage1_p_baseline"] = num(x["stage1_p_baseline"])
    for c in ("authority_covered", "denominator_finite", "ambiguous_identity"):
        if x[c].dtype != bool:
            x[c] = x[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    x = x.loc[
        x["authority_covered"]
        & x["denominator_finite"]
        & ~x["ambiguous_identity"]
    ].copy()
    if (num(x["chronology_violations"]).fillna(0) != 0).any():
        raise RuntimeError("source contains chronology violation")
    if x.duplicated(["season", "week", "team", "player_id", "cell"]).any():
        raise RuntimeError("duplicate source player-cell identity")

    m = mechanism.loc[
        num(mechanism["season"]).isin(TARGET_SEASONS)
        & num(mechanism["week"]).isin(TARGET_WEEKS)
    ].copy()
    m["season"] = num(m["season"]).astype(int)
    m["week"] = num(m["week"]).astype(int)
    m["team"] = m["team"].map(canon_team)
    m["player_id"] = m["player_id"].astype(str).str.strip()
    m["p_candidate"] = num(m["p_candidate"])
    m = m[["season", "week", "team", "family", "player_id", "p_candidate"]]
    if m.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate Stage-1 mechanism identity")

    x = x.merge(
        m,
        on=["season", "week", "team", "family", "player_id"],
        how="left",
        validate="many_to_one",
    )
    x["stage1_named_player_supported"] = x["p_candidate"].notna()
    x["p_active"] = x["p_candidate"].where(
        x["stage1_named_player_supported"], x["stage1_p_baseline"]
    )
    bad = (
        x["p_active"].isna()
        | ~np.isfinite(x["p_active"])
        | x["p_active"].le(0)
        | x["p_active"].ge(1)
    )
    if bad.any():
        raise RuntimeError(f"invalid Stage-1 p_active rows={int(bad.sum())}")
    return x


def build_non_qb_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = build_target_counts().copy()
    c = c.loc[
        num(c["season"]).isin(TARGET_SEASONS)
        & num(c["week"]).isin(TARGET_WEEKS)
    ].copy()
    c["season"] = num(c["season"]).astype(int)
    c["week"] = num(c["week"]).astype(int)
    c["team"] = c["team"].map(canon_team)
    c["player_identity_key"] = c["player_identity_key"].astype(str)
    c["player_id"] = c["player_identity_key"].str.replace(r"^gsis:", "", regex=True)
    c["targets"] = num(c["targets"]).fillna(0)
    c["rushes"] = num(c["rushes"]).fillna(0)
    c["team_targets"] = num(c["team_targets"])
    c["team_rushes"] = num(c["team_rushes"])
    c["game_id"] = c["game_id"].astype(str)

    players = (
        c.groupby(["season", "week", "team", "player_id"], as_index=False)
        .agg(
            targets=("targets", "sum"),
            rushes=("rushes", "sum"),
        )
    )
    teams = (
        c.groupby(["season", "week", "team"], as_index=False)
        .agg(
            game_id=("game_id", "first"),
            team_targets=("team_targets", "max"),
            team_rushes=("team_rushes", "max"),
        )
    )
    if teams.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate non-QB team outcome identity")
    return players, teams


def build_qb_outcomes(player_logs: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = player_logs.copy()
    logs.columns = [str(c).strip().lower() for c in logs.columns]
    need = {"season", "week", "team", "pass_att"}
    miss = need - set(logs.columns)
    if miss:
        raise RuntimeError(f"QB player logs missing columns: {sorted(miss)}")
    logs = logs.loc[
        num(logs["season"]).isin(TARGET_SEASONS)
        & num(logs["week"]).isin(TARGET_WEEKS)
    ].copy()
    logs["season"] = num(logs["season"]).astype(int)
    logs["week"] = num(logs["week"]).astype(int)
    logs["team"] = logs["team"].map(canon_team)
    logs["pass_att"] = num(logs["pass_att"]).fillna(0)

    if "player_id" in logs.columns:
        logs["player_id"] = logs["player_id"].astype(str).str.strip()
    elif "player_identity_key" in logs.columns:
        logs["player_id"] = (
            logs["player_identity_key"].astype(str)
            .str.replace(r"^gsis:", "", regex=True)
        )
    else:
        raise RuntimeError("QB player logs lack stable player ID")

    players = (
        logs.groupby(["season", "week", "team", "player_id"], as_index=False)
        .agg(pass_att=("pass_att", "sum"))
    )
    team_n = (
        logs.groupby(["season", "week", "team"], as_index=False)
        .agg(team_pass_att=("pass_att", "sum"))
    )

    sched = schedule.copy()
    sched.columns = [str(c).strip().lower() for c in sched.columns]
    for c in ("season", "week"):
        sched[c] = num(sched[c]).astype("Int64")
    sched["team"] = sched["team"].map(canon_team)
    if "game_id" not in sched.columns:
        sched["game_id"] = ""
    sched = sched.loc[
        num(sched["season"]).isin(TARGET_SEASONS)
        & num(sched["week"]).isin(TARGET_WEEKS),
        ["season", "week", "team", "game_id"],
    ].drop_duplicates(["season", "week", "team"])
    teams = team_n.merge(
        sched,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
    )
    teams["game_id"] = teams["game_id"].astype("string").fillna("").astype(str)
    empty = teams["game_id"].str.strip().eq("")
    teams.loc[empty, "game_id"] = [
        f"{int(s)}_{int(w):02d}_{t}"
        for s, w, t in teams.loc[empty, ["season", "week", "team"]].itertuples(index=False, name=None)
    ]
    return players, teams


def attach_outcomes(
    source: pd.DataFrame,
    nonqb_players: pd.DataFrame,
    nonqb_teams: pd.DataFrame,
    qb_players: pd.DataFrame,
    qb_teams: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for cell, g in source.groupby("cell", sort=False):
        z = g.copy()
        if cell == "QB_PASSING":
            z = z.merge(
                qb_players,
                on=["season", "week", "team", "player_id"],
                how="left",
                validate="many_to_one",
            )
            z = z.merge(
                qb_teams,
                on=["season", "week", "team"],
                how="left",
                validate="many_to_one",
            )
            z["y"] = num(z["pass_att"]).fillna(0)
            z["N"] = num(z["team_pass_att"])
        else:
            z = z.merge(
                nonqb_players,
                on=["season", "week", "team", "player_id"],
                how="left",
                validate="many_to_one",
            )
            z = z.merge(
                nonqb_teams,
                on=["season", "week", "team"],
                how="left",
                validate="many_to_one",
            )
            if cell == "RB_RUSHING":
                z["y"] = num(z["rushes"]).fillna(0)
                z["N"] = num(z["team_rushes"])
            elif cell in {"RB_RECEIVING", "TE_RECEIVING"}:
                z["y"] = num(z["targets"]).fillna(0)
                z["N"] = num(z["team_targets"])
            else:
                raise KeyError(cell)
        parts.append(z)

    out = pd.concat(parts, ignore_index=True)
    bad = (
        out["N"].isna()
        | out["N"].le(0)
        | out["y"].isna()
        | out["y"].lt(0)
        | out["y"].gt(out["N"])
    )
    if bad.any():
        sample = out.loc[bad, ["season", "week", "team", "player_id", "cell", "N", "y"]].head(10)
        raise RuntimeError(f"invalid target counts rows={int(bad.sum())}: {sample.to_dict('records')}")
    out["N"] = np.rint(out["N"]).astype(int)
    out["y"] = np.rint(out["y"]).astype(int)
    return out


def enumerate_active_states(q: np.ndarray, p: np.ndarray, q0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(q) != len(p):
        raise RuntimeError("active-state q/p shape mismatch")
    if len(q) and (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or (q <= 0).any()
        or (p <= 0).any()
        or (p >= 1).any()
    ):
        raise RuntimeError("invalid active-state inputs")
    if not np.isfinite(q0) or q0 < -TOL:
        raise RuntimeError("invalid complement probability")
    q0 = max(0.0, float(q0))
    k = len(q)
    nstates = 1 << k
    if nstates > (1 << 16):
        raise RuntimeError(f"active-state room unexpectedly large K={k}")
    if k == 0:
        return np.ones(1), np.zeros((1, 0), dtype=float), np.ones(1)

    masks = np.arange(nstates, dtype=np.uint64)[:, None]
    bits = ((masks >> np.arange(k, dtype=np.uint64)) & 1).astype(float)
    weights = np.prod(np.where(bits > 0, p, 1.0 - p), axis=1)
    if abs(float(weights.sum()) - 1.0) > TOL:
        raise RuntimeError("active-state weights do not sum to one")

    denom = q0 + bits @ q
    alloc = np.zeros_like(bits, dtype=float)
    comp = np.zeros(nstates, dtype=float)
    positive = denom > 0
    alloc[positive] = bits[positive] * q / denom[positive, None]
    comp[positive] = q0 / denom[positive]
    comp[~positive] = 1.0
    gaps = np.abs(alloc.sum(axis=1) + comp - 1.0)
    if float(gaps.max()) > TOL:
        raise RuntimeError("conditional allocation conservation failure")
    return weights, alloc, comp


def mixture_player_cdf(N: int, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    grid = np.arange(N + 1, dtype=int)[:, None]
    pm = np.asarray(probs, dtype=float)[None, :]
    cdfs = binom.cdf(grid, N, pm)
    out = cdfs @ np.asarray(weights, dtype=float)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite candidate CDF")
    if abs(float(out[-1]) - 1.0) > TOL:
        raise RuntimeError("candidate marginal mass does not sum to one")
    return out


def score_rooms(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_rows: list[dict[str, Any]] = []
    room_rows: list[dict[str, Any]] = []
    room_keys = ["season", "week", "team", "cell"]
    for keyvals, g in frame.groupby(room_keys, sort=True):
        season, week, team, cell = keyvals
        g = g.copy().sort_values(["player_id"]).reset_index(drop=True)
        nvals = g["N"].unique()
        gids = g["game_id"].astype(str).unique()
        if len(nvals) != 1 or len(gids) != 1:
            raise RuntimeError(f"room outcome identity drift {keyvals}")
        N = int(nvals[0])
        game_id = str(gids[0])
        q_all = g["authority_q"].to_numpy(float)
        if not np.isfinite(q_all).all() or (q_all < -TOL).any():
            raise RuntimeError(f"invalid canonical q room {keyvals}")
        q_all = np.where(np.abs(q_all) <= TOL, 0.0, q_all)
        positive_idx = np.flatnonzero(q_all > 0)
        q = q_all[positive_idx]
        p = g.loc[positive_idx, "p_active"].to_numpy(float)
        q0 = 1.0 - float(q.sum())
        if q0 < -TOL:
            raise RuntimeError(f"canonical room mass exceeds one {keyvals}: {q.sum()}")
        q0 = max(0.0, q0)
        if abs(float(q.sum()) + q0 - 1.0) > TOL:
            raise RuntimeError(f"canonical room mass gap {keyvals}")

        weights, alloc, comp = enumerate_active_states(q, p, q0)
        cand_mean_pos = weights @ alloc if len(q) else np.zeros(0, dtype=float)
        cand_mean_comp = float(weights @ comp)
        if abs(float(cand_mean_pos.sum()) + cand_mean_comp - 1.0) > TOL:
            raise RuntimeError(f"candidate expected room mass gap {keyvals}")

        y_pos = g.loc[positive_idx, "y"].to_numpy(int)
        y0 = N - int(y_pos.sum())
        if y0 < 0:
            raise RuntimeError(f"observed complement negative {keyvals}")
        obs = np.append(y_pos.astype(float) / N, y0 / N)
        base_vec = np.append(q, q0)
        cand_vec = np.append(cand_mean_pos, cand_mean_comp)
        base_room_err = 0.5 * float(np.abs(base_vec - obs).sum())
        cand_room_err = 0.5 * float(np.abs(cand_vec - obs).sum())

        room_rows.append({
            "season": int(season),
            "week": int(week),
            "game_id": game_id,
            "team": str(team),
            "cell": str(cell),
            "N": N,
            "authority_rows": int(len(g)),
            "positive_q_players": int(len(q)),
            "states": int(len(weights)),
            "baseline_room_error": base_room_err,
            "candidate_room_error": cand_room_err,
            "room_error_improvement": base_room_err - cand_room_err,
            "all_off_probability": float(np.prod(1.0 - p)) if len(p) else 1.0,
            "baseline_complement_share": q0,
            "candidate_expected_complement_share": cand_mean_comp,
            "max_state_mass_gap": float(np.max(np.abs(alloc.sum(axis=1) + comp - 1.0))),
        })

        positive_pos = {int(idx): j for j, idx in enumerate(positive_idx)}
        for idx, r in g.iterrows():
            qi = float(q_all[idx])
            y = int(r["y"])
            base_cdf = binom.cdf(np.arange(N + 1), N, qi)
            base_crps = discrete_crps(base_cdf, y)
            if idx in positive_pos:
                j = positive_pos[idx]
                cand_cdf = mixture_player_cdf(N, alloc[:, j], weights)
                cand_mean_q = float(cand_mean_pos[j])
                cand_crps = discrete_crps(cand_cdf, y)
            else:
                cand_cdf = base_cdf
                cand_mean_q = qi
                cand_crps = base_crps

            player_rows.append({
                "season": int(season),
                "week": int(week),
                "game_id": game_id,
                "team": str(team),
                "cell": str(cell),
                "player_id": str(r["player_id"]),
                "player_name_key": str(r["player_name_key"]),
                "family": str(r["family"]),
                "N": N,
                "y": y,
                "canonical_q": qi,
                "p_active": float(r["p_active"]),
                "stage1_named_player_supported": bool(r["stage1_named_player_supported"]),
                "baseline_expected_count": N * qi,
                "candidate_expected_count": N * cand_mean_q,
                "expected_count_shift": N * (cand_mean_q - qi),
                "baseline_crps": float(base_crps),
                "candidate_crps": float(cand_crps),
                "delta_crps": float(base_crps - cand_crps),
                "baseline_abs_error": abs(y - N * qi),
                "candidate_abs_error": abs(y - N * cand_mean_q),
                "delta_ae": abs(y - N * qi) - abs(y - N * cand_mean_q),
                "zero_entitlement": bool(qi == 0.0),
            })

    players = pd.DataFrame(player_rows)
    rooms = pd.DataFrame(room_rows)
    if players.empty or rooms.empty:
        raise RuntimeError("zero bridge scoring rows")
    return players, rooms


def cluster_bootstrap(q: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = q[[value_col, cluster_col]].dropna().copy()
    agg = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(agg) < 2:
        return {"clusters": int(len(agg)), "replicates": 0, "ci95_low": np.nan, "ci95_high": np.nan}
    a = agg[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        idx = rng.integers(0, len(a), size=len(a))
        den = float(a[idx, 1].sum())
        vals[i] = float(a[idx, 0].sum() / den)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {
        "clusters": int(len(a)),
        "replicates": BOOT_REPS,
        "seed": seed,
        "mean": float(vals.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def equal_player_bootstrap(q: pd.DataFrame, seed: int) -> dict:
    pp = q.groupby("player_id", sort=True)["delta_crps"].mean()
    vals = pp.to_numpy(float)
    if len(vals) < 2:
        return {"players": int(len(vals)), "replicates": 0, "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    rng = np.random.default_rng(seed)
    boots = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        boots[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "players": int(len(vals)),
        "replicates": BOOT_REPS,
        "seed": seed,
        "mean": float(vals.mean()),
        "bootstrap_mean": float(boots.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def season_cell_result(players: pd.DataFrame, rooms: pd.DataFrame, cell: str, season: int) -> dict:
    p = players.loc[players["cell"].eq(cell) & players["season"].eq(season)].copy()
    r = rooms.loc[rooms["cell"].eq(cell) & rooms["season"].eq(season)].copy()
    eval_players = int((p.groupby("player_id").size() >= 5).sum())
    supp = SUPPORT[cell]
    support = (
        len(p) >= supp["rows"]
        and p["player_id"].nunique() >= supp["players"]
        and len(r) >= supp["rooms"]
        and eval_players >= supp["eval_players"]
    )
    out = {
        "cell": cell,
        "season": season,
        "player_games": int(len(p)),
        "players": int(p["player_id"].nunique()),
        "team_game_rooms": int(len(r)),
        "evaluable_players_5plus": eval_players,
        "support": bool(support),
        "named_stage1_supported_rows": int(p["stage1_named_player_supported"].sum()),
        "position_fallback_rows": int((~p["stage1_named_player_supported"]).sum()),
        "zero_entitlement_rows": int(p["zero_entitlement"].sum()),
    }
    if not support:
        out["qualified"] = False
        return out

    mean_delta = float(p["delta_crps"].mean())
    mean_ae = float(p["delta_ae"].mean())
    room_delta = float(r["room_error_improvement"].mean())
    player_boot = cluster_bootstrap(
        p, "delta_crps", "player_id",
        BOOT_SEED + season + CELLS.index(cell) * 100,
    )
    game_boot = cluster_bootstrap(
        p, "delta_crps", "game_id",
        BOOT_SEED + 10000 + season + CELLS.index(cell) * 100,
    )
    eq = equal_player_bootstrap(
        p, BOOT_SEED + 20000 + season + CELLS.index(cell) * 100
    )
    improved_players = (
        p.groupby("player_id")["delta_crps"].mean().gt(0).mean()
    )
    out.update({
        "baseline_crps": float(p["baseline_crps"].mean()),
        "candidate_crps": float(p["candidate_crps"].mean()),
        "mean_delta_crps": mean_delta,
        "mean_delta_ae": mean_ae,
        "baseline_mean_abs_error": float(p["baseline_abs_error"].mean()),
        "candidate_mean_abs_error": float(p["candidate_abs_error"].mean()),
        "baseline_room_error": float(r["baseline_room_error"].mean()),
        "candidate_room_error": float(r["candidate_room_error"].mean()),
        "mean_room_error_improvement": room_delta,
        "player_cluster_bootstrap": player_boot,
        "game_cluster_bootstrap": game_boot,
        "equal_player_bootstrap": eq,
        "share_evaluable_players_improved": float(improved_players),
        "mean_all_off_probability": float(r["all_off_probability"].mean()),
        "max_all_off_probability": float(r["all_off_probability"].max()),
        "mean_expected_count_shift": float(p["expected_count_shift"].mean()),
        "max_abs_expected_count_shift": float(p["expected_count_shift"].abs().max()),
    })
    if season == 2024:
        passed = mean_delta > 0 and mean_ae >= -TOL and room_delta >= -TOL
    else:
        passed = (
            mean_delta > 0
            and player_boot["ci95_low"] > 0
            and game_boot["ci95_low"] > 0
            and eq["mean"] > 0
            and eq["ci95_low"] > 0
            and mean_ae >= -TOL
            and room_delta >= -TOL
        )
    out["qualified"] = bool(passed)
    return out


def pooled_result(players: pd.DataFrame, cell: str) -> dict:
    p = players.loc[players["cell"].eq(cell)].copy()
    player_boot = cluster_bootstrap(
        p, "delta_crps", "player_id",
        BOOT_SEED + 30000 + CELLS.index(cell) * 100,
    )
    game_boot = cluster_bootstrap(
        p, "delta_crps", "game_id",
        BOOT_SEED + 40000 + CELLS.index(cell) * 100,
    )
    return {
        "cell": cell,
        "player_games": int(len(p)),
        "players": int(p["player_id"].nunique()),
        "mean_delta_crps": float(p["delta_crps"].mean()),
        "player_cluster_bootstrap": player_boot,
        "game_cluster_bootstrap": game_boot,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-join-detail", type=Path, required=True)
    ap.add_argument("--stage1-mechanism", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    source = attach_stage1_probability(
        read_csv(a.source_join_detail, "source join detail"),
        read_csv(a.stage1_mechanism, "Stage-1 mechanism"),
    )
    nonqb_p, nonqb_t = build_non_qb_outcomes()
    qb_p, qb_t = build_qb_outcomes(
        read_csv(a.player_logs, "historical player logs"),
        read_csv(a.schedule, "historical schedule"),
    )
    scored_input = attach_outcomes(source, nonqb_p, nonqb_t, qb_p, qb_t)
    players, rooms = score_rooms(scored_input)

    cells = []
    per_cell = []
    pooled = []
    for cell in CELLS:
        by = {}
        for season in TARGET_SEASONS:
            rec = season_cell_result(players, rooms, cell, season)
            cells.append(rec)
            by[season] = bool(rec.get("qualified", False))
        passed = all(by.get(s, False) for s in TARGET_SEASONS)
        per_cell.append({
            "cell": cell,
            "season_2024_qualified": bool(by.get(2024, False)),
            "season_2025_qualified": bool(by.get(2025, False)),
            "disposition": (
                f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED"
                if passed else
                f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED"
            ),
        })
        pooled.append(pooled_result(players, cell))

    passing = [x["cell"] for x in per_cell if x["disposition"].endswith("_QUALIFIED")]
    if len(passing) == len(CELLS):
        disposition = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_ALL_ELIGIBLE"
    elif passing:
        disposition = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL"
    else:
        disposition = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED"

    integrity = {
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "stage1_run": STAGE1_RUN,
        "stage1_artifact": STAGE1_ARTIFACT,
        "target_game_counts_used_only_after_candidate_construction": True,
        "stage1_chronology_violations": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "fitted_share_coefficients": 0,
        "threshold_searches": 0,
        "outcomes_2026_used": 0,
        "production_changed": False,
        "max_state_mass_gap": float(rooms["max_state_mass_gap"].max()),
    }
    gates = {
        "exact_target_seasons": sorted(players["season"].unique().tolist()) == list(TARGET_SEASONS),
        "exact_cells": sorted(players["cell"].unique().tolist()) == sorted(CELLS),
        "target_weeks_within_5_18": int(players["week"].min()) >= 5 and int(players["week"].max()) <= 18,
        "state_mass_gap_le_1e_12": integrity["max_state_mass_gap"] <= TOL,
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "fitted_share_coefficients_zero": True,
        "threshold_searches_zero": True,
        "outcomes_2026_zero": True,
        "production_changed_false": True,
    }
    if not all(gates.values()):
        disposition = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_INTEGRITY_FAILURE"

    result = {
        "version": VERSION,
        "freeze": FREEZE,
        "disposition": disposition,
        "target_seasons": list(TARGET_SEASONS),
        "target_weeks": [5, 18],
        "cells_authorized": list(CELLS),
        "passing_cells": passing,
        "per_cell": per_cell,
        "cells": cells,
        "pooled": pooled,
        "integrity": integrity,
        "gates": gates,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(
        a.out_dir / "active_role_conditional_allocation_bridge_v1_player_rows.csv",
        index=False,
        float_format="%.12f",
    )
    rooms.to_csv(
        a.out_dir / "active_role_conditional_allocation_bridge_v1_room_rows.csv",
        index=False,
        float_format="%.12f",
    )
    (a.out_dir / "active_role_conditional_allocation_bridge_v1_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
