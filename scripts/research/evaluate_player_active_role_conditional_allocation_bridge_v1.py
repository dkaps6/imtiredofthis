#!/usr/bin/env python3
"""Score the frozen PRISM Stage-2 active-role -> conditional-allocation bridge V1.

The scientific contract is frozen in:
docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md

This evaluator:
- consumes the already-qualified Stage-1 p_active semantics;
- preserves the already-source-qualified canonical entitlement q;
- exactly enumerates independent Bernoulli active sets;
- scores named-player opportunity-count CRPS as the primary endpoint;
- protects expected-count MAE and room allocation error;
- uses no sportsbook input, no 2026 outcomes, and fits no coefficients.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from scripts._opponent_map import canon_team
from scripts.research.audit_player_active_role_conditional_allocation_source_v1 import (
    prepare_stage1,
)

VERSION = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM"
FREEZE_VERSION = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE"
FREEZE_COMMIT = "87400a9c9c1db465c9c4bd70b0cb241dba27c6c9"
SOURCE_RUN = 38955368643
SOURCE_HEAD = "cb65a634a0e09296bfeabc94ffe1e77363a30f8a"
SOURCE_ARTIFACT = 12070806828
SOURCE_DIGEST = "sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83"

SEASONS = (2024, 2025)
CELLS = ("QB_PASSING", "RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")
BOOT_REPS = 5000
BOOT_SEED = 20261016
TOL = 1e-12

SUPPORT = {
    "QB_PASSING": {"rows": 350, "players": 45, "games": 300, "eval_players": 30},
    "RB_RUSHING": {"rows": 1200, "players": 100, "games": 400, "eval_players": 75},
    "RB_RECEIVING": {"rows": 1200, "players": 100, "games": 400, "eval_players": 75},
    "TE_RECEIVING": {"rows": 1200, "players": 100, "games": 400, "eval_players": 75},
}

COUNT_COL = {
    "QB_PASSING": "pass_att",
    "RB_RUSHING": "rushes",
    "RB_RECEIVING": "targets",
    "TE_RECEIVING": "targets",
}
TEAM_N_COL = {
    "QB_PASSING": "team_dropbacks",
    "RB_RUSHING": "team_rushes",
    "RB_RECEIVING": "team_targets",
    "TE_RECEIVING": "team_targets",
}
FAMILY = {
    "QB_PASSING": "QB",
    "RB_RUSHING": "RB",
    "RB_RECEIVING": "RB",
    "TE_RECEIVING": "TE",
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    forbidden = [
        c for c in x.columns
        if any(t in c for t in (
            "sportsbook", "bookmaker", "prop_line", "market_line",
            "over_odds", "under_odds", "moneyline", "closing_line",
            "no_vig", "implied_prob",
        ))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden sportsbook columns in {label}: {forbidden}")
    return x


def discrete_crps(cdf: np.ndarray, y: int) -> float:
    k = np.arange(len(cdf), dtype=int)
    obs = (k >= int(y)).astype(float)
    return float(np.square(np.asarray(cdf, dtype=float) - obs).sum())


def enumerate_active_states(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return state weights, named-player state probabilities, complement probabilities."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(q) != len(p):
        raise RuntimeError("q/p length mismatch")
    if not np.isfinite(q).all() or (q <= 0).any():
        raise RuntimeError("active-state q must be finite and strictly positive")
    if not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
        raise RuntimeError("active-state p must be finite and interior to (0,1)")
    q0 = 1.0 - float(q.sum())
    if q0 < -TOL:
        raise RuntimeError(f"canonical named mass exceeds one: q0={q0}")
    if abs(q0) <= TOL:
        q0 = 0.0

    weights: list[float] = []
    named: list[np.ndarray] = []
    comp: list[float] = []
    for bits in itertools.product((0, 1), repeat=len(q)):
        a = np.asarray(bits, dtype=float)
        w = float(np.prod(np.where(a > 0, p, 1.0 - p)))
        z = float(q0 + np.dot(a, q))
        if z > TOL:
            r = (a * q) / z
            r0 = q0 / z
        else:
            # Frozen degenerate all-off fallback: anonymous complement receives
            # the finite opportunity; no zero-q backup gets invented entitlement.
            r = np.zeros(len(q), dtype=float)
            r0 = 1.0
        if not np.isfinite(r).all() or not np.isfinite(r0):
            raise RuntimeError("non-finite conditional allocation")
        if (r < -TOL).any() or r0 < -TOL:
            raise RuntimeError("negative conditional allocation")
        if abs(float(r.sum()) + float(r0) - 1.0) > TOL:
            raise RuntimeError("conditional allocation conservation drift")
        weights.append(w)
        named.append(r)
        comp.append(float(r0))

    w = np.asarray(weights, dtype=float)
    r = np.vstack(named) if named else np.zeros((1, 0), dtype=float)
    r0 = np.asarray(comp, dtype=float)
    if abs(float(w.sum()) - 1.0) > TOL:
        raise RuntimeError("active-state probability mass drift")
    return w, r, r0


def mixture_binom_cdf(N: int, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ks = np.arange(int(N) + 1)
    out = np.zeros(len(ks), dtype=float)
    for pr, w in zip(np.asarray(probs, dtype=float), np.asarray(weights, dtype=float)):
        out += float(w) * binom(int(N), float(pr)).cdf(ks)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite mixture CDF")
    return out


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna().copy()
    g = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(g) < 2:
        return {"clusters": int(len(g)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    a = g[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    n = len(a)
    for i in range(BOOT_REPS):
        idx = rng.integers(0, n, size=n)
        total = a[idx].sum(axis=0)
        vals[i] = total[0] / total[1] if total[1] > 0 else np.nan
    vals = vals[np.isfinite(vals)]
    return {
        "clusters": int(n),
        "valid_reps": int(len(vals)),
        "ci95_low": float(np.quantile(vals, 0.025)) if len(vals) else math.nan,
        "ci95_high": float(np.quantile(vals, 0.975)) if len(vals) else math.nan,
    }


def equal_player_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("player_id", sort=True):
        if len(g) < 5:
            continue
        rows.append({
            "player_id": str(pid),
            "rows": int(len(g)),
            "cell": str(g["cell"].iloc[0]),
            "season": int(g["season"].iloc[0]) if g["season"].nunique() == 1 else 0,
            "mean_delta_crps": float(g["delta_crps"].mean()),
            "mean_delta_ae": float(g["delta_ae"].mean()),
            "canonical_zero_share_rate": float(g["canonical_zero_share"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(player_df: pd.DataFrame, seed: int) -> dict:
    vals = num(player_df["mean_delta_crps"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {"players": int(len(vals)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        out[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "players": int(len(vals)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(out, 0.025)),
        "ci95_high": float(np.quantile(out, 0.975)),
    }


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    required = {"season", "week", "team", "opponent"}
    if required - set(s.columns):
        raise RuntimeError(f"schedule missing {sorted(required-set(s.columns))}")
    s["season"] = num(s["season"]).astype("Int64")
    s["week"] = num(s["week"]).astype("Int64")
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    if "game_id" not in s:
        s["game_id"] = ""
    s = s.loc[s["season"].isin(SEASONS) & s["week"].between(5, 18)].copy()
    game = s["game_id"].astype("string").fillna("").str.strip()
    fallback = [
        f"{int(season)}_{int(week):02d}_{'_'.join(sorted([str(team), str(opp)]))}"
        for season, week, team, opp in zip(s["season"], s["week"], s["team"], s["opponent"])
    ]
    s["game_key"] = np.where(game.ne(""), game.astype(str), fallback)
    return s[["season", "week", "team", "opponent", "game_key"]].drop_duplicates(
        ["season", "week", "team"]
    )


def prepare_outcomes(logs: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "season", "week", "team", "player_id", "targets", "rushes", "pass_att",
        "team_targets", "team_rushes", "team_dropbacks",
    }
    if required - set(logs.columns):
        raise RuntimeError(f"player logs missing {sorted(required-set(logs.columns))}")
    x = logs.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].astype("string").fillna("").str.strip()
    x = x.loc[x["season"].isin(SEASONS) & x["week"].between(5, 18)].copy()
    for c in ("targets", "rushes", "pass_att", "team_targets", "team_rushes", "team_dropbacks"):
        x[c] = num(x[c])

    # One observed row for every player who recorded offensive usage. Missing
    # source-universe players are valid zero-opportunity outcomes.
    player = x[
        ["season", "week", "team", "player_id", "targets", "rushes", "pass_att"]
    ].copy()
    if player.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError("duplicate outcome player identity")

    team_rows = []
    for (season, week, team), g in x.groupby(["season", "week", "team"], sort=False):
        rec = {"season": int(season), "week": int(week), "team": str(team)}
        for c in ("team_targets", "team_rushes", "team_dropbacks"):
            vals = num(g[c]).dropna().unique()
            if len(vals) != 1:
                raise RuntimeError(f"team outcome denominator drift {season} W{week} {team} {c}: {vals[:5]}")
            rec[c] = float(vals[0])
        team_rows.append(rec)
    team = pd.DataFrame(team_rows).merge(
        schedule, on=["season", "week", "team"], how="left", validate="one_to_one"
    )
    if team["game_key"].isna().any():
        raise RuntimeError("outcome team rows missing game key")
    return player, team


def attach_stage1_probability(
    detail: pd.DataFrame,
    stage1_universe: pd.DataFrame,
    stage1_mechanism: pd.DataFrame,
) -> pd.DataFrame:
    _, mechanism = prepare_stage1(stage1_universe, stage1_mechanism)
    keep = [
        "season", "week", "team", "family", "player_id",
        "stage1_p_active", "stage1_named_player_supported",
    ]
    z = detail.merge(
        mechanism[keep],
        on=["season", "week", "team", "family", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if z["stage1_p_active"].isna().any():
        raise RuntimeError(f"Stage-1 p_active attach misses={int(z['stage1_p_active'].isna().sum())}")
    p = num(z["stage1_p_active"])
    if not ((p > 0) & (p < 1) & np.isfinite(p)).all():
        raise RuntimeError("Stage-1 p_active escaped open interval")
    return z


def score_room(
    g: pd.DataFrame,
    *,
    N: int,
    count_col: str,
    game_key: str,
) -> tuple[list[dict], dict]:
    if N <= 0:
        raise RuntimeError("non-positive finite opportunity total")
    g = g.copy().sort_values(["player_id"]).reset_index(drop=True)
    q_all = num(g["authority_q"]).to_numpy(float)
    if not np.isfinite(q_all).all() or (q_all < -TOL).any():
        raise RuntimeError("invalid canonical entitlement")

    # Positive canonical entitlement is explicit in the exact active-set law.
    # Canonical-zero named players remain named marginal rows but sit inside the
    # anonymous complement for the room allocation.
    pos = q_all > TOL
    pos_idx = np.flatnonzero(pos)
    q = q_all[pos]
    p = num(g.loc[pos, "stage1_p_active"]).to_numpy(float)
    q0 = 1.0 - float(q.sum())
    if q0 < -TOL:
        raise RuntimeError(f"room canonical mass exceeds one: {q.sum()}")
    if abs(q0) <= TOL:
        q0 = 0.0

    if len(q):
        weights, state_named, state_comp = enumerate_active_states(q, p)
        expected_named = weights @ state_named
        expected_comp = float(weights @ state_comp)
        all_inactive_weight = float(np.prod(1.0 - p))
        degenerate_fallback_weight = all_inactive_weight if q0 == 0.0 else 0.0
    else:
        weights = np.asarray([1.0])
        state_named = np.zeros((1, 0), dtype=float)
        state_comp = np.asarray([1.0])
        expected_named = np.zeros(0, dtype=float)
        expected_comp = 1.0
        all_inactive_weight = 1.0
        degenerate_fallback_weight = 1.0

    y_all = num(g[count_col]).fillna(0.0).to_numpy(float)
    if not np.allclose(y_all, np.rint(y_all), atol=TOL):
        raise RuntimeError("non-integer player opportunity count")
    y_all = np.rint(y_all).astype(int)
    if (y_all < 0).any() or (y_all > N).any():
        raise RuntimeError("player opportunity count outside finite team total")

    y_pos = y_all[pos]
    y_comp = int(N - int(y_pos.sum()))
    if y_comp < 0:
        raise RuntimeError("explicit player opportunity counts exceed team total")

    base_vec = np.append(q, q0)
    cand_vec = np.append(expected_named, expected_comp)
    obs_vec = np.append(y_pos.astype(float) / N, float(y_comp) / N)
    if abs(float(base_vec.sum()) - 1.0) > TOL or abs(float(cand_vec.sum()) - 1.0) > TOL:
        raise RuntimeError("room expected-share conservation failure")
    room_base_error = 0.5 * float(np.abs(base_vec - obs_vec).sum())
    room_cand_error = 0.5 * float(np.abs(cand_vec - obs_vec).sum())

    rows: list[dict] = []
    positive_position = {int(idx): j for j, idx in enumerate(pos_idx.tolist())}
    ks = np.arange(N + 1)
    for idx, r in g.iterrows():
        qi = float(q_all[idx])
        yi = int(y_all[idx])
        base_cdf = binom(N, min(max(qi, 0.0), 1.0)).cdf(ks)
        base_crps = discrete_crps(base_cdf, yi)
        base_mu = N * qi

        if idx in positive_position:
            j = positive_position[idx]
            probs = state_named[:, j]
            cand_cdf = mixture_binom_cdf(N, probs, weights)
            cand_mu = N * float(expected_named[j])
        else:
            cand_cdf = base_cdf
            cand_mu = base_mu
        cand_crps = discrete_crps(cand_cdf, yi)

        rows.append({
            "season": int(r["season"]),
            "week": int(r["week"]),
            "game_key": str(game_key),
            "team": str(r["team"]),
            "cell": str(r["cell"]),
            "family": str(r["family"]),
            "player_id": str(r["player_id"]),
            "player_name_key": str(r["player_name_key"]),
            "stage1_p_active": float(r["stage1_p_active"]),
            "stage1_named_player_supported": bool(r["stage1_named_player_supported"]),
            "N": int(N),
            "y": yi,
            "authority_q": qi,
            "canonical_zero_share": bool(qi <= TOL),
            "baseline_expected_count": float(base_mu),
            "candidate_expected_count": float(cand_mu),
            "candidate_expected_shift": float(cand_mu - base_mu),
            "baseline_crps": float(base_crps),
            "candidate_crps": float(cand_crps),
            "delta_crps": float(base_crps - cand_crps),
            "baseline_ae": float(abs(yi - base_mu)),
            "candidate_ae": float(abs(yi - cand_mu)),
            "delta_ae": float(abs(yi - base_mu) - abs(yi - cand_mu)),
        })

    room = {
        "season": int(g["season"].iloc[0]),
        "week": int(g["week"].iloc[0]),
        "game_key": str(game_key),
        "team": str(g["team"].iloc[0]),
        "cell": str(g["cell"].iloc[0]),
        "N": int(N),
        "authority_rows": int(len(g)),
        "positive_q_players": int(len(q)),
        "canonical_zero_rows": int((~pos).sum()),
        "baseline_room_error": room_base_error,
        "candidate_room_error": room_cand_error,
        "delta_room_error": room_base_error - room_cand_error,
        "all_positive_q_players_inactive_weight": all_inactive_weight,
        "degenerate_all_off_fallback_weight": degenerate_fallback_weight,
        "baseline_complement_share": float(q0),
        "candidate_expected_complement_share": float(expected_comp),
        "complement_expected_shift": float(expected_comp - q0),
        "state_count": int(len(weights)),
        "state_weight_gap": float(abs(float(weights.sum()) - 1.0)),
        "candidate_mass_gap": float(abs(float(cand_vec.sum()) - 1.0)),
    }
    return rows, room


def support_gates(cell: str, rows: pd.DataFrame, rooms: pd.DataFrame, players: pd.DataFrame) -> dict:
    req = SUPPORT[cell]
    return {
        "rows": int(len(rows)) >= req["rows"],
        "players": int(rows["player_id"].nunique()) >= req["players"],
        "games": int(rooms["game_key"].nunique()) >= req["games"],
        "evaluable_players": int(len(players)) >= req["eval_players"],
    }


def score_slice(rows: pd.DataFrame, rooms: pd.DataFrame, *, cell: str, season: int) -> tuple[dict, pd.DataFrame]:
    r = rows.loc[rows["cell"].eq(cell) & rows["season"].eq(season)].copy()
    rm = rooms.loc[rooms["cell"].eq(cell) & rooms["season"].eq(season)].copy()
    players = equal_player_frame(r)
    support = support_gates(cell, r, rm, players)

    rec = {
        "cell": cell,
        "season": season,
        "rows": int(len(r)),
        "players": int(r["player_id"].nunique()),
        "team_rooms": int(len(rm)),
        "games": int(rm["game_key"].nunique()),
        "evaluable_players_ge5": int(len(players)),
        "support_gates": support,
        "stage1_named_player_supported_rate": float(r["stage1_named_player_supported"].mean()) if len(r) else math.nan,
        "canonical_zero_share_rows": int(r["canonical_zero_share"].sum()) if len(r) else 0,
        "mean_delta_crps": float(r["delta_crps"].mean()) if len(r) else math.nan,
        "mean_delta_ae": float(r["delta_ae"].mean()) if len(r) else math.nan,
        "mean_delta_room_error": float(rm["delta_room_error"].mean()) if len(rm) else math.nan,
        "mean_candidate_expected_shift": float(r["candidate_expected_shift"].mean()) if len(r) else math.nan,
        "share_player_rows_crps_improved": float((r["delta_crps"] > 0).mean()) if len(r) else math.nan,
        "share_player_rows_ae_improved": float((r["delta_ae"] > 0).mean()) if len(r) else math.nan,
        "mean_all_inactive_state_weight": float(rm["all_positive_q_players_inactive_weight"].mean()) if len(rm) else math.nan,
        "mean_degenerate_fallback_weight": float(rm["degenerate_all_off_fallback_weight"].mean()) if len(rm) else math.nan,
        "max_state_weight_gap": float(rm["state_weight_gap"].max()) if len(rm) else math.inf,
        "max_candidate_mass_gap": float(rm["candidate_mass_gap"].max()) if len(rm) else math.inf,
    }

    if len(r):
        rec["player_cluster_bootstrap"] = cluster_bootstrap(
            r, "delta_crps", "player_id", BOOT_SEED + season + CELLS.index(cell) * 100
        )
        rec["game_cluster_bootstrap"] = cluster_bootstrap(
            r, "delta_crps", "game_key", BOOT_SEED + 10000 + season + CELLS.index(cell) * 100
        )
    else:
        rec["player_cluster_bootstrap"] = {}
        rec["game_cluster_bootstrap"] = {}

    if len(players):
        rec["equal_player_mean_delta_crps"] = float(players["mean_delta_crps"].mean())
        rec["equal_player_share_improved"] = float((players["mean_delta_crps"] > 0).mean())
        rec["equal_player_bootstrap"] = equal_player_bootstrap(
            players, BOOT_SEED + 20000 + season + CELLS.index(cell) * 100
        )
    else:
        rec["equal_player_mean_delta_crps"] = math.nan
        rec["equal_player_share_improved"] = math.nan
        rec["equal_player_bootstrap"] = {}

    return rec, players


def pooled_diagnostics(rows: pd.DataFrame, rooms: pd.DataFrame, cell: str) -> dict:
    r = rows.loc[rows["cell"].eq(cell)].copy()
    rm = rooms.loc[rooms["cell"].eq(cell)].copy()
    r["pooled_player_key"] = r["season"].astype(str) + "|" + r["player_id"].astype(str)
    r["pooled_game_key"] = r["season"].astype(str) + "|" + r["game_key"].astype(str)
    return {
        "rows": int(len(r)),
        "mean_delta_crps": float(r["delta_crps"].mean()),
        "mean_delta_ae": float(r["delta_ae"].mean()),
        "mean_delta_room_error": float(rm["delta_room_error"].mean()),
        "player_cluster_bootstrap": cluster_bootstrap(
            r, "delta_crps", "pooled_player_key", BOOT_SEED + 30000 + CELLS.index(cell) * 100
        ),
        "game_cluster_bootstrap": cluster_bootstrap(
            r, "delta_crps", "pooled_game_key", BOOT_SEED + 40000 + CELLS.index(cell) * 100
        ),
    }


def qualification(cell: str, m24: dict, m25: dict) -> tuple[dict, str]:
    gates = {
        "support_2024": all(m24["support_gates"].values()),
        "support_2025": all(m25["support_gates"].values()),
        "development_2024_crps_gt_0": m24["mean_delta_crps"] > 0,
        "development_2024_mean_ae_nonworse": m24["mean_delta_ae"] >= -TOL,
        "development_2024_room_error_nonworse": m24["mean_delta_room_error"] >= -TOL,
        "confirm_2025_crps_gt_0": m25["mean_delta_crps"] > 0,
        "confirm_2025_player_ci_low_gt_0": m25["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0,
        "confirm_2025_game_ci_low_gt_0": m25["game_cluster_bootstrap"].get("ci95_low", math.nan) > 0,
        "confirm_2025_equal_player_mean_gt_0": m25["equal_player_mean_delta_crps"] > 0,
        "confirm_2025_equal_player_ci_low_gt_0": m25["equal_player_bootstrap"].get("ci95_low", math.nan) > 0,
        "confirm_2025_mean_ae_nonworse": m25["mean_delta_ae"] >= -TOL,
        "confirm_2025_room_error_nonworse": m25["mean_delta_room_error"] >= -TOL,
        "state_weight_integrity_2024": m24["max_state_weight_gap"] <= TOL,
        "state_weight_integrity_2025": m25["max_state_weight_gap"] <= TOL,
        "candidate_mass_integrity_2024": m24["max_candidate_mass_gap"] <= TOL,
        "candidate_mass_integrity_2025": m25["max_candidate_mass_gap"] <= TOL,
    }
    disposition = (
        f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED"
        if all(gates.values())
        else f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED"
    )
    return gates, disposition


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-result", type=Path, required=True)
    ap.add_argument("--source-detail", type=Path, required=True)
    ap.add_argument("--stage1-universe", type=Path, required=True)
    ap.add_argument("--stage1-mechanism", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
        if source_result.get("disposition") != "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL":
            raise RuntimeError(f"unexpected source disposition: {source_result.get('disposition')}")
        if source_result.get("ready_cells") != list(CELLS):
            raise RuntimeError(f"source-ready cell drift: {source_result.get('ready_cells')}")

        detail = read_csv(args.source_detail, "source join detail")
        detail = detail.loc[
            detail["cell"].isin(CELLS)
            & num(detail["season"]).isin(SEASONS)
            & detail["authority_covered"].fillna(False).astype(bool)
            & ~detail["ambiguous_identity"].fillna(False).astype(bool)
        ].copy()
        detail["season"] = num(detail["season"]).astype(int)
        detail["week"] = num(detail["week"]).astype(int)
        detail["team"] = detail["team"].map(canon_team)
        detail["player_id"] = detail["player_id"].astype("string").fillna("").str.strip()
        detail["authority_q"] = num(detail["authority_q"])
        if detail.duplicated(["season", "week", "team", "cell", "player_id"]).any():
            raise RuntimeError("duplicate source detail identity")

        stage1_universe = read_csv(args.stage1_universe, "Stage-1 universe")
        stage1_mechanism = read_csv(args.stage1_mechanism, "Stage-1 mechanism")
        detail = attach_stage1_probability(detail, stage1_universe, stage1_mechanism)

        schedule = prepare_schedule(read_csv(args.schedule, "schedule"))
        player_outcomes, team_outcomes = prepare_outcomes(
            read_csv(args.player_logs, "historical player logs"), schedule
        )

        detail = detail.merge(
            player_outcomes,
            on=["season", "week", "team", "player_id"],
            how="left",
            validate="one_to_one",
        )
        for c in ("targets", "rushes", "pass_att"):
            detail[c] = num(detail[c]).fillna(0.0)

        detail = detail.merge(
            team_outcomes[
                ["season", "week", "team", "team_targets", "team_rushes", "team_dropbacks", "game_key"]
            ],
            on=["season", "week", "team"],
            how="left",
            validate="many_to_one",
        )
        if detail[["game_key", "team_targets", "team_rushes", "team_dropbacks"]].isna().any(axis=None):
            raise RuntimeError("source detail missing target-game team denominator")

        player_rows: list[dict] = []
        room_rows: list[dict] = []
        for (season, week, team, cell), g in detail.groupby(
            ["season", "week", "team", "cell"], sort=True
        ):
            Nraw = float(g[TEAM_N_COL[cell]].iloc[0])
            if not np.isfinite(Nraw) or abs(Nraw - round(Nraw)) > TOL or Nraw <= 0:
                raise RuntimeError(f"invalid team opportunity N {season} W{week} {team} {cell}: {Nraw}")
            N = int(round(Nraw))
            rows, room = score_room(
                g,
                N=N,
                count_col=COUNT_COL[cell],
                game_key=str(g["game_key"].iloc[0]),
            )
            player_rows.extend(rows)
            room_rows.append(room)

        scored = pd.DataFrame(player_rows)
        rooms = pd.DataFrame(room_rows)
        if scored.empty or rooms.empty:
            raise RuntimeError("bridge scorer produced zero rows")

        season_metrics: dict[str, dict] = {}
        player_diag_parts = []
        position_dispositions: dict[str, str] = {}
        qualification_gates: dict[str, dict] = {}
        pooled: dict[str, dict] = {}

        for cell in CELLS:
            season_metrics[cell] = {}
            by_season = {}
            for season in SEASONS:
                m, p = score_slice(scored, rooms, cell=cell, season=season)
                season_metrics[cell][str(season)] = m
                p["scope"] = str(season)
                player_diag_parts.append(p)
                by_season[season] = m
            gates, disp = qualification(cell, by_season[2024], by_season[2025])
            qualification_gates[cell] = gates
            position_dispositions[cell] = disp
            pooled[cell] = pooled_diagnostics(scored, rooms, cell)

        n_qualified = sum(v.endswith("_BRIDGE_QUALIFIED") for v in position_dispositions.values())
        if n_qualified == len(CELLS):
            overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_ALL_ELIGIBLE"
        elif n_qualified:
            overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL"
        else:
            overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED"

        player_diag = pd.concat(player_diag_parts, ignore_index=True) if player_diag_parts else pd.DataFrame()
        scored.to_csv(args.out_dir / "active_role_conditional_allocation_bridge_player_rows.csv", index=False)
        rooms.to_csv(args.out_dir / "active_role_conditional_allocation_bridge_room_rows.csv", index=False)
        player_diag.to_csv(args.out_dir / "active_role_conditional_allocation_bridge_player_diagnostics.csv", index=False)

        result = {
            "version": VERSION,
            "freeze_version": FREEZE_VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "source_run": SOURCE_RUN,
            "source_head": SOURCE_HEAD,
            "source_artifact": SOURCE_ARTIFACT,
            "source_digest": SOURCE_DIGEST,
            "disposition": overall,
            "eligible_cells": list(CELLS),
            "wr_receiving_scored": False,
            "position_dispositions": position_dispositions,
            "season_metrics": season_metrics,
            "pooled_metrics": pooled,
            "qualification_gates": qualification_gates,
            "frozen_parameters": {
                "seasons": list(SEASONS),
                "weeks": [5, 18],
                "active_state_law": "INDEPENDENT_BERNOULLI_EXACT_ENUMERATION",
                "all_off_fallback": "ANONYMOUS_COMPLEMENT_EQUALS_ONE",
                "bootstrap_replicates": BOOT_REPS,
                "bootstrap_seed": BOOT_SEED,
                "tolerance": TOL,
            },
            "target_game_outcomes_used_as_predictors": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "new_share_models_fit": 0,
            "threshold_searches": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_version": FREEZE_VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "source_run": SOURCE_RUN,
            "source_head": SOURCE_HEAD,
            "source_artifact": SOURCE_ARTIFACT,
            "source_digest": SOURCE_DIGEST,
            "disposition": "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_INTEGRITY_FAILURE",
            "error": str(exc),
            "target_game_outcomes_used_as_predictors": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "new_share_models_fit": 0,
            "threshold_searches": 0,
            "production_changed": False,
        }
        (args.out_dir / "active_role_conditional_allocation_bridge_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "active_role_conditional_allocation_bridge_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
