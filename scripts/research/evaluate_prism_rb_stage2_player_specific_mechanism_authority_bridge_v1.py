#!/usr/bin/env python3
"""Score the frozen RB player-specific Stage-2 mechanism-authority bridge V1."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    discrete_crps,
    mixture_binom_cdf,
)

VERSION = "PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1"
FREEZE_COMMIT = "0831ed274161fa75a7a338041c365bda161041ee"
PARENT_RUN = 38957575645
PARENT_ARTIFACT = 12070774378
PARENT_DIGEST = "sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0"
SIGNAL_RUN = 39099164620
SIGNAL_ARTIFACT = 12125359950
SIGNAL_DIGEST = "sha256:efa919e2590e7409e1d989b59cafe293d6d4831dfb434635df8fb5d15143e95b"

CELLS = ("RB_RUSHING", "RB_RECEIVING")
SEASONS = (2024, 2025)
MIN_PLAYER_PRIOR = 3
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-12

SUPPORT = {
    "RB_RUSHING": {
        2024: {"rows": 1250, "players": 140, "rooms": 400, "teams": 28},
        2025: {"rows": 1250, "players": 140, "rooms": 400, "teams": 28},
    },
    "RB_RECEIVING": {
        2024: {"rows": 1350, "players": 145, "rooms": 400, "teams": 28},
        2025: {"rows": 1350, "players": 145, "rooms": 400, "teams": 28},
    },
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


def prepare(parent: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    required_parent = {
        "season", "week", "game_key", "team", "cell", "family", "player_id",
        "player_name_key", "stage1_p_active", "n", "y", "authority_q",
        "candidate_expected_count", "candidate_crps", "candidate_ae",
    }
    if required_parent - set(parent.columns):
        raise RuntimeError(f"parent missing {sorted(required_parent-set(parent.columns))}")
    required_signal = {
        "season", "week", "game_key", "team", "cell", "player_id",
        "authority_q", "player_prior_positive_q_rows", "hierarchical_expected_benefit",
    }
    if required_signal - set(signal.columns):
        raise RuntimeError(f"signal missing {sorted(required_signal-set(signal.columns))}")

    p = parent.loc[parent["cell"].isin(CELLS)].copy()
    s = signal.loc[signal["cell"].isin(CELLS)].copy()
    for z in (p, s):
        z["season"] = num(z["season"]).astype("Int64")
        z["week"] = num(z["week"]).astype("Int64")
        for c in ("game_key", "team", "cell", "player_id"):
            z[c] = z[c].astype("string").fillna("").str.strip()
    for c in (
        "stage1_p_active", "n", "y", "authority_q", "candidate_expected_count",
        "candidate_crps", "candidate_ae",
    ):
        p[c] = num(p[c])
    for c in ("authority_q", "player_prior_positive_q_rows", "hierarchical_expected_benefit"):
        s[c] = num(s[c])

    keys = ["season", "week", "game_key", "team", "cell", "player_id"]
    if p.duplicated(keys).any() or s.duplicated(keys).any():
        raise RuntimeError("duplicate parent/signal identity")
    z = p.merge(
        s[keys + ["authority_q", "player_prior_positive_q_rows", "hierarchical_expected_benefit"]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_signal"),
    )
    if z[["authority_q_signal", "player_prior_positive_q_rows", "hierarchical_expected_benefit"]].isna().any(axis=None):
        raise RuntimeError("signal attach miss")
    if float((z["authority_q"] - z["authority_q_signal"]).abs().max()) > 1e-12:
        raise RuntimeError("parent/signal authority_q mismatch")
    if not z["season"].isin(SEASONS).all():
        raise RuntimeError("unexpected season")
    pactive = num(z["stage1_p_active"])
    if not ((pactive > 0) & (pactive < 1) & np.isfinite(pactive)).all():
        raise RuntimeError("parent p_active outside open interval")
    return z.drop(columns=["authority_q_signal"])


def enumerate_states(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact active-set enumeration allowing deterministic p=1 authority."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(q) != len(p):
        raise RuntimeError("q/p length mismatch")
    if not np.isfinite(q).all() or (q <= 0).any():
        raise RuntimeError("q must be positive finite")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise RuntimeError("p escaped [0,1]")
    q0 = 1.0 - float(q.sum())
    if q0 < -TOL:
        raise RuntimeError(f"canonical mass exceeds one: {q.sum()}")
    if abs(q0) <= TOL:
        q0 = 0.0

    deterministic = p >= 1.0 - TOL
    stochastic_idx = np.flatnonzero(~deterministic)
    weights: list[float] = []
    named: list[np.ndarray] = []
    comp: list[float] = []

    for bits in itertools.product((0, 1), repeat=len(stochastic_idx)):
        a = np.ones(len(q), dtype=float)
        if len(stochastic_idx):
            a[stochastic_idx] = np.asarray(bits, dtype=float)
            pp = p[stochastic_idx]
            bb = a[stochastic_idx]
            w = float(np.prod(np.where(bb > 0, pp, 1.0 - pp)))
        else:
            w = 1.0
        z = float(q0 + np.dot(a, q))
        if z > TOL:
            r = (a * q) / z
            r0 = q0 / z
        else:
            r = np.zeros(len(q), dtype=float)
            r0 = 1.0
        if abs(float(r.sum()) + float(r0) - 1.0) > TOL:
            raise RuntimeError("state conservation drift")
        weights.append(w)
        named.append(r)
        comp.append(float(r0))

    w = np.asarray(weights, dtype=float)
    r = np.vstack(named)
    r0 = np.asarray(comp, dtype=float)
    if abs(float(w.sum()) - 1.0) > TOL:
        raise RuntimeError("state probability drift")
    return w, r, r0


def support_tier(n: int) -> str:
    if n < 3:
        return "LOW_LT3"
    if n <= 5:
        return "TIER2_3_5"
    if n <= 9:
        return "TIER3_6_9"
    return "TIER4_10_PLUS"


def score_room(g: pd.DataFrame) -> tuple[list[dict], dict]:
    g = g.copy().sort_values("player_id").reset_index(drop=True)
    Nraw = float(g["n"].iloc[0])
    if not np.isfinite(Nraw) or abs(Nraw - round(Nraw)) > TOL or Nraw <= 0:
        raise RuntimeError("invalid finite opportunity N")
    N = int(round(Nraw))
    if not np.allclose(num(g["n"]).to_numpy(float), N, atol=TOL):
        raise RuntimeError("room N drift")

    qall = num(g["authority_q"]).to_numpy(float)
    pos = qall > TOL
    idxs = np.flatnonzero(pos)
    q = qall[pos]
    porig = num(g.loc[pos, "stage1_p_active"]).to_numpy(float)
    prior_n = num(g.loc[pos, "player_prior_positive_q_rows"]).fillna(0).to_numpy(int)
    benefit = num(g.loc[pos, "hierarchical_expected_benefit"]).to_numpy(float)
    disable = (prior_n >= MIN_PLAYER_PRIOR) & (benefit <= 0)

    peff = porig.copy()
    peff[disable] = 1.0
    weights, state_named, state_comp = enumerate_states(q, peff)
    expected_named = weights @ state_named
    expected_comp = float(weights @ state_comp)

    yall = num(g["y"]).fillna(0).to_numpy(float)
    if not np.allclose(yall, np.rint(yall), atol=TOL):
        raise RuntimeError("noninteger y")
    yall = np.rint(yall).astype(int)
    ypos = yall[pos]
    if (ypos < 0).any() or (ypos > N).any():
        raise RuntimeError("y outside finite total")

    universal_shares = num(g.loc[pos, "candidate_expected_count"]).to_numpy(float) / N
    universal_comp = 1.0 - float(universal_shares.sum())
    if universal_comp < -TOL:
        raise RuntimeError("universal expected mass exceeds one")
    if abs(universal_comp) <= TOL:
        universal_comp = 0.0
    personalized_vec = np.append(expected_named, expected_comp)
    universal_vec = np.append(universal_shares, universal_comp)
    observed_comp = float(N - int(ypos.sum())) / N
    if observed_comp < -TOL:
        raise RuntimeError("observed named opportunity exceeds team total")
    observed_vec = np.append(ypos.astype(float) / N, observed_comp)

    universal_room_error = 0.5 * float(np.abs(universal_vec - observed_vec).sum())
    personalized_room_error = 0.5 * float(np.abs(personalized_vec - observed_vec).sum())

    pos_map = {int(idx): j for j, idx in enumerate(idxs.tolist())}
    rows: list[dict] = []
    for idx, row in g.iterrows():
        if idx not in pos_map:
            continue
        j = pos_map[idx]
        yi = int(yall[idx])
        probs = state_named[:, j]
        cdf = mixture_binom_cdf(N, probs, weights)
        personalized_crps = discrete_crps(cdf, yi)
        personalized_mu = N * float(expected_named[j])
        universal_crps = float(row["candidate_crps"])
        universal_mu = float(row["candidate_expected_count"])
        universal_ae = float(row["candidate_ae"])
        personalized_ae = float(abs(yi - personalized_mu))
        nprior = int(row["player_prior_positive_q_rows"])
        h = float(row["hierarchical_expected_benefit"])
        own_disabled = bool(nprior >= MIN_PLAYER_PRIOR and h <= 0)
        rows.append({
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_key": str(row["game_key"]),
            "team": str(row["team"]),
            "cell": str(row["cell"]),
            "player_id": str(row["player_id"]),
            "player_name_key": str(row["player_name_key"]),
            "authority_q": float(row["authority_q"]),
            "stage1_p_active": float(row["stage1_p_active"]),
            "p_effective": 1.0 if own_disabled else float(row["stage1_p_active"]),
            "player_prior_positive_q_rows": nprior,
            "support_tier": support_tier(nprior),
            "hierarchical_expected_benefit": h,
            "own_authority_disabled": own_disabled,
            "y": yi,
            "N": N,
            "universal_stage2_expected_count": universal_mu,
            "personalized_expected_count": personalized_mu,
            "universal_stage2_crps": universal_crps,
            "personalized_crps": personalized_crps,
            "delta_personalized_crps": universal_crps - personalized_crps,
            "universal_stage2_ae": universal_ae,
            "personalized_ae": personalized_ae,
            "delta_personalized_ae": universal_ae - personalized_ae,
        })

    room = {
        "season": int(g["season"].iloc[0]),
        "week": int(g["week"].iloc[0]),
        "game_key": str(g["game_key"].iloc[0]),
        "team": str(g["team"].iloc[0]),
        "cell": str(g["cell"].iloc[0]),
        "N": N,
        "positive_q_players": int(len(q)),
        "own_authority_disabled_players": int(disable.sum()),
        "universal_room_error": universal_room_error,
        "personalized_room_error": personalized_room_error,
        "delta_room_error": universal_room_error - personalized_room_error,
        "state_count": int(len(weights)),
        "state_weight_gap": float(abs(float(weights.sum()) - 1.0)),
        "candidate_mass_gap": float(abs(float(personalized_vec.sum()) - 1.0)),
        "universal_mass_gap": float(abs(float(universal_vec.sum()) - 1.0)),
    }
    return rows, room


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, seed: int) -> dict:
    z = df[[value_col, cluster_col]].dropna()
    agg = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(agg) < 2:
        return {"clusters": int(len(agg)), "valid_reps": 0, "ci95_low": math.nan, "ci95_high": math.nan}
    a = agg[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        idx = rng.integers(0, len(a), size=len(a))
        total = a[idx].sum(axis=0)
        vals[i] = total[0] / total[1]
    return {
        "clusters": int(len(a)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
    }


def equal_player_frame(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, x in g.groupby("player_id", sort=True):
        if len(x) < 5:
            continue
        rows.append({
            "player_id": str(pid),
            "rows": int(len(x)),
            "mean_delta_crps": float(x["delta_personalized_crps"].mean()),
            "universal_stage2_crps": float(x["universal_stage2_crps"].mean()),
            "own_disabled_rows": int(x["own_authority_disabled"].sum()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict:
    vals = num(p["mean_delta_crps"]).dropna().to_numpy(float)
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


def score_slice(rows: pd.DataFrame, rooms: pd.DataFrame, cell: str, season: int) -> tuple[dict, pd.DataFrame]:
    g = rows.loc[rows["cell"].eq(cell) & rows["season"].eq(season)].copy()
    rm = rooms.loc[rooms["cell"].eq(cell) & rooms["season"].eq(season)].copy()
    req = SUPPORT[cell][season]
    support = {
        "rows": int(len(g)) >= req["rows"],
        "players": int(g["player_id"].nunique()) >= req["players"],
        "rooms": int(len(rm)) >= req["rooms"],
        "teams": int(g["team"].nunique()) >= req["teams"],
    }
    p = equal_player_frame(g)
    vals = num(p["mean_delta_crps"]).dropna() if len(p) else pd.Series(dtype=float)
    tol = 1e-12
    low = g.loc[g["player_prior_positive_q_rows"].lt(MIN_PLAYER_PRIOR)]
    by_tier = {}
    for tier, tg in g.groupby("support_tier", sort=True):
        by_tier[str(tier)] = {
            "rows": int(len(tg)),
            "players": int(tg["player_id"].nunique()),
            "mean_delta_crps": float(tg["delta_personalized_crps"].mean()),
        }
    rec = {
        "cell": cell,
        "season": season,
        "rows": int(len(g)),
        "players": int(g["player_id"].nunique()),
        "teams": int(g["team"].nunique()),
        "rooms": int(len(rm)),
        "support_gates": support,
        "own_authority_disabled_rows": int(g["own_authority_disabled"].sum()),
        "own_authority_disabled_players": int(g.loc[g["own_authority_disabled"], "player_id"].nunique()),
        "affected_rooms": int((rm["own_authority_disabled_players"] > 0).sum()),
        "mean_delta_crps": float(g["delta_personalized_crps"].mean()),
        "mean_delta_ae": float(g["delta_personalized_ae"].mean()),
        "mean_delta_room_error": float(rm["delta_room_error"].mean()),
        "low_support_rows": int(len(low)),
        "low_support_mean_delta_crps": float(low["delta_personalized_crps"].mean()) if len(low) else math.nan,
        "evaluable_players_ge5": int(len(p)),
        "equal_player_mean_improvement": float(vals.mean()) if len(vals) else math.nan,
        "equal_player_median_improvement": float(vals.median()) if len(vals) else math.nan,
        "equal_player_improved_share": float((vals > tol).mean()) if len(vals) else math.nan,
        "equal_player_tied_share": float((vals.abs() <= tol).mean()) if len(vals) else math.nan,
        "equal_player_worsened_share": float((vals < -tol).mean()) if len(vals) else math.nan,
        "equal_player_p10_improvement": float(vals.quantile(0.10)) if len(vals) else math.nan,
        "equal_player_p25_improvement": float(vals.quantile(0.25)) if len(vals) else math.nan,
        "equal_player_p50_improvement": float(vals.quantile(0.50)) if len(vals) else math.nan,
        "equal_player_p75_improvement": float(vals.quantile(0.75)) if len(vals) else math.nan,
        "equal_player_p90_improvement": float(vals.quantile(0.90)) if len(vals) else math.nan,
        "worst_named_player_mean_improvement": float(vals.min()) if len(vals) else math.nan,
        "universal_stage2_equal_player_crps": float(p["universal_stage2_crps"].mean()) if len(p) else math.nan,
        "max_state_weight_gap": float(rm["state_weight_gap"].max()) if len(rm) else math.inf,
        "max_candidate_mass_gap": float(rm["candidate_mass_gap"].max()) if len(rm) else math.inf,
        "max_universal_mass_gap": float(rm["universal_mass_gap"].max()) if len(rm) else math.inf,
        "support_tier_diagnostics": by_tier,
    }
    if len(g):
        rec["player_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_personalized_crps", "player_id", BOOT_SEED + season + CELLS.index(cell) * 100
        )
        rec["game_cluster_bootstrap"] = cluster_bootstrap(
            g, "delta_personalized_crps", "game_key", BOOT_SEED + 10000 + season + CELLS.index(cell) * 100
        )
    else:
        rec["player_cluster_bootstrap"] = {}
        rec["game_cluster_bootstrap"] = {}
    rec["equal_player_bootstrap"] = (
        equal_player_bootstrap(p, BOOT_SEED + 20000 + season + CELLS.index(cell) * 100)
        if len(p) else {}
    )
    return rec, p


def qualify(dev: dict, confirm: dict) -> tuple[dict, bool]:
    tail_floor = (
        -0.25 * float(confirm["universal_stage2_equal_player_crps"])
        if np.isfinite(confirm["universal_stage2_equal_player_crps"])
        else math.nan
    )
    gates = {
        "development_support": bool(all(dev["support_gates"].values())),
        "development_crps_improves": bool(dev["mean_delta_crps"] > 0),
        "development_ae_nonworse": bool(dev["mean_delta_ae"] >= 0),
        "development_room_error_nonworse": bool(dev["mean_delta_room_error"] >= 0),
        "development_conservation": bool(
            dev["max_state_weight_gap"] <= 1e-10 and dev["max_candidate_mass_gap"] <= 1e-10
        ),
        "confirmation_support": bool(all(confirm["support_gates"].values())),
        "confirmation_crps_improves": bool(confirm["mean_delta_crps"] > 0),
        "confirmation_player_cluster_ci_low_gt_zero": bool(
            confirm["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_game_cluster_ci_low_gt_zero": bool(
            confirm["game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_equal_player_mean_improves": bool(confirm["equal_player_mean_improvement"] > 0),
        "confirmation_equal_player_ci_low_gt_zero": bool(
            confirm["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_median_player_nonworse": bool(confirm["equal_player_median_improvement"] >= 0),
        "confirmation_improved_share_gt_worsened_share": bool(
            confirm["equal_player_improved_share"] > confirm["equal_player_worsened_share"]
        ),
        "confirmation_ae_nonworse": bool(confirm["mean_delta_ae"] >= 0),
        "confirmation_room_error_nonworse": bool(confirm["mean_delta_room_error"] >= 0),
        "confirmation_low_support_nonworse": bool(confirm["low_support_mean_delta_crps"] >= 0),
        "confirmation_tail_harm_guard": bool(
            np.isfinite(tail_floor) and confirm["equal_player_p10_improvement"] >= tail_floor
        ),
        "confirmation_conservation": bool(
            confirm["max_state_weight_gap"] <= 1e-10 and confirm["max_candidate_mass_gap"] <= 1e-10
        ),
    }
    gates["tail_floor"] = tail_floor
    return gates, all(v for k, v in gates.items() if k != "tail_floor")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rows", required=True, type=Path)
    ap.add_argument("--signal-rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        parent = read_csv(args.parent_rows, "parent rows")
        signal = read_csv(args.signal_rows, "signal rows")
        z = prepare(parent, signal)

        player_rows: list[dict] = []
        room_rows: list[dict] = []
        for _, g in z.groupby(["season", "week", "team", "cell"], sort=True):
            rows, room = score_room(g)
            player_rows.extend(rows)
            room_rows.append(room)

        rows = pd.DataFrame(player_rows)
        rooms = pd.DataFrame(room_rows)
        if rows.empty or rooms.empty:
            raise RuntimeError("zero scored bridge rows")

        metrics: dict[str, dict] = {}
        player_diag = []
        qualification_gates: dict[str, dict] = {}
        dispositions: dict[str, str] = {}
        for cell in CELLS:
            metrics[cell] = {}
            dev, p24 = score_slice(rows, rooms, cell, 2024)
            confirm, p25 = score_slice(rows, rooms, cell, 2025)
            metrics[cell]["2024"] = dev
            metrics[cell]["2025"] = confirm
            p24["cell"] = cell
            p24["season"] = 2024
            p25["cell"] = cell
            p25["season"] = 2025
            player_diag.extend([p24, p25])
            gates, ok = qualify(dev, confirm)
            qualification_gates[cell] = gates
            dispositions[cell] = (
                f"{cell}_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED"
                if ok else
                f"{cell}_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_CLOSED"
            )

        qualified = sum(v.endswith("_QUALIFIED") for v in dispositions.values())
        overall = (
            "PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_ALL"
            if qualified == len(CELLS)
            else "PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL"
            if qualified
            else "PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED_ALL"
        )

        rows.to_csv(args.out_dir / "rb_stage2_player_specific_authority_bridge_player_rows.csv", index=False)
        rooms.to_csv(args.out_dir / "rb_stage2_player_specific_authority_bridge_room_rows.csv", index=False)
        pd.concat(player_diag, ignore_index=True).to_csv(
            args.out_dir / "rb_stage2_player_specific_authority_bridge_player_diagnostics.csv",
            index=False,
        )
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "parent_run": PARENT_RUN,
            "parent_artifact": PARENT_ARTIFACT,
            "parent_digest": PARENT_DIGEST,
            "signal_run": SIGNAL_RUN,
            "signal_artifact": SIGNAL_ARTIFACT,
            "signal_digest": SIGNAL_DIGEST,
            "cells": list(CELLS),
            "minimum_player_prior": MIN_PLAYER_PRIOR,
            "negative_or_zero_signal_action": "SET_P_EFFECTIVE_TO_ONE",
            "low_support_action": "PRESERVE_UNIVERSAL_STAGE2_P_ACTIVE",
            "metrics": metrics,
            "qualification_gates": qualification_gates,
            "cell_dispositions": dispositions,
            "disposition": overall,
            "projection_values_changed_in_research_score": True,
            "production_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_INTEGRITY_FAILURE",
            "error": str(exc),
            "production_changed": False,
            "coefficients_fit": 0,
            "threshold_searches": 0,
            "outcomes_2026_used": 0,
            "week6_outcomes_used": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
        }
        (args.out_dir / "rb_stage2_player_specific_authority_bridge_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "rb_stage2_player_specific_authority_bridge_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
