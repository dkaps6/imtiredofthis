#!/usr/bin/env python3
"""Score the frozen PRISM RB receiving team-context history-authority reset V1."""
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

VERSION = "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1"
FREEZE_COMMIT = "c4226d642ac5b7ac4f20260949c9eddd726c19b8"
SOURCE_RUN = 39254338930
SOURCE_ARTIFACT = 12187431307
SOURCE_DIGEST = "sha256:7e7b51c75508b4e883db2934c2e892b377a3d284cb61b448e77e9b7c7eca96cd"
AUTHORITY_RUN = 39099474655
AUTHORITY_ARTIFACT = 12125534698
AUTHORITY_DIGEST = "sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4"

SEASONS = (2024, 2025)
BOOT_REPS = 5000
BOOT_SEED = 20261019
TOL = 1e-12
PARITY_TOL = 1e-10


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


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def prepare_source(x: pd.DataFrame) -> pd.DataFrame:
    req = {
        "transition_id", "season", "team", "stable_id", "transition_kind",
        "target_week", "game_key", "baseline_p_effective", "reset_p_active",
        "actionable_joint_history_reset",
    }
    if req - set(x.columns):
        raise RuntimeError(f"source rows missing {sorted(req-set(x.columns))}")
    z = x.copy()
    z["season"] = num(z["season"]).astype("Int64")
    z["target_week"] = num(z["target_week"]).astype("Int64")
    for c in ("transition_id", "team", "stable_id", "transition_kind", "game_key"):
        z[c] = z[c].astype("string").fillna("").str.strip()
    z["baseline_p_effective"] = num(z["baseline_p_effective"])
    z["reset_p_active"] = num(z["reset_p_active"])
    z["actionable_joint_history_reset"] = bool_series(z["actionable_joint_history_reset"])
    z = z.loc[
        z["season"].isin(SEASONS)
        & z["actionable_joint_history_reset"]
    ].copy()
    if z["transition_id"].duplicated().any():
        raise RuntimeError("duplicate actionable transition")
    if not (
        np.isfinite(z["reset_p_active"]).all()
        and z["reset_p_active"].gt(0).all()
        and z["reset_p_active"].lt(1).all()
    ):
        raise RuntimeError("reset_p_active outside open interval")
    return z.sort_values(
        ["season", "target_week", "team", "stable_id"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_players(x: pd.DataFrame) -> pd.DataFrame:
    req = {
        "season", "week", "game_key", "team", "cell", "player_id",
        "authority_q", "p_effective", "y", "n",
        "personalized_expected_count", "personalized_crps", "personalized_ae",
    }
    if req - set(x.columns):
        raise RuntimeError(f"authority player rows missing {sorted(req-set(x.columns))}")
    z = x.copy()
    z["season"] = num(z["season"]).astype("Int64")
    z["week"] = num(z["week"]).astype("Int64")
    for c in ("game_key", "team", "cell", "player_id"):
        z[c] = z[c].astype("string").fillna("").str.strip()
    for c in (
        "authority_q", "p_effective", "y", "n",
        "personalized_expected_count", "personalized_crps", "personalized_ae",
    ):
        z[c] = num(z[c])
    z = z.loc[
        z["season"].isin(SEASONS)
        & z["cell"].eq("RB_RECEIVING")
        & z["player_id"].ne("")
    ].copy()
    keys = ["season", "week", "game_key", "team", "player_id"]
    if z.duplicated(keys).any():
        raise RuntimeError("duplicate authority player identity")
    return z.reset_index(drop=True)


def prepare_rooms(x: pd.DataFrame) -> pd.DataFrame:
    req = {
        "season", "week", "game_key", "team", "cell",
        "personalized_room_error",
    }
    if req - set(x.columns):
        raise RuntimeError(f"authority room rows missing {sorted(req-set(x.columns))}")
    z = x.copy()
    z["season"] = num(z["season"]).astype("Int64")
    z["week"] = num(z["week"]).astype("Int64")
    for c in ("game_key", "team", "cell"):
        z[c] = z[c].astype("string").fillna("").str.strip()
    z["personalized_room_error"] = num(z["personalized_room_error"])
    z = z.loc[
        z["season"].isin(SEASONS)
        & z["cell"].eq("RB_RECEIVING")
    ].copy()
    keys = ["season", "week", "game_key", "team"]
    if z.duplicated(keys).any():
        raise RuntimeError("duplicate authority room identity")
    return z.reset_index(drop=True)


def enumerate_states(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def room_error(expected_named: np.ndarray, expected_comp: float, y: np.ndarray, N: int) -> float:
    observed_comp = float(N - int(y.sum())) / N
    if observed_comp < -TOL:
        raise RuntimeError("observed named opportunity exceeds team total")
    expected_vec = np.append(expected_named, expected_comp)
    observed_vec = np.append(y.astype(float) / N, observed_comp)
    if abs(float(expected_vec.sum()) - 1.0) > TOL:
        raise RuntimeError("expected room mass drift")
    return 0.5 * float(np.abs(expected_vec - observed_vec).sum())


def score_event(
    event: pd.Series,
    players: pd.DataFrame,
    rooms: pd.DataFrame,
) -> dict[str, Any]:
    room = players.loc[
        players["season"].eq(int(event["season"]))
        & players["week"].eq(int(event["target_week"]))
        & players["game_key"].eq(str(event["game_key"]))
        & players["team"].eq(str(event["team"]))
    ].copy()
    if room.empty:
        raise RuntimeError(f"missing authority room for {event['transition_id']}")
    room = room.sort_values("player_id", kind="mergesort").reset_index(drop=True)

    target_idx = np.flatnonzero(
        room["player_id"].astype(str).eq(str(event["stable_id"])).to_numpy()
    )
    if len(target_idx) != 1:
        raise RuntimeError(
            f"target identity count {len(target_idx)} for {event['transition_id']}"
        )
    j = int(target_idx[0])

    Nvals = num(room["n"]).to_numpy(float)
    if not np.isfinite(Nvals).all() or not np.allclose(Nvals, Nvals[0], atol=TOL):
        raise RuntimeError("room N drift")
    if abs(float(Nvals[0]) - round(float(Nvals[0]))) > TOL or Nvals[0] <= 0:
        raise RuntimeError("invalid room N")
    N = int(round(float(Nvals[0])))

    q = num(room["authority_q"]).to_numpy(float)
    pbase = num(room["p_effective"]).to_numpy(float)
    yraw = num(room["y"]).to_numpy(float)
    if not np.allclose(yraw, np.rint(yraw), atol=TOL):
        raise RuntimeError("noninteger outcome count")
    y = np.rint(yraw).astype(int)
    if (y < 0).any() or (y > N).any():
        raise RuntimeError("outcome count outside room total")

    if abs(float(pbase[j]) - float(event["baseline_p_effective"])) > TOL:
        raise RuntimeError("source/baseline p_effective mismatch")

    pcand = pbase.copy()
    pcand[j] = float(event["reset_p_active"])

    wb, rb, rb0 = enumerate_states(q, pbase)
    wc, rc, rc0 = enumerate_states(q, pcand)
    eb = wb @ rb
    ec = wc @ rc
    eb0 = float(wb @ rb0)
    ec0 = float(wc @ rc0)

    baseline_cdf = mixture_binom_cdf(N, rb[:, j], wb)
    candidate_cdf = mixture_binom_cdf(N, rc[:, j], wc)
    yi = int(y[j])
    baseline_crps = discrete_crps(baseline_cdf, yi)
    candidate_crps = discrete_crps(candidate_cdf, yi)
    baseline_mu = N * float(eb[j])
    candidate_mu = N * float(ec[j])
    baseline_ae = abs(yi - baseline_mu)
    candidate_ae = abs(yi - candidate_mu)

    stored = room.iloc[j]
    stored_mu = float(stored["personalized_expected_count"])
    stored_crps = float(stored["personalized_crps"])
    stored_ae = float(stored["personalized_ae"])

    rr = rooms.loc[
        rooms["season"].eq(int(event["season"]))
        & rooms["week"].eq(int(event["target_week"]))
        & rooms["game_key"].eq(str(event["game_key"]))
        & rooms["team"].eq(str(event["team"]))
    ]
    if len(rr) != 1:
        raise RuntimeError(f"authority room summary count {len(rr)}")
    stored_room_error = float(rr.iloc[0]["personalized_room_error"])

    baseline_room_error = room_error(eb, eb0, y, N)
    candidate_room_error = room_error(ec, ec0, y, N)

    parity = {
        "expected_count_abs_diff": abs(baseline_mu - stored_mu),
        "crps_abs_diff": abs(baseline_crps - stored_crps),
        "ae_abs_diff": abs(baseline_ae - stored_ae),
        "room_error_abs_diff": abs(baseline_room_error - stored_room_error),
    }

    return {
        "transition_id": str(event["transition_id"]),
        "season": int(event["season"]),
        "transition_kind": str(event["transition_kind"]),
        "target_week": int(event["target_week"]),
        "game_key": str(event["game_key"]),
        "team": str(event["team"]),
        "stable_id": str(event["stable_id"]),
        "baseline_p_effective": float(pbase[j]),
        "candidate_reset_p_active": float(pcand[j]),
        "y": yi,
        "N": N,
        "baseline_expected_count": baseline_mu,
        "candidate_expected_count": candidate_mu,
        "baseline_crps": baseline_crps,
        "candidate_crps": candidate_crps,
        "delta_crps": baseline_crps - candidate_crps,
        "baseline_ae": baseline_ae,
        "candidate_ae": candidate_ae,
        "delta_ae": baseline_ae - candidate_ae,
        "baseline_room_error": baseline_room_error,
        "candidate_room_error": candidate_room_error,
        "delta_room_error": baseline_room_error - candidate_room_error,
        "baseline_state_weight_gap": abs(float(wb.sum()) - 1.0),
        "candidate_state_weight_gap": abs(float(wc.sum()) - 1.0),
        "baseline_mass_gap": abs(float(eb.sum()) + eb0 - 1.0),
        "candidate_mass_gap": abs(float(ec.sum()) + ec0 - 1.0),
        "baseline_expected_count_abs_diff": parity["expected_count_abs_diff"],
        "baseline_crps_abs_diff": parity["crps_abs_diff"],
        "baseline_ae_abs_diff": parity["ae_abs_diff"],
        "baseline_room_error_abs_diff": parity["room_error_abs_diff"],
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str,
    seed: int,
) -> dict[str, Any]:
    z = df[[value_col, cluster_col]].dropna()
    agg = z.groupby(cluster_col, sort=True)[value_col].agg(["sum", "count"])
    if len(agg) < 2:
        return {
            "clusters": int(len(agg)),
            "valid_reps": 0,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
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
    for pid, x in g.groupby("stable_id", sort=True):
        rows.append({
            "stable_id": str(pid),
            "events": int(len(x)),
            "mean_delta_crps": float(x["delta_crps"].mean()),
            "mean_baseline_crps": float(x["baseline_crps"].mean()),
            "mean_delta_ae": float(x["delta_ae"].mean()),
        })
    return pd.DataFrame(rows)


def equal_player_bootstrap(p: pd.DataFrame, seed: int) -> dict[str, Any]:
    vals = num(p["mean_delta_crps"]).dropna().to_numpy(float)
    if len(vals) < 2:
        return {
            "players": int(len(vals)),
            "valid_reps": 0,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    out = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        idx = rng.integers(0, len(vals), size=len(vals))
        out[i] = float(vals[idx].mean())
    return {
        "players": int(len(vals)),
        "valid_reps": BOOT_REPS,
        "ci95_low": float(np.quantile(out, 0.025)),
        "ci95_high": float(np.quantile(out, 0.975)),
    }


def score_season(rows: pd.DataFrame, season: int) -> tuple[dict[str, Any], pd.DataFrame]:
    g = rows.loc[rows["season"].eq(season)].copy()
    p = equal_player_frame(g)
    vals = num(p["mean_delta_crps"]).dropna()

    rec = {
        "season": season,
        "actionable_events": int(len(g)),
        "stable_ids": int(g["stable_id"].nunique()),
        "games": int(g["game_key"].nunique()),
        "mean_delta_crps": float(g["delta_crps"].mean()) if len(g) else math.nan,
        "mean_delta_ae": float(g["delta_ae"].mean()) if len(g) else math.nan,
        "mean_delta_room_error": float(g["delta_room_error"].mean()) if len(g) else math.nan,
        "equal_player_mean_improvement": float(vals.mean()) if len(vals) else math.nan,
        "equal_player_median_improvement": float(vals.median()) if len(vals) else math.nan,
        "equal_player_p10_improvement": float(vals.quantile(0.10)) if len(vals) else math.nan,
        "equal_player_improved_share": float((vals > TOL).mean()) if len(vals) else math.nan,
        "equal_player_tied_share": float((vals.abs() <= TOL).mean()) if len(vals) else math.nan,
        "equal_player_worsened_share": float((vals < -TOL).mean()) if len(vals) else math.nan,
        "baseline_equal_player_crps": float(p["mean_baseline_crps"].mean()) if len(p) else math.nan,
        "max_baseline_expected_count_parity_diff": float(g["baseline_expected_count_abs_diff"].max()) if len(g) else math.nan,
        "max_baseline_crps_parity_diff": float(g["baseline_crps_abs_diff"].max()) if len(g) else math.nan,
        "max_baseline_ae_parity_diff": float(g["baseline_ae_abs_diff"].max()) if len(g) else math.nan,
        "max_baseline_room_error_parity_diff": float(g["baseline_room_error_abs_diff"].max()) if len(g) else math.nan,
        "max_baseline_state_weight_gap": float(g["baseline_state_weight_gap"].max()) if len(g) else math.nan,
        "max_candidate_state_weight_gap": float(g["candidate_state_weight_gap"].max()) if len(g) else math.nan,
        "max_baseline_mass_gap": float(g["baseline_mass_gap"].max()) if len(g) else math.nan,
        "max_candidate_mass_gap": float(g["candidate_mass_gap"].max()) if len(g) else math.nan,
        "transition_kind_diagnostics": {
            str(k): {
                "events": int(len(x)),
                "stable_ids": int(x["stable_id"].nunique()),
                "mean_delta_crps": float(x["delta_crps"].mean()),
            }
            for k, x in g.groupby("transition_kind", sort=True)
        },
    }
    rec["player_cluster_bootstrap"] = cluster_bootstrap(
        g, "delta_crps", "stable_id", BOOT_SEED + season
    )
    rec["game_cluster_bootstrap"] = cluster_bootstrap(
        g, "delta_crps", "game_key", BOOT_SEED + 10000 + season
    )
    rec["equal_player_bootstrap"] = equal_player_bootstrap(
        p, BOOT_SEED + 20000 + season
    )
    return rec, p


def baseline_parity_ok(s: dict[str, Any]) -> bool:
    return bool(
        s["max_baseline_expected_count_parity_diff"] <= PARITY_TOL
        and s["max_baseline_crps_parity_diff"] <= PARITY_TOL
        and s["max_baseline_ae_parity_diff"] <= PARITY_TOL
        and s["max_baseline_room_error_parity_diff"] <= PARITY_TOL
    )


def conservation_ok(s: dict[str, Any]) -> bool:
    return bool(
        s["max_baseline_state_weight_gap"] <= PARITY_TOL
        and s["max_candidate_state_weight_gap"] <= PARITY_TOL
        and s["max_baseline_mass_gap"] <= PARITY_TOL
        and s["max_candidate_mass_gap"] <= PARITY_TOL
    )


def qualify(dev: dict[str, Any], confirm: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    tail_floor = (
        -0.25 * float(confirm["baseline_equal_player_crps"])
        if np.isfinite(confirm["baseline_equal_player_crps"])
        else math.nan
    )
    gates: dict[str, Any] = {
        "development_actionable_events_ge_20": dev["actionable_events"] >= 20,
        "development_distinct_ids_ge_20": dev["stable_ids"] >= 20,
        "development_baseline_parity": baseline_parity_ok(dev),
        "development_mean_crps_gain_positive": dev["mean_delta_crps"] > 0,
        "development_mean_ae_gain_nonnegative": dev["mean_delta_ae"] >= 0,
        "development_equal_player_mean_positive": dev["equal_player_mean_improvement"] > 0,
        "development_median_player_nonnegative": dev["equal_player_median_improvement"] >= 0,
        "development_improved_share_gt_worsened": (
            dev["equal_player_improved_share"] > dev["equal_player_worsened_share"]
        ),
        "development_room_error_gain_nonnegative": dev["mean_delta_room_error"] >= 0,
        "development_conservation": conservation_ok(dev),
        "confirmation_actionable_events_ge_15": confirm["actionable_events"] >= 15,
        "confirmation_distinct_ids_ge_15": confirm["stable_ids"] >= 15,
        "confirmation_baseline_parity": baseline_parity_ok(confirm),
        "confirmation_mean_crps_gain_positive": confirm["mean_delta_crps"] > 0,
        "confirmation_player_cluster_ci_low_gt_zero": (
            confirm["player_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_game_cluster_ci_low_gt_zero": (
            confirm["game_cluster_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_equal_player_mean_positive": confirm["equal_player_mean_improvement"] > 0,
        "confirmation_equal_player_ci_low_gt_zero": (
            confirm["equal_player_bootstrap"].get("ci95_low", math.nan) > 0
        ),
        "confirmation_median_player_nonnegative": confirm["equal_player_median_improvement"] >= 0,
        "confirmation_improved_share_gt_worsened": (
            confirm["equal_player_improved_share"] > confirm["equal_player_worsened_share"]
        ),
        "confirmation_mean_ae_gain_nonnegative": confirm["mean_delta_ae"] >= 0,
        "confirmation_room_error_gain_nonnegative": confirm["mean_delta_room_error"] >= 0,
        "confirmation_tail_harm_guard": (
            np.isfinite(tail_floor)
            and confirm["equal_player_p10_improvement"] >= tail_floor
        ),
        "confirmation_conservation": conservation_ok(confirm),
    }
    gates["tail_floor"] = tail_floor
    return gates, bool(all(v for k, v in gates.items() if k != "tail_floor"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-rows", type=Path, required=True)
    ap.add_argument("--authority-player-rows", type=Path, required=True)
    ap.add_argument("--authority-room-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = prepare_source(read_csv(args.source_rows, "source rows"))
        players = prepare_players(read_csv(args.authority_player_rows, "authority player rows"))
        rooms = prepare_rooms(read_csv(args.authority_room_rows, "authority room rows"))

        if len(source.loc[source["season"].eq(2024)]) != 22:
            raise RuntimeError("unexpected 2024 actionable source cohort")
        if len(source.loc[source["season"].eq(2025)]) != 17:
            raise RuntimeError("unexpected 2025 actionable source cohort")

        scored = pd.DataFrame([
            score_event(r, players, rooms)
            for _, r in source.iterrows()
        ])
        if len(scored) != len(source):
            raise RuntimeError("scored/source row count mismatch")

        dev, p24 = score_season(scored, 2024)
        confirm, p25 = score_season(scored, 2025)

        if not baseline_parity_ok(dev) or not baseline_parity_ok(confirm):
            disposition = "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_INTEGRITY_FAILURE"
            gates: dict[str, Any] = {
                "development_baseline_parity": baseline_parity_ok(dev),
                "confirmation_baseline_parity": baseline_parity_ok(confirm),
            }
        else:
            gates, qualified = qualify(dev, confirm)
            disposition = (
                "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_QUALIFIED"
                if qualified else
                "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_PREDICTIVE_CLOSED"
            )

        pdiag = pd.concat([
            p24.assign(season=2024),
            p25.assign(season=2025),
        ], ignore_index=True)

        scored.to_csv(
            args.out_dir / "rb_receiving_team_context_history_authority_reset_scored_rows.csv",
            index=False,
        )
        pdiag.to_csv(
            args.out_dir / "rb_receiving_team_context_history_authority_reset_player_diagnostics.csv",
            index=False,
        )

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "source_parent": {
                "run": SOURCE_RUN,
                "artifact": SOURCE_ARTIFACT,
                "digest": SOURCE_DIGEST,
            },
            "qualified_authority_parent": {
                "run": AUTHORITY_RUN,
                "artifact": AUTHORITY_ARTIFACT,
                "digest": AUTHORITY_DIGEST,
            },
            "baseline": "CURRENT_QUALIFIED_RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY",
            "candidate": "RESET_TRANSITIONED_TARGET_OLD_HISTORY_AUTHORITIES_TO_STAGE1_ROLE_PRIOR_ON_FIRST_CANONICAL_POST_CHANGE_ROW",
            "development_2024": dev,
            "confirmation_2025": confirm,
            "qualification_gates": gates,
            "disposition": disposition,
            "bootstrap_reps": BOOT_REPS,
            "bootstrap_seed": BOOT_SEED,
            "coefficients_fit": 0,
            "thresholds_searched": 0,
            "subgroup_rescues": 0,
            "rushing_yards_scored": False,
            "receiving_yards_scored": False,
            "touchdowns_scored": False,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        }
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_INTEGRITY_FAILURE",
            "error": str(exc),
            "coefficients_fit": 0,
            "thresholds_searched": 0,
            "subgroup_rescues": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "production_changed": False,
        }
        (args.out_dir / "rb_receiving_team_context_history_authority_reset_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "rb_receiving_team_context_history_authority_reset_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
