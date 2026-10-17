#!/usr/bin/env python3
"""Score the frozen PRISM joint RB+TE receiving active-role allocation V1.

The mechanism is frozen in:
docs/research/PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_V1_FREEZE.md

Candidate construction uses only preserved pregame q_i and qualified Stage-1
p_active. Target-game counts are read only after the complete candidate state
has been materialized. RB and TE are never allowed to rescue one another.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from scripts._opponent_map import canon_team
from scripts.research.audit_player_active_role_conditional_allocation_source_v1 import prepare_stage1
from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    cluster_bootstrap,
    discrete_crps,
    enumerate_active_states,
    equal_player_bootstrap,
    prepare_outcomes,
    prepare_schedule,
)

VERSION = "PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_V1"
FREEZE_VERSION = "PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_V1_FREEZE"
FREEZE_COMMIT = "73fc9e92aa3848f289a11fdc2276391e24d449be"
SOURCE_RUN = 38955368643
SOURCE_ARTIFACT = 12070806828
SOURCE_DIGEST = "sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83"
STAGE1_RUN = 38907459996
STAGE1_ARTIFACT = 12049120604
SEPARATE_STAGE2_RUN = 38957575645
SEPARATE_STAGE2_ARTIFACT = 12070774378

SEASONS = (2024, 2025)
WEEKS = tuple(range(5, 19))
FAMILIES = ("RB", "TE")
CELLS = ("RB_RECEIVING", "TE_RECEIVING")
TOL = 1e-12
BOOT_REPS = 5000
BOOT_SEED = 20261017
SUPPORT = {
    "RB": {"rows": 1200, "players": 100, "rooms": 400, "eval_players": 75},
    "TE": {"rows": 1200, "players": 100, "rooms": 400, "eval_players": 75},
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
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


def boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def candidate_source(
    source: pd.DataFrame,
    stage1_universe: pd.DataFrame,
    stage1_mechanism: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], dict], pd.DataFrame]:
    """Construct the complete joint candidate before any target-game count read."""
    _, pactive = prepare_stage1(stage1_universe, stage1_mechanism)

    s = source.loc[
        num(source["season"]).isin(SEASONS)
        & num(source["week"]).isin(WEEKS)
        & source["cell"].astype(str).isin(CELLS)
    ].copy()
    s["season"] = num(s["season"]).astype(int)
    s["week"] = num(s["week"]).astype(int)
    s["team"] = s["team"].map(canon_team)
    s["family"] = s["family"].astype(str).str.upper().str.strip()
    s["authority_q"] = num(s["authority_q"])
    s["authority_covered"] = boolish(s["authority_covered"])
    s["denominator_finite"] = boolish(s["denominator_finite"])
    s["ambiguous_identity"] = boolish(s["ambiguous_identity"])

    # Match the parent source-qualified population exactly: uncovered q stays in
    # the shared complement and is never imputed.
    s = s.loc[
        s["authority_covered"]
        & s["denominator_finite"]
        & ~s["ambiguous_identity"]
        & s["authority_q"].notna()
    ].copy()
    if s.empty:
        raise RuntimeError("joint source population empty")
    if s.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate joint source player identity")

    p = pactive.copy()
    p["season"] = num(p["season"]).astype(int)
    p["week"] = num(p["week"]).astype(int)
    p["team"] = p["team"].map(canon_team)
    p["family"] = p["family"].astype(str).str.upper().str.strip()
    keep = [
        "season", "week", "team", "family", "player_id",
        "stage1_p_active", "stage1_named_player_supported",
    ]
    s = s.merge(
        p[keep],
        on=["season", "week", "team", "family", "player_id"],
        how="left",
        validate="one_to_one",
    )
    pa = num(s["stage1_p_active"])
    if pa.isna().any() or not np.isfinite(pa).all() or not ((pa > 0) & (pa < 1)).all():
        raise RuntimeError(
            f"joint Stage-1 p_active attach invalid rows={int((~((pa>0)&(pa<1)&np.isfinite(pa))).sum())}"
        )
    s["stage1_p_active"] = pa
    s["stage1_named_player_supported"] = boolish(s["stage1_named_player_supported"])
    if (s["authority_q"] < -TOL).any() or not np.isfinite(s["authority_q"]).all():
        raise RuntimeError("invalid joint authority q")
    s.loc[s["authority_q"].abs().le(TOL), "authority_q"] = 0.0

    candidate_rows: list[dict] = []
    room_meta: list[dict] = []
    states: dict[tuple[int, int, str], dict] = {}

    for (season, week, team), g0 in s.groupby(["season", "week", "team"], sort=True):
        g = g0.sort_values(["family", "player_id"]).reset_index(drop=True).copy()
        q_all = num(g["authority_q"]).to_numpy(float)
        pos = q_all > TOL
        pos_idx = np.flatnonzero(pos)
        q = q_all[pos]
        pvec = num(g.loc[pos, "stage1_p_active"]).to_numpy(float)
        q0 = 1.0 - float(q.sum())
        if q0 < -TOL:
            raise RuntimeError(f"joint canonical mass exceeds one {season} W{week} {team}: q0={q0}")
        if abs(q0) <= TOL:
            q0 = 0.0
        if len(q):
            weights, state_named, state_comp = enumerate_active_states(q, pvec)
            expected = weights @ state_named
            expected_comp = float(weights @ state_comp)
        else:
            weights = np.asarray([1.0])
            state_named = np.zeros((1, 0), float)
            state_comp = np.asarray([1.0])
            expected = np.zeros(0, float)
            expected_comp = 1.0
        if abs(float(weights.sum()) - 1.0) > TOL:
            raise RuntimeError("joint state weight drift")
        if abs(float(expected.sum()) + expected_comp - 1.0) > TOL:
            raise RuntimeError("joint expected mass drift")

        exp_map = {int(idx): float(expected[j]) for j, idx in enumerate(pos_idx)}
        state_pos_map = {int(idx): j for j, idx in enumerate(pos_idx)}
        key = (int(season), int(week), str(team))
        states[key] = {
            "weights": weights,
            "state_named": state_named,
            "state_comp": state_comp,
            "positive_position": state_pos_map,
            "q0": q0,
        }

        room_meta.append({
            "season": int(season),
            "week": int(week),
            "team": str(team),
            "source_rows": int(len(g)),
            "positive_q_players": int(pos.sum()),
            "rb_rows": int(g["family"].eq("RB").sum()),
            "te_rows": int(g["family"].eq("TE").sum()),
            "canonical_joint_named_mass": float(q.sum()),
            "canonical_complement_mass": float(q0),
            "candidate_expected_joint_named_mass": float(expected.sum()),
            "candidate_expected_complement_mass": float(expected_comp),
            "state_count": int(len(weights)),
            "state_weight_gap": float(abs(float(weights.sum()) - 1.0)),
            "candidate_mass_gap": float(abs(float(expected.sum()) + expected_comp - 1.0)),
        })

        for i, r in g.iterrows():
            qi = float(q_all[i])
            candidate_rows.append({
                "season": int(season),
                "week": int(week),
                "team": str(team),
                "cell": str(r["cell"]),
                "family": str(r["family"]),
                "player_id": str(r["player_id"]),
                "player_name_key": str(r["player_name_key"]),
                "position": str(r["position"]),
                "stage1_p_active": float(r["stage1_p_active"]),
                "stage1_named_player_supported": bool(r["stage1_named_player_supported"]),
                "authority_q": qi,
                "canonical_zero_share": bool(qi <= TOL),
                "candidate_expected_q": float(exp_map.get(i, qi)),
                "candidate_expected_shift": float(exp_map.get(i, qi) - qi),
                "room_state_count": int(len(weights)),
                "candidate_construction_outcomes_read": 0,
            })

    cand = pd.DataFrame(candidate_rows)
    rooms = pd.DataFrame(room_meta)
    if len(rooms) != 832:
        raise RuntimeError(f"expected 832 joint source rooms, got {len(rooms)}")
    if int(rooms["positive_q_players"].max()) > 9:
        raise RuntimeError("joint active-set size exceeded frozen preflight max 9")
    if float(rooms["canonical_joint_named_mass"].max()) > 1.0 + TOL:
        raise RuntimeError("joint source q exceeds finite target pool")
    return cand, states, rooms


def mixture_binom_cdf(N: int, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ks = np.arange(int(N) + 1)
    out = np.zeros(len(ks), float)
    for pr, wt in zip(np.asarray(probs, float), np.asarray(weights, float)):
        out += float(wt) * binom(int(N), float(pr)).cdf(ks)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite joint mixture CDF")
    return out


def equal_player_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in rows.groupby("player_id", sort=True):
        if len(g) < 5:
            continue
        out.append({
            "player_id": str(pid),
            "family": str(g["family"].iloc[0]),
            "season": int(g["season"].iloc[0]),
            "rows": int(len(g)),
            "mean_delta_crps": float(g["delta_crps"].mean()),
            "mean_delta_ae": float(g["delta_ae"].mean()),
        })
    return pd.DataFrame(out)


def support_gates(family: str, rows: pd.DataFrame, rooms: pd.DataFrame, players: pd.DataFrame) -> dict:
    req = SUPPORT[family]
    family_rooms = rows[["season", "week", "team"]].drop_duplicates()
    return {
        "rows": int(len(rows)) >= req["rows"],
        "players": int(rows["player_id"].nunique()) >= req["players"],
        "team_rooms": int(len(family_rooms)) >= req["rooms"],
        "evaluable_players": int(len(players)) >= req["eval_players"],
        "joint_rooms_present": int(len(rooms)) >= req["rooms"],
    }


def score(
    candidate: pd.DataFrame,
    states: dict,
    pre_rooms: pd.DataFrame,
    player_logs_path: Path,
    schedule_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Outcome authority is opened only here, after candidate is complete.
    schedule = prepare_schedule(read_csv(schedule_path, "historical schedule"))
    player_outcomes, team_outcomes = prepare_outcomes(
        read_csv(player_logs_path, "historical player logs"),
        schedule,
    )
    po = player_outcomes.copy()
    po["targets"] = num(po["targets"]).fillna(0.0)
    team = team_outcomes[
        ["season", "week", "team", "game_key", "team_targets"]
    ].copy()
    team["team_targets"] = num(team["team_targets"])

    x = candidate.merge(
        team,
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if x["team_targets"].isna().any() or x["game_key"].isna().any():
        raise RuntimeError("joint candidate missing target denominator/game key")
    x = x.merge(
        po[["season", "week", "team", "player_id", "targets"]],
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="one_to_one",
    )
    x["targets"] = num(x["targets"]).fillna(0.0)
    if not np.allclose(x["targets"], np.rint(x["targets"]), atol=TOL):
        raise RuntimeError("non-integer player target count")
    x["y"] = np.rint(x["targets"]).astype(int)
    x["N"] = np.rint(num(x["team_targets"])).astype(int)
    if (x["N"] <= 0).any() or (x["y"] < 0).any() or (x["y"] > x["N"]).any():
        raise RuntimeError("invalid finite target outcome count")

    baseline_crps = []
    candidate_crps = []
    for r in x.itertuples(index=False):
        key = (int(r.season), int(r.week), str(r.team))
        st = states[key]
        qi = float(r.authority_q)
        yi = int(r.y)
        N = int(r.N)
        ks = np.arange(N + 1)
        base_cdf = binom(N, min(max(qi, 0.0), 1.0)).cdf(ks)
        baseline_crps.append(discrete_crps(base_cdf, yi))
        # Candidate is unchanged for canonical-zero rows.
        room_rows = candidate.loc[
            candidate["season"].eq(r.season)
            & candidate["week"].eq(r.week)
            & candidate["team"].eq(r.team)
        ].sort_values(["family", "player_id"]).reset_index(drop=True)
        idx = room_rows.index[
            room_rows["family"].eq(r.family)
            & room_rows["player_id"].astype(str).eq(str(r.player_id))
        ]
        if len(idx) != 1:
            raise RuntimeError("joint candidate room row identity drift")
        local_idx = int(idx[0])
        if local_idx in st["positive_position"]:
            j = st["positive_position"][local_idx]
            cdf = mixture_binom_cdf(
                N, st["state_named"][:, j], st["weights"]
            )
        else:
            cdf = base_cdf
        candidate_crps.append(discrete_crps(cdf, yi))

    x["baseline_crps"] = baseline_crps
    x["candidate_crps"] = candidate_crps
    x["delta_crps"] = x["baseline_crps"] - x["candidate_crps"]
    x["baseline_expected_count"] = x["N"] * x["authority_q"]
    x["candidate_expected_count"] = x["N"] * x["candidate_expected_q"]
    x["baseline_ae"] = (x["y"] - x["baseline_expected_count"]).abs()
    x["candidate_ae"] = (x["y"] - x["candidate_expected_count"]).abs()
    x["delta_ae"] = x["baseline_ae"] - x["candidate_ae"]
    x["outcomes_used_as_predictors"] = 0

    room_rows = []
    for (season, week, team), g in x.groupby(["season", "week", "team"], sort=True):
        key = (int(season), int(week), str(team))
        st = states[key]
        Nvals = g["N"].unique()
        gamevals = g["game_key"].astype(str).unique()
        if len(Nvals) != 1 or len(gamevals) != 1:
            raise RuntimeError("joint room target denominator/game drift")
        N = int(Nvals[0])
        q = num(g["authority_q"]).to_numpy(float)
        cq = num(g["candidate_expected_q"]).to_numpy(float)
        y = num(g["y"]).to_numpy(float)
        q0 = float(st["q0"])
        cy0 = float(st["weights"] @ st["state_comp"])
        y0 = int(N - int(np.rint(y).sum()))
        if y0 < 0:
            raise RuntimeError(f"joint explicit targets exceed team total {season} W{week} {team}")
        obs = np.append(y / N, y0 / N)
        base = np.append(q, q0)
        cand = np.append(cq, cy0)
        if abs(float(base.sum()) - 1.0) > TOL or abs(float(cand.sum()) - 1.0) > TOL:
            raise RuntimeError("joint room probability conservation drift")
        be = 0.5 * float(np.abs(base - obs).sum())
        ce = 0.5 * float(np.abs(cand - obs).sum())
        room_rows.append({
            "season": int(season),
            "week": int(week),
            "game_key": str(gamevals[0]),
            "team": str(team),
            "N": N,
            "explicit_rb_te_targets": int(np.rint(y).sum()),
            "complement_targets": y0,
            "baseline_room_error": be,
            "candidate_room_error": ce,
            "delta_room_error": be - ce,
            "positive_q_players": int(pre_rooms.loc[
                pre_rooms["season"].eq(season)
                & pre_rooms["week"].eq(week)
                & pre_rooms["team"].eq(team),
                "positive_q_players",
            ].iloc[0]),
            "state_count": int(len(st["weights"])),
        })
    room_score = pd.DataFrame(room_rows)

    family_metrics = []
    family_players = []
    family_gates: dict[str, dict[str, bool]] = {}
    for season in SEASONS:
        for fam in FAMILIES:
            r = x.loc[x["season"].eq(season) & x["family"].eq(fam)].copy()
            rm = room_score.loc[room_score["season"].eq(season)].copy()
            ep = equal_player_frame(r)
            support = support_gates(fam, r, rm, ep)
            pcb = cluster_bootstrap(r, "delta_crps", "player_id", BOOT_SEED + season + (0 if fam=="RB" else 100))
            gcb = cluster_bootstrap(r, "delta_crps", "game_key", BOOT_SEED + season + (10 if fam=="RB" else 110))
            epb = equal_player_bootstrap(ep, BOOT_SEED + season + (20 if fam=="RB" else 120))
            mean_crps = float(r["delta_crps"].mean())
            mean_ae = float(r["delta_ae"].mean())
            ep_mean = float(ep["mean_delta_crps"].mean()) if len(ep) else math.nan
            rec = {
                "season": season,
                "family": fam,
                "rows": int(len(r)),
                "players": int(r["player_id"].nunique()),
                "team_rooms": int(r[["season","week","team"]].drop_duplicates().shape[0]),
                "evaluable_players_ge5": int(len(ep)),
                "mean_delta_crps": mean_crps,
                "mean_delta_ae": mean_ae,
                "equal_player_mean_delta_crps": ep_mean,
                "player_cluster_ci95_low": float(pcb["ci95_low"]),
                "player_cluster_ci95_high": float(pcb["ci95_high"]),
                "game_cluster_ci95_low": float(gcb["ci95_low"]),
                "game_cluster_ci95_high": float(gcb["ci95_high"]),
                "equal_player_ci95_low": float(epb["ci95_low"]),
                "equal_player_ci95_high": float(epb["ci95_high"]),
                **{f"support_{k}": bool(v) for k,v in support.items()},
            }
            if season == 2024:
                gates = {
                    "support_all": all(support.values()),
                    "mean_crps_positive": mean_crps > 0,
                    "mean_ae_nonworse": mean_ae >= -TOL,
                }
            else:
                gates = {
                    "support_all": all(support.values()),
                    "mean_crps_positive": mean_crps > 0,
                    "player_cluster_ci_low_positive": float(pcb["ci95_low"]) > 0,
                    "game_cluster_ci_low_positive": float(gcb["ci95_low"]) > 0,
                    "equal_player_mean_positive": ep_mean > 0,
                    "equal_player_ci_low_positive": float(epb["ci95_low"]) > 0,
                    "mean_ae_nonworse": mean_ae >= -TOL,
                }
            rec.update({f"gate_{k}": bool(v) for k,v in gates.items()})
            family_metrics.append(rec)
            ep["season"] = season
            ep["family"] = fam
            family_players.append(ep)
            family_gates[f"{season}_{fam}"] = gates

    fam_metrics = pd.DataFrame(family_metrics)
    player_diag = pd.concat(family_players, ignore_index=True)

    room_metrics = {}
    room_pass = True
    for season in SEASONS:
        rr = room_score.loc[room_score["season"].eq(season)]
        delta = float(rr["delta_room_error"].mean())
        room_metrics[str(season)] = {
            "rooms": int(len(rr)),
            "mean_delta_room_error": delta,
            "baseline_mean_room_error": float(rr["baseline_room_error"].mean()),
            "candidate_mean_room_error": float(rr["candidate_room_error"].mean()),
            "gate_nonworse": bool(delta >= -TOL),
        }
        room_pass = room_pass and delta >= -TOL

    families_pass = all(all(g.values()) for g in family_gates.values())
    qualified = bool(families_pass and room_pass)

    result = {
        "version": VERSION,
        "freeze_version": FREEZE_VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "source_digest": SOURCE_DIGEST,
        "stage1_run": STAGE1_RUN,
        "stage1_artifact": STAGE1_ARTIFACT,
        "separate_stage2_run": SEPARATE_STAGE2_RUN,
        "separate_stage2_artifact": SEPARATE_STAGE2_ARTIFACT,
        "disposition": (
            "PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_QUALIFIED"
            if qualified
            else "PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_CLOSED"
        ),
        "joint_source_rooms": int(len(pre_rooms)),
        "joint_source_rows": int(len(candidate)),
        "max_positive_q_players": int(pre_rooms["positive_q_players"].max()),
        "max_state_count": int(pre_rooms["state_count"].max()),
        "joint_named_mass_mean": float(pre_rooms["canonical_joint_named_mass"].mean()),
        "joint_named_mass_max": float(pre_rooms["canonical_joint_named_mass"].max()),
        "family_gates": family_gates,
        "room_metrics": room_metrics,
        "rb_receiving_joint_qualified": bool(
            all(all(family_gates[f"{s}_RB"].values()) for s in SEASONS) and room_pass
        ),
        "te_receiving_joint_qualified": bool(
            all(all(family_gates[f"{s}_TE"].values()) for s in SEASONS) and room_pass
        ),
        "pooled_family_rescue_allowed": False,
        "wr_stage2_scored": False,
        "wr_remained_in_shared_complement": True,
        "canonical_q_refit": 0,
        "p_active_retuned": 0,
        "new_share_coefficients_fit": 0,
        "threshold_searches": 0,
        "target_game_outcomes_used_as_predictors": 0,
        "week4_5_forensic_outcomes_used_to_define_law": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
        "week6_locks_changed": False,
    }
    return x, room_score, fam_metrics, player_diag, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-detail", type=Path, required=True)
    ap.add_argument("--stage1-universe", type=Path, required=True)
    ap.add_argument("--stage1-mechanism", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate, states, pre_rooms = candidate_source(
        read_csv(args.source_detail, "Stage-2 source detail"),
        read_csv(args.stage1_universe, "Stage-1 universe"),
        read_csv(args.stage1_mechanism, "Stage-1 mechanism"),
    )
    # Materialize the outcome-blind candidate before opening target-game logs.
    candidate.to_csv(
        args.out_dir / "joint_rb_te_receiving_candidate_preoutcome.csv",
        index=False,
        float_format="%.12f",
    )
    pre_rooms.to_csv(
        args.out_dir / "joint_rb_te_receiving_rooms_preoutcome.csv",
        index=False,
        float_format="%.12f",
    )

    rows, rooms, metrics, players, result = score(
        candidate, states, pre_rooms, args.player_logs, args.schedule
    )
    rows.to_csv(
        args.out_dir / "joint_rb_te_receiving_scored_rows.csv",
        index=False,
        float_format="%.12f",
    )
    rooms.to_csv(
        args.out_dir / "joint_rb_te_receiving_scored_rooms.csv",
        index=False,
        float_format="%.12f",
    )
    metrics.to_csv(
        args.out_dir / "joint_rb_te_receiving_family_metrics.csv",
        index=False,
        float_format="%.12f",
    )
    players.to_csv(
        args.out_dir / "joint_rb_te_receiving_player_diagnostics.csv",
        index=False,
        float_format="%.12f",
    )
    (args.out_dir / "joint_rb_te_receiving_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
