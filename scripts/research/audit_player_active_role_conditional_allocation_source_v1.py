#!/usr/bin/env python3
"""Source audit for the Stage-1 active-role -> conditional-allocation bridge V1.

Frozen parent: docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md

This audit does not score a bridge outcome. It asks whether 2024/2025 historical
pregame rows have a leakage-safe canonical conditional-opportunity authority that
can later consume the already-qualified Stage-1 active-role probability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.research.score_joint_opportunity_allocation_dirichlet_v1 import build_week_metrics
from scripts.simulation_v2 import _top_n_shares
from scripts.utils.canonical_names import canonicalize_player_name_safe

VERSION = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_AUDIT"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
CELLS = ("QB_PASSING", "RB_RUSHING", "RB_RECEIVING", "WR_RECEIVING", "TE_RECEIVING")
FAMILY_FOR_CELL = {
    "QB_PASSING": "QB", "RB_RUSHING": "RB", "RB_RECEIVING": "RB",
    "WR_RECEIVING": "WR", "TE_RECEIVING": "TE",
}
SUPPORT = {
    "QB_PASSING": (350, 45), "RB_RUSHING": (800, 80), "RB_RECEIVING": (800, 80),
    "WR_RECEIVING": (1200, 140), "TE_RECEIVING": (700, 75),
}
WR_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
TOL = 1e-10


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def key(value: Any) -> str:
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


def prepare_stage1(universe: pd.DataFrame, mechanism: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ureq = {
        "season", "week", "team", "player_id", "position", "status",
        "player_name_key", "family", "known_official_inactive",
        "stable_roster_identity", "chronology_violations",
    }
    mreq = {"season", "week", "team", "family", "player_id", "player_name_key", "p_candidate"}
    if ureq - set(universe.columns):
        raise RuntimeError(f"Stage-1 universe missing {sorted(ureq-set(universe.columns))}")
    if mreq - set(mechanism.columns):
        raise RuntimeError(f"Stage-1 mechanism missing {sorted(mreq-set(mechanism.columns))}")

    u = universe.loc[
        num(universe["season"]).isin(TARGET_SEASONS)
        & num(universe["week"]).isin(TARGET_WEEKS)
    ].copy()
    u["season"] = num(u["season"]).astype(int)
    u["week"] = num(u["week"]).astype(int)
    u["team"] = u["team"].map(canon_team)
    u["player_name_key"] = u["player_name_key"].map(key)
    u = u.loc[
        u["status"].astype(str).str.upper().eq("ACT")
        & num(u["known_official_inactive"]).fillna(0).eq(0)
        & num(u["stable_roster_identity"]).fillna(0).eq(1)
    ].copy()
    if u.duplicated(["season", "week", "team", "player_id", "family"]).any():
        raise RuntimeError("duplicate Stage-1 pregame identity")

    m = mechanism.loc[
        num(mechanism["season"]).isin(TARGET_SEASONS)
        & num(mechanism["week"]).isin(TARGET_WEEKS)
    ].copy()
    m["season"] = num(m["season"]).astype(int)
    m["week"] = num(m["week"]).astype(int)
    m["team"] = m["team"].map(canon_team)
    m["player_name_key"] = m["player_name_key"].map(key)
    m["stage1_history_attached"] = num(m["p_candidate"]).notna()
    m = m[["season", "week", "team", "family", "player_id", "stage1_history_attached"]].copy()
    if m.duplicated(["season", "week", "team", "family", "player_id"]).any():
        raise RuntimeError("duplicate Stage-1 scored identity")
    return u, m


def cap_with_residual(shares: np.ndarray) -> tuple[np.ndarray, float]:
    clean = np.nan_to_num(np.asarray(shares, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    if float(clean.sum()) > 0.95:
        clean *= 0.95 / float(clean.sum())
    residual = max(0.0, 1.0 - float(clean.sum()))
    probs = np.append(clean, residual)
    probs /= float(probs.sum())
    return probs[:-1], float(probs[-1])


def build_base_authority(metrics: pd.DataFrame, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = metrics.copy()
    x["team"] = x["team"].map(canon_team)
    x["player_name_key"] = x.get("player_clean_key", x["player"]).map(key)
    pcols = ["event_id", "team", "player_name_key"]
    x = x.sort_values(pcols + ["market"]).drop_duplicates(pcols, keep="last").copy()
    if x.duplicated(pcols).any():
        raise RuntimeError(f"duplicate canonical player authority {season} W{week}")

    ent, _ = materialize_target_entitlement(x)
    ent["player_name_key"] = ent["player_clean_key"].map(key)
    ent["target_q"] = num(ent["entitlement_tgt_share"])
    ent["target_residual"] = num(ent["entitlement_residual_share"])
    ent["plays"] = num(ent["rules_plays_est"])
    ent["pass_rate"] = num(ent["rules_pass_rate"])
    ent["target_denominator_mean"] = ent["plays"] * ent["pass_rate"]
    ent["rush_denominator_mean"] = ent["plays"] * (1.0 - ent["pass_rate"])
    ent["qb_pass_att_share"] = num(ent.get("qb_pass_att_share", pd.Series(np.nan, index=ent.index)))
    ent["qb_projection_eligible"] = num(ent.get("qb_projection_eligible", pd.Series(0, index=ent.index))).fillna(0)
    # Historical QB opportunity is already player-specific upstream: one leakage-safe
    # primary QB per team receives a strictly-prior pass-attempt share and backups
    # receive explicit zero share. The finite shared denominator here is the
    # canonical pregame pass/dropback opportunity state; no new QB mean is fit.
    ent["qb_pass_denominator_mean"] = ent["target_denominator_mean"]

    player_rows, room_rows = [], []
    for (event_id, team), g in ent.groupby(["event_id", "team"], sort=False):
        raw = num(g["rules_rush_share"]).fillna(0.0).to_numpy(float)
        used = _top_n_shares(raw, 5)
        q, residual = cap_with_residual(used)
        target_sum = float(num(g["target_q"]).fillna(0).sum())
        target_residual = float(num(g["target_residual"]).dropna().iloc[0])
        if abs(target_sum + target_residual - 1.0) > TOL:
            raise RuntimeError(f"target entitlement conservation drift {season} W{week} {team}")
        for j, (_, r) in enumerate(g.iterrows()):
            player_rows.append({
                "season": season, "week": week, "event_id": str(event_id), "team": str(team),
                "player_name_key": str(r["player_name_key"]),
                "position": str(r.get("position", "")).upper(),
                "rush_q": float(q[j]), "target_q": float(r["target_q"]),
                "target_denominator_mean": float(r["target_denominator_mean"]),
                "rush_denominator_mean": float(r["rush_denominator_mean"]),
                "qb_pass_att_share": float(r["qb_pass_att_share"]) if np.isfinite(r["qb_pass_att_share"]) else np.nan,
                "qb_projection_eligible": float(r["qb_projection_eligible"]),
                "qb_pass_denominator_mean": float(r["qb_pass_denominator_mean"]),
            })
        qbg = g.loc[g.get("position", pd.Series("", index=g.index)).astype(str).str.upper().eq("QB")].copy()
        qbshares = num(qbg.get("qb_pass_att_share", pd.Series(dtype=float))).dropna()
        qb_sum = float(qbshares.sum()) if len(qbshares) else np.nan
        qb_residual = max(0.0, 1.0 - qb_sum) if np.isfinite(qb_sum) and qb_sum <= 1.0 + TOL else np.nan
        qb_mass_gap = abs(qb_sum + qb_residual - 1.0) if np.isfinite(qb_residual) else np.inf
        qb_primary_count = int(num(qbg.get("qb_projection_eligible", pd.Series(dtype=float))).fillna(0).eq(1).sum())
        room_rows.append({
            "season": season, "week": week, "event_id": str(event_id), "team": str(team),
            "target_mass_gap": abs(target_sum + target_residual - 1.0),
            "rush_mass_gap": abs(float(q.sum()) + residual - 1.0),
            "qb_mass_gap": float(qb_mass_gap),
            "qb_primary_count": qb_primary_count,
            "qb_denominator_finite": bool(
                len(qbg)
                and np.isfinite(num(qbg["qb_pass_denominator_mean"])).all()
                and (num(qbg["qb_pass_denominator_mean"]) > 0).all()
            ),
            "target_denominator_finite": bool(
                np.isfinite(g["target_denominator_mean"]).all()
                and (g["target_denominator_mean"] > 0).all()
            ),
            "rush_denominator_finite": bool(
                np.isfinite(g["rush_denominator_mean"]).all()
                and (g["rush_denominator_mean"] > 0).all()
            ),
        })
    return pd.DataFrame(player_rows), pd.DataFrame(room_rows)


def wr_fold_authority(fold_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = read_csv(
        fold_root / "wr_r15_original/backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv",
        "WR-R15 OOS predictions",
    )
    audit = read_csv(
        fold_root / "wr_r15_original/backtests/wr_r15_wr1_anchor_v1/wr_r15_conservation_audit.csv",
        "WR-R15 conservation audit",
    )
    q = pred.loc[pred["variant"].astype(str).eq(WR_VARIANT)].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_name_key"] = q["player_clean_key"].map(key)
    q["candidate_q"] = num(q["entitlement_tgt_share"])
    q["pred_targets"] = num(q["pred_targets"])
    q["denominator_implied"] = np.where(
        q["candidate_q"].gt(0), q["pred_targets"] / q["candidate_q"], np.nan
    )
    q = q.loc[q["season"].eq(2024) & q["week"].isin(TARGET_WEEKS)].copy()

    a = audit.loc[
        num(audit["test_season"]).eq(2024) & num(audit["week"]).isin(TARGET_WEEKS)
    ].copy()
    a["season"] = 2024
    a["week"] = num(a["week"]).astype(int)
    a["team"] = a["team"].map(canon_team)
    a["conservation_pass"] = (
        num(a["wr_room_mass_gap"]).abs().le(TOL)
        & num(a["team_player_mass_gap"]).abs().le(TOL)
        & num(a["secondary_pool_gap"]).abs().le(TOL)
        & num(a["max_non_wr_entitlement_delta"]).abs().le(TOL)
        & num(a["sportsbook_inputs_used"]).fillna(0).eq(0)
        & num(a["current_or_future_outcomes_used"]).fillna(0).eq(0)
        & num(a["train_season"]).lt(num(a["test_season"]))
    )
    return q, a


def te_fold_authority(fold_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    case = read_csv(
        fold_root / "te_r5p_recovered/te_r5p_oos_player_casebook.csv", "TE-R5P OOS casebook"
    )
    mass = read_csv(
        fold_root / "te_r5p_recovered/te_r5p_team_mass_audit.csv", "TE-R5P mass audit"
    )
    result = json.loads(
        (fold_root / "te_r5p_recovered/te_r5p_result.json").read_text(encoding="utf-8")
    )
    q = case.loc[
        num(case["season"]).isin(TARGET_SEASONS) & num(case["week"]).isin(TARGET_WEEKS)
    ].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_name_key"] = q["player_clean_key"].map(key)
    q["b0_target_probability"] = num(q["b0_target_probability"])
    q["candidate_room_share_r5p"] = num(q["candidate_room_share_r5p"])
    pool = q.groupby(["season", "week", "team"])["b0_target_probability"].transform("sum")
    q["candidate_q"] = q["candidate_room_share_r5p"] * pool
    q["denominator_implied"] = np.where(
        q["b0_target_probability"].gt(0),
        num(q["b0_expected_targets"]) / q["b0_target_probability"],
        np.nan,
    )

    m = mass.loc[
        num(mass["season"]).isin(TARGET_SEASONS) & num(mass["week"]).isin(TARGET_WEEKS)
    ].copy()
    m["season"] = num(m["season"]).astype(int)
    m["week"] = num(m["week"]).astype(int)
    m["team"] = m["team"].map(canon_team)
    m["conservation_pass"] = num(m["mass_gap"]).abs().le(TOL)
    return q, m, result


def overlay_specialist(base: pd.DataFrame, spec: pd.DataFrame, family: str) -> pd.DataFrame:
    positions = {"TE"} if family == "TE" else {"WR", "LWR", "RWR", "SWR"}
    b = base.loc[base["position"].astype(str).str.upper().isin(positions)].copy()
    keys = ["season", "week", "team", "player_name_key"]
    s = spec[keys + ["candidate_q", "denominator_implied"]].copy()
    if s.duplicated(keys).any():
        raise RuntimeError(f"duplicate specialist authority family={family}")
    out = b.merge(s, on=keys, how="left", validate="one_to_one")
    out["authority_q"] = np.where(
        out["candidate_q"].notna(), out["candidate_q"], out["target_q"]
    )
    out["specialist_row"] = out["candidate_q"].notna()
    out["authority_covered"] = (
        out["specialist_row"] | num(out["target_q"]).fillna(np.inf).abs().le(TOL)
    )
    out["denominator_finite"] = np.where(
        out["specialist_row"],
        num(out["denominator_implied"]).gt(0) & np.isfinite(num(out["denominator_implied"])),
        num(out["target_denominator_mean"]).gt(0)
        & np.isfinite(num(out["target_denominator_mean"])),
    )
    return out


def evaluate_cell(
    *, cell: str, season: int, stage1: pd.DataFrame, mechanism: pd.DataFrame,
    authority: pd.DataFrame | None, rooms: pd.DataFrame | None,
    forced_block_reason: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    family = FAMILY_FOR_CELL[cell]
    target = stage1.loc[
        stage1["season"].eq(season) & stage1["family"].eq(family)
    ].copy()
    target_teamweeks = target[["week", "team"]].drop_duplicates()
    target_rows = len(target)

    if forced_block_reason:
        rec = {
            "cell": cell, "season": season, "target_player_games": int(target_rows),
            "target_players": int(target["player_id"].nunique()),
            "scheduled_team_games": int(len(target_teamweeks)),
            "authority_player_games": 0, "authority_players": 0,
            "scheduled_team_game_coverage": 0.0,
            "stable_identity_coverage": float(num(target["stable_roster_identity"]).fillna(0).mean()) if len(target) else 0.0,
            "stage1_history_attach_rate": 0.0,
            "canonical_pregame_entitlement_coverage": 0.0,
            "finite_denominator_coverage": 0.0,
            "conservation_pass_rate": 0.0,
            "ambiguous_identity_rate": 0.0,
            "chronology_violations": int(num(target["chronology_violations"]).fillna(0).sum()),
            "blocked_reason": forced_block_reason,
            "source_ready": False,
            "disposition": f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED",
        }
        return rec, target.assign(authority_covered=False)

    if authority is None or authority.empty:
        return evaluate_cell(
            cell=cell, season=season, stage1=stage1, mechanism=mechanism,
            authority=None, rooms=None, forced_block_reason="NO_CANONICAL_AUTHORITY_ROWS",
        )

    a = authority.loc[authority["season"].eq(season)].copy()
    a["team"] = a["team"].map(canon_team)
    a["player_name_key"] = a["player_name_key"].map(key)
    join_keys = ["season", "week", "team", "player_name_key"]
    dup = a.duplicated(join_keys, keep=False)
    ambiguous_keys = set(
        tuple(v) for v in a.loc[dup, join_keys].itertuples(index=False, name=None)
    )
    a = a.loc[~dup].copy()
    keep = join_keys + ["authority_q", "authority_covered", "denominator_finite"]
    z = target.merge(a[keep], on=join_keys, how="left", validate="many_to_one")
    z["authority_covered"] = z["authority_covered"].fillna(False).astype(bool)
    z["denominator_finite"] = z["denominator_finite"].fillna(False).astype(bool)
    z["ambiguous_identity"] = [
        tuple(v) in ambiguous_keys
        for v in z[join_keys].itertuples(index=False, name=None)
    ]

    mh = mechanism.loc[
        mechanism["season"].eq(season) & mechanism["family"].eq(family)
    ].copy()
    z = z.merge(
        mh[["season", "week", "team", "family", "player_id", "stage1_history_attached"]],
        on=["season", "week", "team", "family", "player_id"],
        how="left", validate="one_to_one",
    )
    z["stage1_history_attached"] = z["stage1_history_attached"].fillna(False).astype(bool)

    covered = z.loc[z["authority_covered"] & ~z["ambiguous_identity"]].copy()
    covered_tw = covered[["week", "team"]].drop_duplicates()
    denominator_tw0 = covered.loc[
        covered["denominator_finite"], ["week", "team"]
    ].drop_duplicates()

    room_pass_rate = 0.0
    if rooms is not None and not rooms.empty:
        rr = rooms.loc[rooms["season"].eq(season)].copy()
        rr_tw = rr[["week", "team"]].drop_duplicates()
        auth_tw = covered_tw.merge(rr_tw, on=["week", "team"], how="inner")
        denominator_tw = denominator_tw0.merge(rr_tw, on=["week", "team"], how="inner")
        rr = rr.merge(target_teamweeks, on=["week", "team"], how="inner")
        room_pass_rate = (
            float(rr["conservation_pass"].fillna(False).mean()) if len(rr) else 0.0
        )
    else:
        auth_tw = covered_tw.iloc[0:0].copy()
        denominator_tw = denominator_tw0.iloc[0:0].copy()

    player_games = int(len(covered))
    players = int(covered["player_id"].nunique())
    min_games, min_players = SUPPORT[cell]
    sched_cov = len(auth_tw) / len(target_teamweeks) if len(target_teamweeks) else 0.0
    stable_cov = (
        float(num(target["stable_roster_identity"]).fillna(0).mean()) if len(target) else 0.0
    )
    hist_attach = (
        float(covered["stage1_history_attached"].mean()) if len(covered) else 0.0
    )
    entitlement_cov = player_games / target_rows if target_rows else 0.0
    denom_cov = (
        len(denominator_tw) / len(target_teamweeks) if len(target_teamweeks) else 0.0
    )
    ambiguous_rate = float(z["ambiguous_identity"].mean()) if len(z) else 0.0
    chrono = int(num(target["chronology_violations"]).fillna(0).sum())

    gates = {
        "scheduled_team_game_coverage_ge_099": sched_cov >= 0.99,
        "stable_identity_coverage_ge_099": stable_cov >= 0.99,
        "stage1_history_attach_rate_ge_095": hist_attach >= 0.95,
        "canonical_pregame_entitlement_coverage_ge_095": entitlement_cov >= 0.95,
        "finite_denominator_coverage_ge_099": denom_cov >= 0.99,
        "conservation_pass_rate_ge_099": room_pass_rate >= 0.99,
        "ambiguous_identity_rate_le_0005": ambiguous_rate <= 0.005,
        "chronology_violations_zero": chrono == 0,
        "support_player_games": player_games >= min_games,
        "support_players": players >= min_players,
    }
    ready = all(gates.values())
    rec = {
        "cell": cell, "season": season,
        "target_player_games": int(target_rows),
        "target_players": int(target["player_id"].nunique()),
        "scheduled_team_games": int(len(target_teamweeks)),
        "authority_player_games": player_games, "authority_players": players,
        "scheduled_team_game_coverage": float(sched_cov),
        "stable_identity_coverage": stable_cov,
        "stage1_history_attach_rate": hist_attach,
        "canonical_pregame_entitlement_coverage": float(entitlement_cov),
        "finite_denominator_coverage": float(denom_cov),
        "conservation_pass_rate": room_pass_rate,
        "ambiguous_identity_rate": ambiguous_rate,
        "chronology_violations": chrono,
        "gates": gates,
        "source_ready": bool(ready),
        "disposition": (
            f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY"
            if ready else f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED"
        ),
    }
    return rec, z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-universe", type=Path, required=True)
    ap.add_argument("--stage1-mechanism", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir-2024", type=Path, required=True)
    ap.add_argument("--universe-dir-2025", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    stage1, mechanism = prepare_stage1(
        read_csv(args.stage1_universe, "Stage-1 universe"),
        read_csv(args.stage1_mechanism, "Stage-1 mechanism"),
    )
    team = read_csv(args.team_weekly, "team history")
    logs = read_csv(args.player_logs, "player logs")
    schedule = read_csv(args.schedule, "schedule")
    for frame in (team, logs, schedule):
        frame["season"] = num(frame["season"]).astype("Int64")
        frame["week"] = num(frame["week"]).astype("Int64")
    team["team"] = team["team"].map(canon_team)
    logs["team"] = logs["team"].map(canon_team)
    if "opponent" in logs.columns:
        logs["opponent"] = logs["opponent"].map(canon_team)
    schedule["team"] = schedule["team"].map(canon_team)
    if "opponent" in schedule.columns:
        schedule["opponent"] = schedule["opponent"].map(canon_team)

    wr_spec, wr_rooms = wr_fold_authority(args.fold_root)
    te_spec, te_rooms, te_result = te_fold_authority(args.fold_root)
    if int(te_result.get("same_or_future_observations_used", 0) or 0) != 0:
        raise RuntimeError("TE-R5P recovered authority reports same/future participation")
    if int(te_result.get("sportsbook_inputs_used", 0) or 0) != 0:
        raise RuntimeError("TE-R5P recovered authority reports sportsbook input")

    base_parts, base_room_parts = [], []
    udirs = {2024: args.universe_dir_2024, 2025: args.universe_dir_2025}
    for season in TARGET_SEASONS:
        for week in TARGET_WEEKS:
            universe = read_csv(
                udirs[season] / f"{season}_week_{week:02d}.csv",
                f"historical universe {season} W{week}",
            )
            metrics = build_week_metrics(
                player_logs=logs, team_weekly=team, pregame_universe=universe,
                schedule=schedule, season=season, week=week,
            )
            auth, rooms = build_base_authority(metrics, season, week)
            base_parts.append(auth)
            base_room_parts.append(rooms)
    base = pd.concat(base_parts, ignore_index=True)
    base_rooms = pd.concat(base_room_parts, ignore_index=True)

    qb = base.loc[base["position"].eq("QB")].copy()
    qb_pass = qb.assign(
        authority_q=qb["qb_pass_att_share"],
        authority_covered=qb["qb_pass_att_share"].notna(),
        denominator_finite=np.isfinite(qb["qb_pass_denominator_mean"])
        & qb["qb_pass_denominator_mean"].gt(0),
    )
    qb_rooms = base_rooms.assign(
        conservation_pass=base_rooms["qb_mass_gap"].le(TOL)
        & base_rooms["qb_primary_count"].eq(1)
        & base_rooms["qb_denominator_finite"].fillna(False)
    )

    rb_pos = {"RB", "HB", "FB"}
    rb = base.loc[base["position"].isin(rb_pos)].copy()
    rb_rush = rb.assign(
        authority_q=rb["rush_q"],
        authority_covered=rb["rush_q"].notna(),
        denominator_finite=np.isfinite(rb["rush_denominator_mean"])
        & rb["rush_denominator_mean"].gt(0),
    )
    rb_recv = rb.assign(
        authority_q=rb["target_q"],
        authority_covered=rb["target_q"].notna(),
        denominator_finite=np.isfinite(rb["target_denominator_mean"])
        & rb["target_denominator_mean"].gt(0),
    )
    rb_rush_rooms = base_rooms.assign(
        conservation_pass=base_rooms["rush_mass_gap"].le(TOL)
    )
    rb_recv_rooms = base_rooms.assign(
        conservation_pass=base_rooms["target_mass_gap"].le(TOL)
    )

    te_auth = overlay_specialist(base, te_spec, "TE")
    wr24_auth = overlay_specialist(
        base.loc[base["season"].eq(2024)], wr_spec, "WR"
    )
    te_rooms = te_rooms[["season", "week", "team", "conservation_pass"]].copy()
    wr_rooms = wr_rooms[["season", "week", "team", "conservation_pass"]].copy()

    results, detail_parts = [], []
    for season in TARGET_SEASONS:
        rec, det = evaluate_cell(
            cell="QB_PASSING", season=season, stage1=stage1, mechanism=mechanism,
            authority=qb_pass, rooms=qb_rooms,
        )
        results.append(rec)
        detail_parts.append(det.assign(cell="QB_PASSING"))

        for cell, auth, rooms in (
            ("RB_RUSHING", rb_rush, rb_rush_rooms),
            ("RB_RECEIVING", rb_recv, rb_recv_rooms),
        ):
            rec, det = evaluate_cell(
                cell=cell, season=season, stage1=stage1, mechanism=mechanism,
                authority=auth, rooms=rooms,
            )
            results.append(rec)
            detail_parts.append(det.assign(cell=cell))

        if season == 2025:
            rec, det = evaluate_cell(
                cell="WR_RECEIVING", season=season, stage1=stage1, mechanism=mechanism,
                authority=None, rooms=None,
                forced_block_reason=(
                    "WR_R15_2025_OOS_AUTHORITY_FORBIDDEN_BY_FROZEN_CONTRACT__"
                    "FINAL_FIT_TRAINING_OVERLAP_CANNOT_BE_RELABELED_OOS"
                ),
            )
        else:
            rec, det = evaluate_cell(
                cell="WR_RECEIVING", season=season, stage1=stage1, mechanism=mechanism,
                authority=wr24_auth, rooms=wr_rooms,
            )
        results.append(rec)
        detail_parts.append(det.assign(cell="WR_RECEIVING"))

        rec, det = evaluate_cell(
            cell="TE_RECEIVING", season=season, stage1=stage1, mechanism=mechanism,
            authority=te_auth, rooms=te_rooms,
        )
        results.append(rec)
        detail_parts.append(det.assign(cell="TE_RECEIVING"))

    per_season = pd.DataFrame(results)
    per_cell = []
    for cell in CELLS:
        q = per_season.loc[per_season["cell"].eq(cell)].copy()
        ready = len(q) == 2 and bool(q["source_ready"].all())
        per_cell.append({
            "cell": cell,
            "season_2024_ready": bool(
                q.loc[q["season"].eq(2024), "source_ready"].iloc[0]
            ),
            "season_2025_ready": bool(
                q.loc[q["season"].eq(2025), "source_ready"].iloc[0]
            ),
            "disposition": (
                f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY"
                if ready else
                f"{cell}_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED"
            ),
        })
    ready_cells = [
        x["cell"] for x in per_cell
        if x["disposition"].endswith("_SOURCE_READY")
    ]
    if len(ready_cells) == len(CELLS):
        overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_ALL"
    elif ready_cells:
        overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL"
    else:
        overall = "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_BLOCKED"

    result = {
        "version": VERSION,
        "disposition": overall,
        "target_seasons": list(TARGET_SEASONS),
        "target_weeks": [5, 18],
        "ready_cells": ready_cells,
        "per_cell": per_cell,
        "season_cells": results,
        "integrity": {
            "target_game_outcomes_used_as_predictors": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_used": 0,
            "new_share_models_fit": 0,
            "production_changed": False,
            "stage1_chronology_violations": int(
                num(stage1["chronology_violations"]).fillna(0).sum()
            ),
            "wr_2025_oos_authority_used": False,
            "qb_share_model_invented": False,
        },
    }
    if result["integrity"]["stage1_chronology_violations"] != 0:
        result["disposition"] = (
            "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_INTEGRITY_FAILURE"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_season.to_csv(
        args.out_dir / "active_role_conditional_allocation_source_season_cells.csv",
        index=False,
    )
    pd.DataFrame(per_cell).to_csv(
        args.out_dir / "active_role_conditional_allocation_source_cells.csv",
        index=False,
    )
    pd.concat(detail_parts, ignore_index=True, sort=False).to_csv(
        args.out_dir / "active_role_conditional_allocation_source_join_detail.csv",
        index=False,
    )
    (args.out_dir / "active_role_conditional_allocation_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
