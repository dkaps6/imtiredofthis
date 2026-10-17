#!/usr/bin/env python3
"""PRISM Week-5 Stage-1/Stage-2 Retrospective Replay V1.

Two explicit phases:
  build -- uses ONLY frozen pregame football state + frozen strictly-prior
           Stage-1 source authority. It accepts no Week-5 outcome input.
  score -- opens frozen Week-5 outcome artifacts only after loading the immutable
           candidate row file produced by build.

Research-only. No production mutation. No sportsbook/OddsAPI input.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_WEEK5_STAGE12_RETROSPECTIVE_REPLAY_V1"
TARGET_SEASON = 2026
TARGET_WEEK = 5
PRIOR_WEEKS = (1, 2, 3, 4)
PLAYER_WINDOW = 4
PLAYER_MIN = 3
PRIOR_STRENGTH = 2.0
TOL = 1e-12
CELLS = ("RB_RUSHING", "RB_RECEIVING", "TE_RECEIVING")

FORBIDDEN = (
    "sportsbook", "bookmaker", "prop_line", "market_line", "over_odds",
    "under_odds", "moneyline", "closing_line", "no_vig", "implied_prob",
)


def num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def read_csv(path: Path, label: str, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False, usecols=usecols)
    x.columns = [str(c).strip().lower() for c in x.columns]
    bad = [c for c in x.columns if any(t in c for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden market columns in {label}: {bad}")
    return x


def base_name_key(v: object) -> str:
    s = "".join(ch.lower() for ch in str(v) if ch.isalnum())
    # Identity bridge only; no football state is inferred from suffix removal.
    for suffix in ("iii", "jr", "sr", "ii", "iv"):
        if s.endswith(suffix) and len(s) > len(suffix) + 3:
            return s[: -len(suffix)]
    return s


def baseline_probability(k_pos: int, n_pos: int) -> float:
    return (float(k_pos) + 0.5) / (float(n_pos) + 1.0)


def candidate_probability(k_player: int, n_player: int, p_baseline: float) -> float:
    return (float(k_player) + PRIOR_STRENGTH * float(p_baseline)) / (
        float(n_player) + PRIOR_STRENGTH
    )


def top_n_rush_shares(raw: np.ndarray, n: int = 5) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 0.95)
    if len(clean) > n:
        order = np.argsort(-clean, kind="stable")
        out = np.zeros_like(clean)
        out[order[:n]] = clean[order[:n]]
        clean = out
    if float(clean.sum()) > 0.95:
        clean *= 0.95 / float(clean.sum())
    return clean


def enumerate_active_states(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    weights, named, comp = [], [], []
    for bits in itertools.product((0, 1), repeat=len(q)):
        a = np.asarray(bits, dtype=float)
        w = float(np.prod(np.where(a > 0, p, 1.0 - p)))
        z = float(q0 + np.dot(a, q))
        if z > TOL:
            r = (a * q) / z
            r0 = q0 / z
        else:
            r = np.zeros(len(q), dtype=float)
            r0 = 1.0
        if abs(float(r.sum()) + float(r0) - 1.0) > TOL:
            raise RuntimeError("conditional allocation conservation drift")
        weights.append(w)
        named.append(r)
        comp.append(float(r0))
    w = np.asarray(weights, dtype=float)
    rr = np.vstack(named) if named else np.zeros((1, 0), dtype=float)
    r0 = np.asarray(comp, dtype=float)
    if abs(float(w.sum()) - 1.0) > TOL:
        raise RuntimeError("active-state probability mass drift")
    return w, rr, r0


def resolve_target_ids(
    target: pd.DataFrame,
    week5_identity: pd.DataFrame,
    rb_lock: pd.DataFrame,
    wrte_lock: pd.DataFrame,
) -> pd.DataFrame:
    x = target.copy()
    x["player_id"] = ""
    x["identity_source"] = "UNRESOLVED"

    rb_map = {
        (str(r.team), str(r.player)): str(r.gsis_id).strip()
        for r in rb_lock.itertuples(index=False)
        if pd.notna(r.gsis_id) and str(r.gsis_id).strip()
    }
    te_map = {
        (str(r.team), str(r.player)): str(r.receiver_id).strip()
        for r in wrte_lock.loc[wrte_lock["position_family"].eq("TE")].itertuples(index=False)
        if pd.notna(r.receiver_id) and str(r.receiver_id).strip()
    }

    ri = week5_identity.copy()
    ri["base_key"] = ri["player_name_key"].map(base_name_key)
    ri = ri.loc[
        ri["stable_roster_identity"].eq(1)
        & ri["player_id"].astype("string").fillna("").str.strip().ne("")
        & ri["family"].isin(["RB", "TE"])
    ].copy()

    for idx, r in x.iterrows():
        fam = str(r["position_family"])
        key = (str(r["team"]), str(r["player"]))
        pid = rb_map.get(key, "") if fam == "RB" else te_map.get(key, "")
        if pid:
            x.at[idx, "player_id"] = pid
            x.at[idx, "identity_source"] = "FROZEN_WEEK5_RB_LOCK" if fam == "RB" else "FROZEN_WEEK5_WRTE_LOCK"
            continue

        base = base_name_key(r["player_clean_key"])
        cand = ri.loc[
            ri["team"].astype(str).eq(str(r["team"]))
            & ri["family"].astype(str).eq(fam)
            & (
                ri["player_name_key"].astype(str).eq(str(r["player_clean_key"]))
                | ri["base_key"].astype(str).eq(base)
            )
        ]
        ids = sorted(set(cand["player_id"].astype(str)) - {""})
        if len(ids) == 1:
            x.at[idx, "player_id"] = ids[0]
            x.at[idx, "identity_source"] = "FROZEN_WEEK5_ROSTER_IDENTITY"

    return x


def build(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    state = read_csv(args.player_state, "Week-5 frozen player state")
    required_state = {
        "event_id", "team", "player", "player_clean_key", "position_family",
        "rules_plays_est", "rules_pass_rate", "rules_rush_share",
        "entitlement_tgt_share", "expected_carries", "expected_targets",
    }
    if required_state - set(state.columns):
        raise RuntimeError(f"player state missing {sorted(required_state-set(state.columns))}")

    # Read strict-prior participation columns for Weeks 1-4 only. This source
    # artifact also contains Week 5 rows, but target-week participation columns
    # are never consumed as predictors.
    prior_cols = [
        "season", "week", "team", "player_id", "position", "player_name_key",
        "family", "stable_roster_identity", "teamweek_snap_source_valid",
        "positive_offensive_participation",
    ]
    prior_all = read_csv(args.stage1_prior_universe, "frozen Stage-1 prior universe", usecols=prior_cols)
    prior = prior_all.loc[
        num(prior_all["season"]).eq(TARGET_SEASON)
        & num(prior_all["week"]).isin(PRIOR_WEEKS)
    ].copy()
    if len(prior) == 0 or int(num(prior["week"]).max()) != 4:
        raise RuntimeError("strict-prior Stage-1 history is not complete through Week 4")
    if num(prior["week"]).ge(TARGET_WEEK).any():
        raise RuntimeError("target/future participation leaked into Stage-1 predictor history")

    # Identity-only Week-5 roster view; target-game participation is not read.
    id_cols = [
        "season", "week", "team", "player_id", "player_name_key", "family",
        "stable_roster_identity",
    ]
    week5_identity = read_csv(args.stage1_prior_universe, "frozen Week-5 roster identity", usecols=id_cols)
    week5_identity = week5_identity.loc[
        num(week5_identity["season"]).eq(TARGET_SEASON)
        & num(week5_identity["week"]).eq(TARGET_WEEK)
    ].copy()

    inactive = read_csv(args.official_inactives, "frozen official inactive authority")
    inactive = inactive.loc[
        num(inactive["season"]).eq(TARGET_SEASON)
        & num(inactive["week"]).eq(TARGET_WEEK)
    ].copy()
    if inactive.empty:
        raise RuntimeError("Week-5 official inactive authority empty")
    inactive_keys = set(
        (str(r.team), str(r.inactive_name_key))
        for r in inactive.itertuples(index=False)
    )

    rb_lock = read_csv(args.rb_identity_lock, "frozen Week-5 RB identity lock")
    wrte_lock = read_csv(args.wrte_identity_lock, "frozen Week-5 WR/TE identity lock")

    universe = state.loc[state["position_family"].isin(["RB", "TE"])].copy()
    universe["definitive_inactive"] = [
        int((str(r.team), str(r.player_clean_key)) in inactive_keys)
        for r in universe.itertuples(index=False)
    ]

    # Canonical rushing q is built on the full pregame universe exactly as in
    # the frozen Week-6 Stage-2 shadow, before definitive inactive target rows
    # are removed.
    rush_rows = []
    for (event_id, team), g in universe.loc[universe["position_family"].eq("RB")].groupby(
        ["event_id", "team"], sort=False
    ):
        q = top_n_rush_shares(num(g["rules_rush_share"]).fillna(0.0).to_numpy(float), 5)
        for qq, (_, r) in zip(q, g.iterrows()):
            rush_rows.append({
                "event_id": str(event_id),
                "team": str(team),
                "player_clean_key": str(r["player_clean_key"]),
                "rush_q": float(qq),
            })
    rush_q = pd.DataFrame(rush_rows)

    target = universe.loc[universe["definitive_inactive"].eq(0)].copy()
    target = resolve_target_ids(target, week5_identity, rb_lock, wrte_lock)

    identity_metrics = {}
    for fam in ("RB", "TE"):
        z = target.loc[target["position_family"].eq(fam)]
        rate = float(z["player_id"].astype(str).str.strip().ne("").mean()) if len(z) else 0.0
        identity_metrics[f"{fam.lower()}_target_identity_rate"] = rate
        if rate < 0.99:
            raise RuntimeError(f"{fam} target stable-ID coverage {rate:.6f} < 0.99")

    hist = prior.loc[
        prior["family"].isin(["RB", "TE"])
        & prior["stable_roster_identity"].eq(1)
        & prior["teamweek_snap_source_valid"].eq(1)
    ].copy()
    p0 = {}
    for fam in ("RB", "TE"):
        h = hist.loc[hist["family"].eq(fam)]
        if len(h) < 100:
            raise RuntimeError(f"{fam} Stage-1 baseline support <100")
        p0[fam] = baseline_probability(
            int(num(h["positive_offensive_participation"]).sum()),
            int(len(h)),
        )

    p_rows = []
    for r in target.itertuples(index=False):
        fam = str(r.position_family)
        pid = str(r.player_id).strip()
        if pid:
            h = hist.loc[
                hist["family"].eq(fam)
                & hist["team"].astype(str).eq(str(r.team))
                & hist["player_id"].astype(str).eq(pid)
            ].sort_values("week").tail(PLAYER_WINDOW)
        else:
            h = hist.iloc[0:0]
        n = int(len(h))
        k = int(num(h["positive_offensive_participation"]).sum()) if n else 0
        if pid and n >= PLAYER_MIN:
            p = candidate_probability(k, n, p0[fam])
            source = "NAMED_PLAYER_STAGE1"
        else:
            p = p0[fam]
            source = "POSITION_FALLBACK_STAGE1"
        if not (0.0 < p < 1.0):
            raise RuntimeError("p_active escaped open interval")
        p_rows.append({
            "event_id": str(r.event_id),
            "team": str(r.team),
            "player_clean_key": str(r.player_clean_key),
            "p_active": float(p),
            "stage1_probability_source": source,
            "n_player_prior": n,
            "k_player_prior": k,
        })
    pactive = pd.DataFrame(p_rows)

    target = target.merge(
        pactive,
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    target = target.merge(
        rush_q,
        on=["event_id", "team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    target["rush_q"] = num(target["rush_q"]).fillna(0.0)

    detail_rows, room_rows = [], []
    for cell in CELLS:
        fam = "RB" if cell.startswith("RB_") else "TE"
        qcol = "rush_q" if cell == "RB_RUSHING" else "entitlement_tgt_share"
        original_col = "expected_carries" if cell == "RB_RUSHING" else "expected_targets"
        for (event_id, team), g0 in target.loc[target["position_family"].eq(fam)].groupby(
            ["event_id", "team"], sort=False
        ):
            g = g0.copy().sort_values(["player_id", "player_clean_key"]).reset_index(drop=True)
            q_all = num(g[qcol]).fillna(0.0).to_numpy(float)
            if not np.isfinite(q_all).all() or (q_all < -TOL).any():
                raise RuntimeError(f"{cell} invalid canonical q {event_id} {team}")
            if float(q_all.sum()) > 1.0 + TOL:
                raise RuntimeError(f"{cell} named q exceeds one {event_id} {team}")
            positive = q_all > TOL
            q = q_all[positive]
            p = num(g.loc[positive, "p_active"]).to_numpy(float)
            if len(q):
                weights, state_named, state_comp = enumerate_active_states(q, p)
                expected = weights @ state_named
                expected_comp = float(weights @ state_comp)
            else:
                weights = np.asarray([1.0])
                expected = np.zeros(0, dtype=float)
                expected_comp = 1.0

            plays = float(num(g["rules_plays_est"]).dropna().iloc[0])
            pass_rate = float(num(g["rules_pass_rate"]).dropna().iloc[0])
            team_mean = plays * (1.0 - pass_rate) if cell == "RB_RUSHING" else plays * pass_rate

            exp_map: dict[int, float] = {}
            j = 0
            for idx, pos in enumerate(positive.tolist()):
                exp_map[idx] = float(expected[j]) if pos else float(q_all[idx])
                if pos:
                    j += 1

            mass_gap = abs(float(expected.sum()) + expected_comp - 1.0)
            if mass_gap > TOL:
                raise RuntimeError(f"{cell} candidate mass drift {event_id} {team}")

            for idx, r in g.iterrows():
                cq = float(exp_map[idx])
                original = float(num(pd.Series([r[original_col]])).iloc[0])
                detail_rows.append({
                    "version": VERSION,
                    "season": TARGET_SEASON,
                    "week": TARGET_WEEK,
                    "event_id": str(event_id),
                    "team": str(team),
                    "cell": cell,
                    "family": fam,
                    "player": str(r["player"]),
                    "player_clean_key": str(r["player_clean_key"]),
                    "player_id": str(r["player_id"]).strip(),
                    "identity_source": str(r["identity_source"]),
                    "p_active": float(r["p_active"]),
                    "stage1_probability_source": str(r["stage1_probability_source"]),
                    "n_player_prior": int(r["n_player_prior"]),
                    "k_player_prior": int(r["k_player_prior"]),
                    "canonical_q": float(q_all[idx]),
                    "candidate_expected_q": cq,
                    "share_shift": float(cq - q_all[idx]),
                    "team_opportunity_mean": float(team_mean),
                    "original_week5_expected_opportunity": original,
                    "canonical_parent_expected_opportunity": float(team_mean * q_all[idx]),
                    "stage12_candidate_expected_opportunity": float(team_mean * cq),
                    "target_definitive_inactive": 0,
                    "target_week_outcomes_read_by_builder": 0,
                    "sportsbook_inputs_used": 0,
                })

            room_rows.append({
                "season": TARGET_SEASON,
                "week": TARGET_WEEK,
                "event_id": str(event_id),
                "team": str(team),
                "cell": cell,
                "players": int(len(g)),
                "positive_q_players": int(positive.sum()),
                "canonical_named_mass": float(q.sum()),
                "canonical_complement_mass": float(1.0 - q.sum()),
                "candidate_named_mass": float(expected.sum()),
                "candidate_complement_mass": float(expected_comp),
                "state_count": int(len(weights)),
                "candidate_mass_gap": float(mass_gap),
                "team_opportunity_mean": float(team_mean),
            })

    detail = pd.DataFrame(detail_rows)
    rooms = pd.DataFrame(room_rows)
    if detail.empty:
        raise RuntimeError("candidate detail empty")

    candidate_path = out / "prism_week5_stage12_candidate_rows.csv"
    room_path = out / "prism_week5_stage12_candidate_rooms.csv"
    detail.to_csv(candidate_path, index=False)
    rooms.to_csv(room_path, index=False)

    cert = {
        "version": VERSION,
        "phase": "BUILD",
        "status": "WEEK5_STAGE12_CANDIDATE_FROZEN_BEFORE_OUTCOMES",
        "candidate_rows": int(len(detail)),
        "candidate_players": int(detail["player_id"].replace("", np.nan).nunique()),
        "rooms": int(len(rooms)),
        "cells": sorted(detail["cell"].unique().tolist()),
        **identity_metrics,
        "named_stage1_rows": int(detail["stage1_probability_source"].eq("NAMED_PLAYER_STAGE1").sum()),
        "position_fallback_rows": int(detail["stage1_probability_source"].eq("POSITION_FALLBACK_STAGE1").sum()),
        "max_candidate_mass_gap": float(rooms["candidate_mass_gap"].max()),
        "prior_history_max_week": int(num(prior["week"]).max()),
        "target_week_participation_columns_consumed": 0,
        "target_week_outcomes_read_by_builder": 0,
        "sportsbook_inputs": 0,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "thresholds_fit": 0,
        "production_changed": False,
        "candidate_row_digest": sha256_file(candidate_path),
        "room_row_digest": sha256_file(room_path),
    }
    (out / "prism_week5_stage12_build_certificate.json").write_text(
        json.dumps(cert, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(cert, indent=2, sort_keys=True))
    return 0


def _score_map(df: pd.DataFrame, id_col: str, value_col: str) -> dict[tuple[str, str], float]:
    z = df.copy()
    out = {}
    for r in z.itertuples(index=False):
        pid = getattr(r, id_col)
        val = getattr(r, value_col)
        if pd.isna(pid) or pd.isna(val):
            continue
        out[(str(r.team), str(pid))] = float(val)
    return out


def metric_block(g: pd.DataFrame) -> dict:
    if g.empty:
        return {
            "rows": 0, "players": 0, "rooms": 0,
            "original_mae": math.nan, "parent_mae": math.nan, "candidate_mae": math.nan,
            "candidate_vs_parent_mae_improvement": math.nan,
            "candidate_vs_original_mae_improvement": math.nan,
            "share_improved_vs_parent": math.nan,
            "share_improved_vs_original": math.nan,
        }
    a = num(g["actual_opportunity"]).to_numpy(float)
    o = num(g["original_week5_expected_opportunity"]).to_numpy(float)
    p = num(g["canonical_parent_expected_opportunity"]).to_numpy(float)
    c = num(g["stage12_candidate_expected_opportunity"]).to_numpy(float)
    oe, pe, ce = np.abs(o-a), np.abs(p-a), np.abs(c-a)
    return {
        "rows": int(len(g)),
        "players": int(g["player_id"].nunique()),
        "rooms": int(g[["event_id", "team"]].drop_duplicates().shape[0]),
        "named_stage1_support_rate": float(g["stage1_probability_source"].eq("NAMED_PLAYER_STAGE1").mean()),
        "original_mae": float(np.mean(oe)),
        "parent_mae": float(np.mean(pe)),
        "candidate_mae": float(np.mean(ce)),
        "candidate_vs_parent_mae_improvement": float(np.mean(pe)-np.mean(ce)),
        "candidate_vs_original_mae_improvement": float(np.mean(oe)-np.mean(ce)),
        "share_improved_vs_parent": float(np.mean(ce < pe)),
        "share_improved_vs_original": float(np.mean(ce < oe)),
        "mean_candidate_share_shift": float(num(g["share_shift"]).mean()),
    }


def score(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    candidate = pd.read_csv(args.candidate_rows, low_memory=False)
    candidate.columns = [str(col).strip().lower() for col in candidate.columns]
    allowed_integrity_cols = {"sportsbook_inputs_used"}
    bad_candidate_cols = [
        col for col in candidate.columns
        if col not in allowed_integrity_cols and any(token in col for token in FORBIDDEN)
    ]
    if bad_candidate_cols:
        raise RuntimeError(
            f"forbidden market columns in frozen Stage-1/2 candidate: {bad_candidate_cols}"
        )
    if "sportsbook_inputs_used" not in candidate.columns:
        raise RuntimeError("candidate missing sportsbook_inputs_used integrity field")
    if not num(candidate["sportsbook_inputs_used"]).fillna(-1).eq(0).all():
        raise RuntimeError("candidate sportsbook_inputs_used integrity field is nonzero")
    build_cert = json.loads(args.build_certificate.read_text())
    digest = sha256_file(args.candidate_rows)
    if digest != build_cert.get("candidate_row_digest"):
        raise RuntimeError("candidate digest does not match pre-outcome build certificate")
    if build_cert.get("target_week_outcomes_read_by_builder") != 0:
        raise RuntimeError("build certificate says target outcomes were read")

    rb_rush = read_csv(args.rb_rush_outcomes, "frozen Week-5 RB rush outcomes")
    rb_rec = read_csv(args.rb_rec_outcomes, "frozen Week-5 RB receiving outcomes")
    wrte = read_csv(args.wrte_outcomes, "frozen Week-5 WR/TE target outcomes")
    casebook = pd.read_csv(args.casebook, low_memory=False)
    casebook.columns = [str(c).strip().lower() for c in casebook.columns]

    rr = rb_rush.loc[
        rb_rush["actual_identity_scoreable"].fillna(False)
        & rb_rush["team_scoreable"].fillna(False)
    ].copy()
    rc = rb_rec.loc[
        rb_rec["actual_identity_scoreable"].fillna(False)
        & rb_rec["team_scoreable"].fillna(False)
    ].copy()
    te = wrte.loc[
        wrte["position_family"].eq("TE")
        & wrte["actual_identity_scoreable"].fillna(False)
    ].copy()

    maps = {
        "RB_RUSHING": _score_map(rr, "resolved_gsis_id", "actual_rushes"),
        "RB_RECEIVING": _score_map(rc, "resolved_gsis_id", "actual_targets"),
        "TE_RECEIVING": _score_map(te, "resolved_gsis_id", "actual_targets"),
    }

    scored_parts = []
    exclusions = []
    for cell in CELLS:
        g = candidate.loc[candidate["cell"].eq(cell)].copy()
        mp = maps[cell]
        actual = []
        supported = []
        for r in g.itertuples(index=False):
            key = (str(r.team), str(r.player_id))
            ok = bool(str(r.player_id).strip()) and key in mp
            supported.append(ok)
            actual.append(mp.get(key, math.nan))
            if not ok and float(r.canonical_q) > TOL:
                exclusions.append({
                    "cell": cell, "team": str(r.team), "player": str(r.player),
                    "player_id": str(r.player_id), "canonical_q": float(r.canonical_q),
                    "reason": "FROZEN_WEEK5_OUTCOME_AUTHORITY_NOT_SCOREABLE",
                })
        g["outcome_scoreable"] = supported
        g["actual_opportunity"] = actual
        scored_parts.append(g.loc[g["outcome_scoreable"]].copy())

    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    if scored.empty:
        raise RuntimeError("no scoreable Week-5 rows")

    scored["original_abs_error"] = (
        num(scored["original_week5_expected_opportunity"]) - num(scored["actual_opportunity"])
    ).abs()
    scored["parent_abs_error"] = (
        num(scored["canonical_parent_expected_opportunity"]) - num(scored["actual_opportunity"])
    ).abs()
    scored["candidate_abs_error"] = (
        num(scored["stage12_candidate_expected_opportunity"]) - num(scored["actual_opportunity"])
    ).abs()
    scored["candidate_improved_vs_parent"] = scored["candidate_abs_error"] < scored["parent_abs_error"]
    scored["candidate_improved_vs_original"] = scored["candidate_abs_error"] < scored["original_abs_error"]

    metrics = {cell: metric_block(scored.loc[scored["cell"].eq(cell)]) for cell in CELLS}

    # Conservative no-threshold diagnostic label.
    signs = []
    for cell in CELLS:
        m = metrics[cell]
        if not m["rows"]:
            continue
        signs.append((
            m["candidate_vs_parent_mae_improvement"] > 0,
            m["candidate_vs_original_mae_improvement"] > 0,
        ))
    if signs and all(a and b for a, b in signs):
        disposition = "WEEK5_STAGE12_CURRENT_SEASON_REPLAY_DIRECTIONALLY_SUPPORTIVE"
    elif signs and all((not a) and (not b) for a, b in signs):
        disposition = "WEEK5_STAGE12_CURRENT_SEASON_REPLAY_UNSUPPORTIVE"
    else:
        disposition = "WEEK5_STAGE12_CURRENT_SEASON_REPLAY_MIXED"

    # Frozen top failure casebook linkage after scoring only.
    cb = casebook.loc[
        casebook["component_label"].eq("OPPORTUNITY_DOMINANT")
        & casebook["position"].isin(["RB", "TE"])
    ].copy()
    cb["cell"] = np.where(
        cb["position"].eq("RB") & cb["market"].eq("rush_yards"), "RB_RUSHING",
        np.where(
            cb["position"].eq("RB") & cb["market"].isin(["rec_yards", "receptions"]),
            "RB_RECEIVING",
            np.where(
                cb["position"].eq("TE") & cb["market"].isin(["rec_yards", "receptions"]),
                "TE_RECEIVING", ""
            ),
        ),
    )
    cb = cb.loc[cb["cell"].ne("")].copy()
    cases = cb.merge(
        scored[[
            "cell", "team", "player_clean_key", "player_id",
            "original_week5_expected_opportunity",
            "canonical_parent_expected_opportunity",
            "stage12_candidate_expected_opportunity",
            "actual_opportunity", "candidate_improved_vs_parent",
            "candidate_improved_vs_original", "share_shift",
            "stage1_probability_source", "p_active",
        ]],
        on=["cell", "team", "player_clean_key"],
        how="left",
        validate="many_to_one",
    )
    if len(cases):
        cases["candidate_moved_toward_actual_from_original"] = (
            (num(cases["stage12_candidate_expected_opportunity"]) - num(cases["actual_opportunity"])).abs()
            < (num(cases["original_week5_expected_opportunity"]) - num(cases["actual_opportunity"])).abs()
        )

    scored_path = out / "prism_week5_stage12_scored_rows.csv"
    metric_path = out / "prism_week5_stage12_metrics.json"
    case_path = out / "prism_week5_stage12_casebook_link.csv"
    excl_path = out / "prism_week5_stage12_score_exclusions.csv"
    scored.to_csv(scored_path, index=False)
    cases.to_csv(case_path, index=False)
    pd.DataFrame(exclusions).to_csv(excl_path, index=False)

    support = {
        cell: {
            "candidate_rows": int(candidate["cell"].eq(cell).sum()),
            "scored_rows": int(scored["cell"].eq(cell).sum()),
            "score_rate": float(
                scored["cell"].eq(cell).sum() / max(candidate["cell"].eq(cell).sum(), 1)
            ),
        }
        for cell in CELLS
    }
    result = {
        "version": VERSION,
        "phase": "SCORE",
        "status": disposition,
        "candidate_row_digest_verified": digest,
        "metrics": metrics,
        "support": support,
        "casebook_rows": int(len(cases)),
        "casebook_scoreable_rows": int(cases["actual_opportunity"].notna().sum()) if len(cases) else 0,
        "score_exclusions": int(len(exclusions)),
        "target_week_outcomes_used_only_after_candidate_freeze": True,
        "week6_outcomes_read": 0,
        "sportsbook_predictors": 0,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "thresholds_fit": 0,
        "production_changed": False,
        "scored_row_digest": sha256_file(scored_path),
        "casebook_row_digest": sha256_file(case_path),
    }
    metric_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    b = sub.add_parser("build")
    b.add_argument("--player-state", type=Path, required=True)
    b.add_argument("--stage1-prior-universe", type=Path, required=True)
    b.add_argument("--official-inactives", type=Path, required=True)
    b.add_argument("--rb-identity-lock", type=Path, required=True)
    b.add_argument("--wrte-identity-lock", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.set_defaults(func=build)

    s = sub.add_parser("score")
    s.add_argument("--candidate-rows", type=Path, required=True)
    s.add_argument("--build-certificate", type=Path, required=True)
    s.add_argument("--rb-rush-outcomes", type=Path, required=True)
    s.add_argument("--rb-rec-outcomes", type=Path, required=True)
    s.add_argument("--wrte-outcomes", type=Path, required=True)
    s.add_argument("--casebook", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.set_defaults(func=score)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
