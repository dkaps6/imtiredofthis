#!/usr/bin/env python3
"""PRISM RB Team-Context Carry Identity Persistence V1.

Frozen predictive test for source-certified in-season RB team changes. Compares
canonical current-team RB rushing entitlement against the named player's own
strictly-prior previous-team carry-share identity while preserving finite room
mass. No yards, efficiency, touchdowns, sportsbook, or 2026 outcomes.
"""
from __future__ import annotations

import argparse
import json
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.audit_player_opportunity_share_stability_v1 import build_weekly_universe
from scripts.research.audit_player_opportunity_share_stability_2023_holdout_v1 import (
    build_2023_weekly_universe,
)

VERSION = "PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1"
FREEZE_COMMIT = "668bd3266b6e6c131875ef245ce037b1a9925be7"
SEASONS = (2024, 2025)
WEEKS = tuple(range(5, 19))
RB_POS = {"RB", "HB", "TB", "FB"}
MIN_PRIOR = 4
MAX_PRIOR = 8
BOOT_REPS = 5000
BOOT_SEED = 20261018
TOL = 1e-10


def num(s):
    return pd.to_numeric(s, errors="coerce")


def txt(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.upper() in {"", "NAN", "NONE", "<NA>"} else s


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string").str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False).astype(bool)
    )


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def prepare_transitions(path: Path) -> pd.DataFrame:
    x = read_csv(path, "team-context transition parent")
    req = {
        "season", "week", "team", "stable_id", "family",
        "previous_rostered_team", "transition_kind",
        "prior_team_support_ge4",
    }
    miss = req - set(x.columns)
    if miss:
        raise RuntimeError(f"transition parent missing {sorted(miss)}")
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["previous_rostered_team"] = x["previous_rostered_team"].map(canon_team)
    x["stable_id"] = x["stable_id"].map(txt)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["transition_kind"] = x["transition_kind"].astype("string").fillna("").str.strip()
    x["prior_team_support_ge4"] = as_bool(x["prior_team_support_ge4"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].isin(WEEKS)
        & x["family"].eq("RB")
        & x["stable_id"].ne("")
        & x["transition_kind"].eq("INSEASON_TEAM_CHANGE")
        & x["prior_team_support_ge4"]
        & x["previous_rostered_team"].ne("")
        & x["team"].ne(x["previous_rostered_team"])
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "stable_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate eligible transition identities")
    return x.sort_values(keys, kind="mergesort").reset_index(drop=True)


def prepare_authority(path: Path) -> pd.DataFrame:
    x = read_csv(path, "Stage-2 source detail")
    req = {
        "season", "week", "team", "family", "player_id", "cell",
        "authority_q", "authority_covered",
    }
    miss = req - set(x.columns)
    if miss:
        raise RuntimeError(f"Stage-2 source detail missing {sorted(miss)}")
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x["team"] = x["team"].map(canon_team)
    x["player_id"] = x["player_id"].map(txt)
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()
    x["cell"] = x["cell"].astype("string").fillna("").str.strip()
    x["authority_q"] = num(x["authority_q"])
    x["authority_covered"] = as_bool(x["authority_covered"])
    x = x.loc[
        x["season"].isin(SEASONS)
        & x["week"].isin(WEEKS)
        & x["family"].eq("RB")
        & x["cell"].eq("RB_RUSHING")
        & x["player_id"].ne("")
        & x["authority_covered"]
        & x["authority_q"].notna()
        & np.isfinite(x["authority_q"])
        & x["authority_q"].ge(0)
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    keys = ["season", "week", "team", "player_id"]
    if x.duplicated(keys).any():
        raise RuntimeError("duplicate canonical RB rushing authority identity")
    return x.sort_values(keys, kind="mergesort").reset_index(drop=True)


def build_counts() -> pd.DataFrame:
    u = pd.concat(
        [build_2023_weekly_universe(), build_weekly_universe()],
        ignore_index=True,
        sort=False,
    )
    req = {
        "season", "week", "team", "player_identity_key", "position",
        "rushes", "team_rushes",
    }
    miss = req - set(u.columns)
    if miss:
        raise RuntimeError(f"count universe missing {sorted(miss)}")
    u["season"] = num(u["season"])
    u["week"] = num(u["week"])
    u["team"] = u["team"].map(canon_team)
    u["player_identity_key"] = u["player_identity_key"].map(txt)
    u["position"] = u["position"].astype("string").fillna("").str.upper().str.strip()
    u["rushes"] = num(u["rushes"])
    u["team_rushes"] = num(u["team_rushes"])
    u = u.loc[
        u["season"].isin((2023, 2024, 2025))
        & u["week"].between(1, 18, inclusive="both")
        & u["player_identity_key"].ne("")
    ].copy()
    u["season"] = u["season"].astype(int)
    u["week"] = u["week"].astype(int)
    keys = ["season", "week", "team", "player_identity_key"]
    if u.duplicated(keys).any():
        raise RuntimeError("duplicate count player-week identity")
    return u.sort_values(keys, kind="mergesort").reset_index(drop=True)


def ordinal(season: int, week: int) -> int:
    return int(season) * 100 + int(week)


def old_team_anchor(
    counts: pd.DataFrame,
    *,
    pid: str,
    prev_team: str,
    target_season: int,
    target_week: int,
) -> tuple[float | None, list[dict]]:
    target_ord = ordinal(target_season, target_week)
    q = counts.loc[
        counts["player_identity_key"].eq(pid)
        & counts["team"].eq(prev_team)
        & counts["position"].isin(RB_POS)
        & counts["team_rushes"].notna()
        & counts["team_rushes"].gt(0)
        & counts["rushes"].notna()
        & counts["rushes"].ge(0)
        & counts["rushes"].le(counts["team_rushes"])
    ].copy()
    q["ordinal"] = q["season"].astype(int) * 100 + q["week"].astype(int)
    q = q.loc[q["ordinal"].lt(target_ord)].sort_values(
        ["ordinal"], kind="mergesort"
    ).tail(MAX_PRIOR)
    if len(q) < MIN_PRIOR:
        return None, []
    q["carry_share"] = q["rushes"] / q["team_rushes"]
    vals = q["carry_share"].to_numpy(float)
    if not np.isfinite(vals).all() or ((vals < 0) | (vals > 1)).any():
        raise RuntimeError("invalid old-team carry-share history")
    hist = [
        {
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "rushes": float(r.rushes), "team_rushes": float(r.team_rushes),
            "carry_share": float(r.carry_share),
        }
        for r in q.itertuples(index=False)
    ]
    return float(vals.mean()), hist


def room_probs(authority: pd.DataFrame, season: int, week: int, team: str) -> pd.DataFrame:
    r = authority.loc[
        authority["season"].eq(season)
        & authority["week"].eq(week)
        & authority["team"].eq(team)
    ].copy()
    if r.empty:
        return r
    total = float(r["authority_q"].sum())
    residual = 1.0 - total
    if total > 1.0 + TOL or residual < -TOL:
        raise RuntimeError(f"baseline room mass invalid {season} W{week} {team}: {total}")
    residual = max(0.0, residual)
    return r


def exact_log_multinomial(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or (y < 0).any() or not np.isfinite(p).all():
        return np.nan
    if abs(float(p.sum()) - 1.0) > TOL or (p < 0).any():
        return np.nan
    if ((p <= 0) & (y > 0)).any():
        return -np.inf
    n = int(y.sum())
    out = lgamma(n + 1) - sum(lgamma(int(v) + 1) for v in y)
    for yi, pi in zip(y, p):
        if yi > 0:
            out += int(yi) * float(np.log(pi))
    return float(out)


def score(
    transitions: pd.DataFrame,
    authority: pd.DataFrame,
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    count_map = {
        (int(r.season), int(r.week), str(r.team), str(r.player_identity_key)): r
        for r in counts.itertuples(index=False)
    }

    anchors = {}
    history_rows = []
    source_exclusions = {}
    stable_link_attempts = 0
    stable_link_hits = 0

    def excl(reason: str):
        source_exclusions[reason] = source_exclusions.get(reason, 0) + 1

    for e in transitions.itertuples(index=False):
        key = (int(e.season), int(e.week), str(e.team), str(e.stable_id))
        stable_link_attempts += 1
        target = count_map.get(key)
        if target is not None:
            stable_link_hits += 1
        if target is None:
            excl("target_count_identity_missing")
            continue

        base = authority.loc[
            authority["season"].eq(int(e.season))
            & authority["week"].eq(int(e.week))
            & authority["team"].eq(str(e.team))
            & authority["player_id"].eq(str(e.stable_id))
        ]
        if len(base) != 1:
            excl("canonical_authority_not_unique")
            continue
        qbase = float(base.iloc[0]["authority_q"])
        if not (np.isfinite(qbase) and 0 < qbase < 1):
            excl("canonical_authority_not_positive")
            continue

        n = float(target.team_rushes) if pd.notna(target.team_rushes) else np.nan
        y = float(target.rushes) if pd.notna(target.rushes) else np.nan
        if not (np.isfinite(n) and n > 0 and np.isfinite(y) and 0 <= y <= n):
            excl("invalid_target_carry_count")
            continue

        anchor, hist = old_team_anchor(
            counts,
            pid=str(e.stable_id),
            prev_team=str(e.previous_rostered_team),
            target_season=int(e.season),
            target_week=int(e.week),
        )
        if anchor is None:
            excl("old_team_valid_carry_history_lt4")
            continue

        akey = (int(e.season), int(e.week), str(e.team), str(e.stable_id))
        anchors[akey] = {
            "anchor": float(anchor),
            "q_base": qbase,
            "target_rushes": int(round(y)),
            "team_rushes": int(round(n)),
            "previous_team": str(e.previous_rostered_team),
        }
        for h in hist:
            history_rows.append({
                "target_season": int(e.season),
                "target_week": int(e.week),
                "target_team": str(e.team),
                "stable_id": str(e.stable_id),
                "previous_team": str(e.previous_rostered_team),
                **{f"prior_{k}": v for k, v in h.items()},
            })

    scored = []
    rooms = []

    room_keys = sorted(set((k[0], k[1], k[2]) for k in anchors))
    for season, week, team in room_keys:
        room = room_probs(authority, season, week, team)
        if room.empty:
            continue

        base_q = {
            str(r.player_id): float(r.authority_q)
            for r in room.itertuples(index=False)
            if np.isfinite(float(r.authority_q)) and float(r.authority_q) >= 0
        }
        residual_base = max(0.0, 1.0 - float(sum(base_q.values())))

        trans_ids = sorted(
            pid for (s, w, t, pid) in anchors
            if (s, w, t) == (season, week, team)
        )
        raw = np.array([anchors[(season, week, team, pid)]["anchor"] for pid in trans_ids], dtype=float)
        raw_sum = float(raw.sum())
        cand_trans = raw.copy()
        if raw_sum > 0.95:
            cand_trans *= 0.95 / raw_sum
        t_base = float(sum(base_q.get(pid, 0.0) for pid in trans_ids))
        t_cand = float(cand_trans.sum())
        r_base = 1.0 - t_base
        r_cand = 1.0 - t_cand
        if r_base <= 0:
            raise RuntimeError(f"nonpositive baseline remainder {season} W{week} {team}")
        scale = r_cand / r_base

        cand_q = {}
        trans_set = set(trans_ids)
        for pid, q in base_q.items():
            if pid not in trans_set:
                cand_q[pid] = q * scale
        for pid, q in zip(trans_ids, cand_trans):
            cand_q[pid] = float(q)
        residual_cand = residual_base * scale

        base_mass = float(sum(base_q.values()) + residual_base)
        cand_mass = float(sum(cand_q.values()) + residual_cand)
        if abs(base_mass - 1.0) > TOL or abs(cand_mass - 1.0) > TOL:
            raise RuntimeError(f"room mass drift {season} W{week} {team}")

        # Room score: canonical-positive RBs are explicit categories. Carries
        # by canonical-zero/uncovered rushers remain in the anonymous residual.
        explicit_ids = sorted(pid for pid, q in base_q.items() if q > 0)
        nvals = []
        yvals = []
        for pid in explicit_ids:
            r = count_map.get((season, week, team, pid))
            if r is None or pd.isna(r.team_rushes) or pd.isna(r.rushes):
                continue
            nvals.append(int(round(float(r.team_rushes))))
            yvals.append((pid, int(round(float(r.rushes)))))
        if not nvals or len(set(nvals)) != 1:
            room_base_log = np.nan
            room_cand_log = np.nan
            room_improvement = np.nan
        else:
            N = nvals[0]
            ymap = dict(yvals)
            ys = [ymap.get(pid, 0) for pid in explicit_ids]
            residual_y = N - int(sum(ys))
            if residual_y < 0:
                raise RuntimeError(f"room actual counts exceed team rushes {season} W{week} {team}")
            yvec = np.array(ys + [residual_y], dtype=int)
            pbase = np.array([base_q[pid] for pid in explicit_ids] + [residual_base], dtype=float)
            pcand = np.array([cand_q[pid] for pid in explicit_ids] + [residual_cand], dtype=float)
            room_base_log = exact_log_multinomial(yvec, pbase)
            room_cand_log = exact_log_multinomial(yvec, pcand)
            if np.isfinite(room_base_log) and np.isfinite(room_cand_log):
                room_improvement = room_cand_log - room_base_log
            elif room_base_log == -np.inf and np.isfinite(room_cand_log):
                room_improvement = np.inf
            elif np.isfinite(room_base_log) and room_cand_log == -np.inf:
                room_improvement = -np.inf
            else:
                room_improvement = np.nan

        rooms.append({
            "season": season, "week": week, "team": team,
            "transition_players": len(trans_ids),
            "baseline_mass": base_mass, "candidate_mass": cand_mass,
            "baseline_residual": residual_base, "candidate_residual": residual_cand,
            "room_baseline_log_score": room_base_log,
            "room_candidate_log_score": room_cand_log,
            "room_log_score_improvement": room_improvement,
        })

        for pid, qcand in zip(trans_ids, cand_trans):
            a = anchors[(season, week, team, pid)]
            actual = float(a["target_rushes"]) / float(a["team_rushes"])
            qbase = float(a["q_base"])
            ae_base = abs(qbase - actual)
            ae_cand = abs(float(qcand) - actual)
            count_base = abs(float(a["team_rushes"]) * qbase - float(a["target_rushes"]))
            count_cand = abs(float(a["team_rushes"]) * float(qcand) - float(a["target_rushes"]))
            scored.append({
                "season": season, "week": week, "team": team,
                "stable_id": pid, "previous_team": a["previous_team"],
                "team_rushes": a["team_rushes"], "rushes": a["target_rushes"],
                "actual_share": actual,
                "canonical_q_base": qbase,
                "old_team_identity_q": float(a["anchor"]),
                "candidate_q": float(qcand),
                "share_ae_base": ae_base,
                "share_ae_candidate": ae_cand,
                "share_ae_improvement": ae_base - ae_cand,
                "count_ae_base": count_base,
                "count_ae_candidate": count_cand,
                "count_ae_improvement": count_base - count_cand,
            })

    scored_columns = [
        "season", "week", "team", "stable_id", "previous_team",
        "team_rushes", "rushes", "actual_share", "canonical_q_base",
        "old_team_identity_q", "candidate_q", "share_ae_base",
        "share_ae_candidate", "share_ae_improvement", "count_ae_base",
        "count_ae_candidate", "count_ae_improvement",
    ]
    room_columns = [
        "season", "week", "team", "transition_players",
        "baseline_mass", "candidate_mass", "baseline_residual",
        "candidate_residual", "room_baseline_log_score",
        "room_candidate_log_score", "room_log_score_improvement",
    ]
    scored_df = pd.DataFrame(scored, columns=scored_columns)
    rooms_df = pd.DataFrame(rooms, columns=room_columns)
    audit = {
        "transition_source_rows": int(len(transitions)),
        "stable_link_attempts": int(stable_link_attempts),
        "stable_link_hits": int(stable_link_hits),
        "stable_link_rate": float(stable_link_hits / stable_link_attempts) if stable_link_attempts else 0.0,
        "source_exclusions": dict(sorted(source_exclusions.items())),
        "scored_events": int(len(scored_df)),
        "scored_rooms": int(len(rooms_df)),
        "history_rows": int(len(history_rows)),
    }
    return scored_df, rooms_df, {"audit": audit, "history_rows": history_rows}


def bootstrap_players(z: pd.DataFrame) -> dict:
    p = z.groupby("stable_id", sort=False)["share_ae_improvement"].mean().astype(float)
    vals = p.to_numpy(float)
    if len(vals) < 2:
        return {"reps": 0, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.empty(BOOT_REPS, dtype=float)
    for i in range(BOOT_REPS):
        boots[i] = float(rng.choice(vals, size=len(vals), replace=True).mean())
    return {
        "reps": BOOT_REPS,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def summarize(scored: pd.DataFrame, rooms: pd.DataFrame, season: int, confirmation: bool) -> dict:
    z = scored.loc[scored["season"].eq(season)].copy()
    r = rooms.loc[rooms["season"].eq(season)].copy()
    if z.empty:
        out = {
            "season": season, "eligible_events": 0, "named_players": 0,
            "equal_player_share_ae_improvement": np.nan,
            "median_player_share_ae_improvement": np.nan,
            "improved_player_share": np.nan, "tied_player_share": np.nan,
            "worsened_player_share": np.nan,
            "equal_player_count_ae_improvement": np.nan,
            "mean_room_log_score_improvement": np.nan,
        }
        if confirmation:
            out["player_bootstrap"] = {"reps": 0, "ci_low": np.nan, "ci_high": np.nan}
        return out

    pg = z.groupby("stable_id").agg(
        share=("share_ae_improvement", "mean"),
        count=("count_ae_improvement", "mean"),
        events=("share_ae_improvement", "size"),
    )
    eps = 1e-12
    imp = pg["share"] > eps
    wors = pg["share"] < -eps
    tie = ~(imp | wors)

    finite_room = num(r["room_log_score_improvement"]).replace([np.inf, -np.inf], np.nan).dropna()
    room_mean = float(finite_room.mean()) if len(finite_room) else np.nan

    out = {
        "season": season,
        "eligible_events": int(len(z)),
        "named_players": int(len(pg)),
        "equal_player_share_ae_improvement": float(pg["share"].mean()),
        "median_player_share_ae_improvement": float(pg["share"].median()),
        "improved_player_share": float(imp.mean()),
        "tied_player_share": float(tie.mean()),
        "worsened_player_share": float(wors.mean()),
        "equal_player_count_ae_improvement": float(pg["count"].mean()),
        "mean_room_log_score_improvement": room_mean,
        "finite_room_score_rows": int(len(finite_room)),
        "target_weeks": sorted(int(v) for v in z["week"].unique()),
        "previous_teams": int(z["previous_team"].nunique()),
        "current_teams": int(z["team"].nunique()),
    }
    if confirmation:
        out["player_bootstrap"] = bootstrap_players(z)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-context-events", type=Path, required=True)
    ap.add_argument("--stage2-detail", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    transitions = prepare_transitions(a.team_context_events)
    authority = prepare_authority(a.stage2_detail)
    counts = build_counts()

    scored, rooms, extra = score(transitions, authority, counts)
    hist = pd.DataFrame(extra["history_rows"])

    dev = summarize(scored, rooms, 2024, confirmation=False)
    conf = summarize(scored, rooms, 2025, confirmation=True)

    source_gates = {
        "stable_count_link_rate_ge_099": extra["audit"]["stable_link_rate"] >= 0.99,
        "chronology_violations_zero": True,
        "room_mass_violations_zero": bool(
            len(rooms) > 0
            and (rooms["baseline_mass"].sub(1).abs() <= TOL).all()
            and (rooms["candidate_mass"].sub(1).abs() <= TOL).all()
        ),
        "old_team_history_min4_all_scored": bool(
            len(scored) == 0 or
            hist.groupby(["target_season","target_week","target_team","stable_id"]).size().min() >= MIN_PRIOR
        ),
        "outcomes_2026_zero": True,
        "sportsbook_oddsapi_zero": True,
        "production_changed_false": True,
    }

    dev_gates = {
        "eligible_events_ge_20": dev["eligible_events"] >= 20,
        "named_players_ge_15": dev["named_players"] >= 15,
        "equal_player_share_ae_improvement_positive": bool(
            np.isfinite(dev["equal_player_share_ae_improvement"])
            and dev["equal_player_share_ae_improvement"] > 0
        ),
        "median_player_share_ae_improvement_nonnegative": bool(
            np.isfinite(dev["median_player_share_ae_improvement"])
            and dev["median_player_share_ae_improvement"] >= 0
        ),
        "improved_player_share_gt_worsened": bool(
            np.isfinite(dev["improved_player_share"])
            and dev["improved_player_share"] > dev["worsened_player_share"]
        ),
        "equal_player_count_ae_improvement_nonnegative": bool(
            np.isfinite(dev["equal_player_count_ae_improvement"])
            and dev["equal_player_count_ae_improvement"] >= 0
        ),
        "room_log_score_not_harmful": bool(
            np.isfinite(dev["mean_room_log_score_improvement"])
            and dev["mean_room_log_score_improvement"] >= 0
        ),
    }

    conf_gates = {
        "eligible_events_ge_20": conf["eligible_events"] >= 20,
        "named_players_ge_15": conf["named_players"] >= 15,
        "equal_player_share_ae_improvement_positive": bool(
            np.isfinite(conf["equal_player_share_ae_improvement"])
            and conf["equal_player_share_ae_improvement"] > 0
        ),
        "median_player_share_ae_improvement_nonnegative": bool(
            np.isfinite(conf["median_player_share_ae_improvement"])
            and conf["median_player_share_ae_improvement"] >= 0
        ),
        "improved_player_share_gt_worsened": bool(
            np.isfinite(conf["improved_player_share"])
            and conf["improved_player_share"] > conf["worsened_player_share"]
        ),
        "equal_player_count_ae_improvement_nonnegative": bool(
            np.isfinite(conf["equal_player_count_ae_improvement"])
            and conf["equal_player_count_ae_improvement"] >= 0
        ),
        "room_log_score_not_harmful": bool(
            np.isfinite(conf["mean_room_log_score_improvement"])
            and conf["mean_room_log_score_improvement"] >= 0
        ),
        "player_bootstrap_5000": conf.get("player_bootstrap", {}).get("reps", 0) == BOOT_REPS,
        "player_bootstrap_ci_low_positive": bool(
            np.isfinite(conf.get("player_bootstrap", {}).get("ci_low", np.nan))
            and conf["player_bootstrap"]["ci_low"] > 0
        ),
    }

    source_ok = bool(all(source_gates.values()))
    dev_ok = bool(source_ok and all(dev_gates.values()))
    qualified = bool(dev_ok and all(conf_gates.values()))

    if not source_ok:
        disposition = "PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED"
    elif qualified:
        disposition = "PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_QUALIFIED"
    else:
        disposition = "PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_CLOSED"

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "team_context_parent_run": 39182076127,
        "team_context_parent_artifact": 12156204035,
        "team_context_parent_digest": "sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8",
        "stage2_parent_run": 38955368643,
        "stage2_parent_artifact": 12070806828,
        "stage2_parent_digest": "sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83",
        "disposition": disposition,
        "source_audit": extra["audit"],
        "source_gates": source_gates,
        "development_2024": dev,
        "development_gates": dev_gates,
        "confirmation_2025": conf,
        "confirmation_gates": conf_gates,
        "old_team_prior_min": MIN_PRIOR,
        "old_team_prior_max": MAX_PRIOR,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "rushing_yards_scored": False,
        "touchdowns_scored": False,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    scored.to_csv(a.out_dir/"rb_team_context_carry_identity_scored_rows.csv", index=False, float_format="%.12f")
    rooms.to_csv(a.out_dir/"rb_team_context_carry_identity_room_rows.csv", index=False, float_format="%.12f")
    hist.to_csv(a.out_dir/"rb_team_context_carry_identity_prior_history.csv", index=False, float_format="%.12f")
    (a.out_dir/"rb_team_context_carry_identity_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
