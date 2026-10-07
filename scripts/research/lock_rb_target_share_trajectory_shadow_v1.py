#!/usr/bin/env python3
"""Create immutable Week-5 RB Target Share Trajectory Shadow V1 lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import apply_te_r5p_entitlement
from scripts.modeling.wr_r15_entitlement_adapter_v1 import apply_wr_r15_entitlement
from scripts.player_form_v2 import _normalize_weekly, _to_pandas
from scripts.utils.canonical_names import canon_team

VERSION = "RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1"
SEASON = 2026
PRIOR_SEASON = 2025
WEEK = 5
RB_POS = {"RB", "HB", "FB", "TB"}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def clean(v):
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def team(v):
    try:
        t = canon_team(v)
        return "WAS" if t == "WSH" else t
    except Exception:
        return clean(v).upper()


def lower(x):
    y = _to_pandas(x).copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def regular_only(x):
    y = x.copy()
    c = "season_type" if "season_type" in y.columns else "game_type" if "game_type" in y.columns else None
    if c:
        s = y[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            y = y.loc[keep].copy()
    return y


def pos_family(v):
    p = str(v or "").upper().strip()
    if p in RB_POS or p.startswith("RB") or p.startswith("FB"):
        return "RB"
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p.startswith("QB"):
        return "QB"
    return p


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def assert_2026_pregame_stats_and_identity() -> pd.DataFrame:
    import nflreadpy as nfl

    raw = _to_pandas(nfl.load_player_stats(seasons=[SEASON], summary_level="week"))
    x = _normalize_weekly(raw, SEASON)
    bad = sorted(num(x.loc[num(x["week"]).ge(WEEK), "week"]).dropna().astype(int).unique().tolist())
    if bad:
        raise RuntimeError(f"Week-5-or-later weekly stats present; refuse prospective lock: {bad}")

    z = x[["season", "player_clean_key", "player_id"]].copy()
    z["player_id"] = z["player_id"].map(clean)
    z = z.loc[z["player_clean_key"].astype(str).ne("") & z["player_id"].ne("")].drop_duplicates()
    g = z.groupby("player_clean_key")["player_id"].agg(lambda s: sorted(set(map(str, s)))).reset_index()
    g["id_count"] = g["player_id"].map(len)
    g["receiver_id"] = g["player_id"].map(lambda v: v[0] if len(v) == 1 else "")
    return g[["player_clean_key", "receiver_id", "id_count"]]


def load_pbp() -> pd.DataFrame:
    import nflreadpy as nfl

    x = regular_only(lower(nfl.load_pbp(seasons=[SEASON])))
    for c in [
        "season", "week", "game_id", "posteam", "receiver_player_id",
        "pass_attempt", "sack", "two_point_attempt",
    ]:
        if c not in x.columns:
            x[c] = np.nan
    x["season"] = num(x["season"]).fillna(SEASON).astype(int)
    x["week"] = num(x["week"])
    bad = sorted(x.loc[x["week"].ge(WEEK), "week"].dropna().astype(int).unique().tolist())
    if bad:
        raise RuntimeError(f"Week-5-or-later PBP present; refuse prospective lock: {bad}")
    x = x.loc[x["week"].lt(WEEK)].copy()
    x["team"] = x["posteam"].map(team)
    x["receiver_id"] = x["receiver_player_id"].map(clean)
    raw = num(x["pass_attempt"]).fillna(0).eq(1)
    sack = num(x["sack"]).fillna(0).eq(1)
    two = num(x["two_point_attempt"]).fillna(0).eq(1)
    x["target_event"] = raw & ~sack & ~two & x["receiver_id"].ne("") & x["team"].ne("")
    return x[["season", "week", "game_id", "team", "receiver_id", "target_event"]]


def build_indexes(pbp: pd.DataFrame):
    games = (
        pbp.loc[pbp["team"].ne(""), ["season", "week", "game_id", "team"]]
        .drop_duplicates()
        .sort_values(["team", "week", "game_id"])
    )
    targets = pbp.loc[pbp["target_event"]].copy()
    team_targets = (
        targets.groupby(["season", "week", "game_id", "team"], as_index=False)
        .size()
        .rename(columns={"size": "team_targets"})
    )
    player_targets = (
        targets.groupby(["season", "week", "game_id", "team", "receiver_id"], as_index=False)
        .size()
        .rename(columns={"size": "player_targets"})
    )
    games = games.merge(
        team_targets,
        on=["season", "week", "game_id", "team"],
        how="left",
        validate="one_to_one",
    )
    games["team_targets"] = num(games["team_targets"]).fillna(0.0)
    team_idx = {
        str(tm): g.sort_values(["week", "game_id"]).copy()
        for tm, g in games.groupby("team", sort=False)
    }
    player_lookup = {
        (int(r.week), str(r.game_id), str(r.team), str(r.receiver_id)): float(r.player_targets)
        for r in player_targets.itertuples(index=False)
    }
    return team_idx, player_lookup


def trajectory_state(team_idx, player_lookup, tm: str, pid: str):
    g = team_idx.get(str(tm))
    if g is None:
        return None
    g = g.loc[num(g["week"]).lt(WEEK)].copy()
    if len(g) < 4:
        return None

    recent = g.tail(2).copy()
    earlier = g.iloc[:-2].copy()
    if len(earlier) < 2:
        return None

    recent_den = float(num(recent["team_targets"]).sum())
    earlier_den = float(num(earlier["team_targets"]).sum())
    if recent_den <= 0 or earlier_den <= 0:
        return None

    def player_sum(q):
        total = 0.0
        for r in q.itertuples(index=False):
            total += player_lookup.get(
                (int(r.week), str(r.game_id), str(r.team), str(pid)),
                0.0,
            )
        return total

    recent_num = player_sum(recent)
    earlier_num = player_sum(earlier)
    return {
        "recent2_share": recent_num / recent_den,
        "earlier_share": earlier_num / earlier_den,
        "trajectory_delta": recent_num / recent_den - earlier_num / earlier_den,
        "trajectory_feature_max_week": int(num(recent["week"]).max()),
        "prior_team_games": int(len(g)),
    }


def add_trajectory(final: pd.DataFrame):
    ids = assert_2026_pregame_stats_and_identity()
    pbp = load_pbp()
    team_idx, player_lookup = build_indexes(pbp)

    x = final.copy()
    x = x.merge(ids, on="player_clean_key", how="left", validate="many_to_one")
    x["receiver_id"] = x["receiver_id"].fillna("").astype(str)
    x["identity_ok"] = x["receiver_id"].ne("") & num(x["id_count"]).eq(1)

    rows = []
    violations = 0
    for r in x.itertuples(index=False):
        st = (
            trajectory_state(team_idx, player_lookup, str(r.team), str(r.receiver_id))
            if bool(r.identity_ok)
            else None
        )
        if st is None:
            rows.append({
                "recent2_share": np.nan,
                "earlier_share": np.nan,
                "trajectory_delta": 0.0,
                "trajectory_feature_max_week": np.nan,
                "prior_team_games": 0,
                "trajectory_available": False,
                "trajectory_route": "TRAJECTORY_UNAVAILABLE_NO_FIT_WEIGHT",
            })
        else:
            violations += int(st["trajectory_feature_max_week"] >= WEEK)
            st["trajectory_available"] = True
            st["trajectory_route"] = "TRAJECTORY_AVAILABLE"
            rows.append(st)

    y = pd.concat([x.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return y, violations


def apply_shadow(frame: pd.DataFrame):
    x = frame.copy()
    x["position_family"] = x["position"].map(pos_family)
    x["baseline_entitlement_tgt_share"] = num(x["entitlement_tgt_share"])
    x["shadow_entitlement_tgt_share"] = x["baseline_entitlement_tgt_share"].astype(float)
    x["trajectory_weight"] = (
        x["baseline_entitlement_tgt_share"]
        * np.exp(num(x["trajectory_delta"]).fillna(0.0))
    )

    audits = []
    rb = x.loc[x["position_family"].eq("RB")].copy()
    for (event_id, tm), g in rb.groupby(["event_id", "team"], sort=False):
        idx = list(g.index)
        pool = float(x.loc[idx, "baseline_entitlement_tgt_share"].sum())
        w = num(x.loc[idx, "trajectory_weight"]).to_numpy(float)
        if pool > 0 and w.sum() > 0:
            cand = pool * w / w.sum()
            cand[int(np.argmax(w))] += pool - float(cand.sum())
        else:
            cand = np.zeros(len(idx), float)
        base = num(x.loc[idx, "baseline_entitlement_tgt_share"]).to_numpy(float)
        x.loc[idx, "shadow_entitlement_tgt_share"] = cand
        audits.append({
            "event_id": event_id,
            "team": tm,
            "room": "RB_FB",
            "players": int(len(idx)),
            "baseline_pool": pool,
            "shadow_pool": float(cand.sum()),
            "pool_gap": abs(float(cand.sum()) - pool),
            "max_player_abs_change": float(np.max(np.abs(cand - base))) if len(idx) else 0.0,
        })

    x["entitlement_delta"] = (
        x["shadow_entitlement_tgt_share"] - x["baseline_entitlement_tgt_share"]
    )
    aud = pd.DataFrame(audits)

    if len(aud) and float(num(aud["pool_gap"]).max()) > 1e-12:
        raise RuntimeError(
            f"RB/FB room conservation failed: {aud.sort_values('pool_gap').tail().to_dict('records')}"
        )

    base_team = x.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].sum()
    shadow_team = x.groupby(["event_id", "team"])["shadow_entitlement_tgt_share"].sum()
    team_gap = float((shadow_team - base_team).abs().max()) if len(base_team) else 0.0
    if team_gap > 1e-12:
        raise RuntimeError(f"team modeled target mass changed: {team_gap}")

    non_rb = x.loc[~x["position_family"].eq("RB")]
    non_rb_delta = (
        float(non_rb["entitlement_delta"].abs().max()) if len(non_rb) else 0.0
    )
    if non_rb_delta > 1e-12:
        raise RuntimeError(f"non-RB/FB entitlement changed: {non_rb_delta}")

    return x, aud, team_gap, non_rb_delta


def canonical_lock_bytes(x: pd.DataFrame) -> bytes:
    cols = [
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "position", "position_family", "receiver_id",
        "identity_ok", "trajectory_available", "trajectory_route",
        "recent2_share", "earlier_share", "trajectory_delta",
        "trajectory_feature_max_week", "prior_team_games",
        "baseline_entitlement_tgt_share", "trajectory_weight",
        "shadow_entitlement_tgt_share", "entitlement_delta",
    ]
    y = (
        x[cols]
        .copy()
        .sort_values(["event_id", "team", "player_clean_key"], kind="mergesort")
        .reset_index(drop=True)
    )
    return y.to_csv(
        index=False,
        float_format="%.12f",
        lineterminator="\n",
    ).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    player_logs = read(a.player_logs, "player logs")
    team_weekly = read(a.team_weekly, "team weekly")
    schedule = read(a.schedule, "schedule")
    universe = read(a.universe, "Week-5 pregame universe")

    bundle = build_historical_context_bundle(
        player_logs=player_logs,
        team_weekly=team_weekly,
        pregame_universe=universe,
        schedule=schedule,
        season=SEASON,
        week=WEEK,
        prior_season=PRIOR_SEASON,
        injuries=None,
        weather=None,
    )
    metrics = build_mc_predictions(bundle, iterations=2000, seed=42 + WEEK)
    players = (
        metrics.sort_values(["event_id", "team", "player_clean_key"])
        .drop_duplicates(["event_id", "team", "player_clean_key"], keep="last")
        .copy()
    )

    baseline, _ = materialize_target_entitlement(players)
    te, _, te_audit = apply_te_r5p_entitlement(baseline)
    final, _, wr_audit = apply_wr_r15_entitlement(te)

    enriched, violations = add_trajectory(final)
    lock, room_audit, team_gap, non_rb_delta = apply_shadow(enriched)
    target = lock.loc[lock["position_family"].eq("RB")].copy()

    payload = canonical_lock_bytes(target)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    changed = int(target["entitlement_delta"].abs().gt(1e-12).sum())

    result = {
        "version": VERSION,
        "status": "WEEK5_PREGAME_RB_TARGET_SHARE_LOCK_FROZEN",
        "season": SEASON,
        "week": WEEK,
        "locked_players": int(len(target)),
        "locked_teams": int(target["team"].nunique()) if len(target) else 0,
        "trajectory_available_players": int(target["trajectory_available"].sum()),
        "trajectory_available_fraction": (
            float(target["trajectory_available"].mean()) if len(target) else 0.0
        ),
        "stable_id_coverage": (
            float(target["identity_ok"].mean()) if len(target) else 0.0
        ),
        "changed_players": changed,
        "changed_rooms": (
            int(room_audit["max_player_abs_change"].gt(1e-12).sum())
            if len(room_audit) else 0
        ),
        "median_abs_player_change": (
            float(target["entitlement_delta"].abs().median()) if len(target) else 0.0
        ),
        "max_abs_player_change": (
            float(target["entitlement_delta"].abs().max()) if len(target) else 0.0
        ),
        "max_rb_room_pool_gap": (
            float(num(room_audit["pool_gap"]).max()) if len(room_audit) else 0.0
        ),
        "max_team_modeled_target_mass_gap": team_gap,
        "max_non_rb_entitlement_delta": non_rb_delta,
        "same_or_future_feature_violations": int(violations),
        "row_csv_digest": digest,
        "te_parent_model_version": str(te_audit["model_version"]),
        "wr_parent_model_version": str(wr_audit["model_version"]),
        "r22_applicable_week5": False,
        "r26_applicable_week5": False,
        "sportsbook_inputs_used": 0,
        "week5_outcomes_read": 0,
        "parameters_fit": 0,
        "production_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (a.out_dir / "rb_target_share_trajectory_shadow_week5_lock.csv").write_bytes(payload)
    room_audit.to_csv(
        a.out_dir / "rb_target_share_trajectory_shadow_week5_rooms.csv",
        index=False,
        float_format="%.12f",
    )
    (a.out_dir / "rb_target_share_trajectory_shadow_week5_lock.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
