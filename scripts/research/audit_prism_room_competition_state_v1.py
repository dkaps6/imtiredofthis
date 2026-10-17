#!/usr/bin/env python3
"""PRISM Room Competition State V1 — source/parity audit.

Source-only. Builds named-player room competition state from official pregame
inactive identity plus strictly-prior offensive snaps. It does not attach
target-game outcomes, estimate opportunity transfer, fit a predictive model, or
change production.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.research import audit_player_role_elasticity_teammate_availability_v1 as role

VERSION = "PRISM_ROOM_COMPETITION_STATE_V1_SOURCE_AUDIT"
TARGET_SEASONS = (2024, 2025)
HISTORY_SEASONS = (2023, 2024, 2025)
EVENT_MIN = {"RB": 20, "WR": 40, "TE": 15}
PLAYER_MIN = {"RB": 30, "WR": 50, "TE": 20}
READY = {
    "RB": "RB_ROOM_COMPETITION_STATE_SOURCE_READY",
    "WR": "WR_ROOM_COMPETITION_STATE_SOURCE_READY",
    "TE": "TE_ROOM_COMPETITION_STATE_SOURCE_READY",
}
BLOCKED = {
    "RB": "RB_ROOM_COMPETITION_STATE_SOURCE_BLOCKED",
    "WR": "WR_ROOM_COMPETITION_STATE_SOURCE_BLOCKED",
    "TE": "TE_ROOM_COMPETITION_STATE_SOURCE_BLOCKED",
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_all_rosters(source_meta: list[dict]) -> pd.DataFrame:
    parts = []
    for season in HISTORY_SEASONS:
        raw = m78.load_weekly_rosters(season, source_meta)
        parts.append(role.normalize_roster(raw, season))
    out = pd.concat(parts, ignore_index=True)
    out["season"] = num(out["season"]).astype(int)
    out["week"] = num(out["week"]).astype(int)
    out["player_id"] = out["player_id"].astype("string").fillna("").str.strip()
    return out


def snap_team_history(snaps: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(
            columns=["season", "week", "team", "family", "player_id", "off_pct", "ordinal"]
        )
    rr = rosters.loc[
        rosters["player_id"].astype(str).ne(""),
        ["season", "week", "team", "family", "player_id"],
    ].drop_duplicates(["season", "week", "player_id", "team"])
    q = snaps.merge(rr, on=["season", "week", "player_id"], how="inner")
    q["off_pct"] = num(q["off_pct"])
    q["ordinal"] = q["season"].astype(int) * 100 + q["week"].astype(int)
    q = q.dropna(subset=["off_pct"])
    return q.sort_values(["player_id", "team", "season", "week"]).reset_index(drop=True)


def prior_snap_mean3(
    snap_index: dict[tuple[str, str], pd.DataFrame],
    player_id: str,
    team: str,
    season: int,
    week: int,
) -> tuple[float, int, int]:
    target = int(season) * 100 + int(week)
    g = snap_index.get((str(player_id), str(team)))
    if g is None or g.empty:
        return np.nan, 0, 0
    q = g.loc[g["ordinal"].lt(target)].sort_values("ordinal").tail(3)
    if q.empty:
        return np.nan, 0, 0
    bad = int((q["ordinal"] >= target).sum())
    vals = num(q["off_pct"]).dropna()
    if vals.empty:
        return np.nan, 0, bad
    return float(vals.mean()), int(len(vals)), bad


def attach_prior_snap_state(rows: pd.DataFrame, snap_hist: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    idx = {
        (str(pid), str(team)): g
        for (pid, team), g in snap_hist.groupby(["player_id", "team"], sort=False)
    }
    means, games, bad = [], [], []
    for r in out.itertuples(index=False):
        mean, n, violations = prior_snap_mean3(
            idx,
            str(r.player_id),
            str(r.team),
            int(r.season),
            int(r.week),
        )
        means.append(mean)
        games.append(n)
        bad.append(violations)
    out["prior_snap_mean3"] = means
    out["prior_snap_games"] = games
    out["chronology_violations"] = bad
    return out


def map_inactive_players(
    inactive_rows: pd.DataFrame,
    target_roster: pd.DataFrame,
) -> pd.DataFrame:
    q = inactive_rows.loc[inactive_rows["family"].isin(role.FAMILY_POS)].copy()
    rr = target_roster[
        ["season", "week", "team", "player_name_key", "player_id", "position", "family"]
    ].copy()
    rr = rr.rename(columns={"family": "roster_family", "position": "roster_position"})
    out = q.merge(
        rr,
        left_on=["season", "week", "team", "inactive_name_key"],
        right_on=["season", "week", "team", "player_name_key"],
        how="left",
        validate="many_to_one",
    )
    out["player_id"] = out["player_id"].astype("string").fillna("").str.strip()
    out["stable_inactive_identity"] = out["player_id"].ne("").astype(int)
    out["family_match"] = (out["family"] == out["roster_family"]).astype(int)
    return out


def aggregate_room_state(
    active: pd.DataFrame,
    inactive: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["season", "week", "team", "family"]
    a = (
        active.groupby(keys, as_index=False)
        .agg(
            active_room_player_count=("player_name_key", "nunique"),
            active_room_prior_snap_mass=("prior_snap_mean3", "sum"),
            active_with_prior_snap=("prior_snap_mean3", lambda s: int(s.notna().sum())),
        )
    )
    i = (
        inactive.groupby(keys, as_index=False)
        .agg(
            inactive_room_player_count=("inactive_name_key", "nunique"),
            inactive_room_prior_snap_mass=("prior_snap_mean3", "sum"),
            inactive_with_prior_snap=("prior_snap_mean3", lambda s: int(s.notna().sum())),
        )
    )
    room = a.merge(i, on=keys, how="outer").fillna(
        {
            "active_room_player_count": 0,
            "active_room_prior_snap_mass": 0.0,
            "active_with_prior_snap": 0,
            "inactive_room_player_count": 0,
            "inactive_room_prior_snap_mass": 0.0,
            "inactive_with_prior_snap": 0,
        }
    )
    room["total_observed_room_prior_snap_mass"] = (
        num(room["active_room_prior_snap_mass"]).fillna(0)
        + num(room["inactive_room_prior_snap_mass"]).fillna(0)
    )
    denom = room["total_observed_room_prior_snap_mass"].replace(0, np.nan)
    room["competition_release_fraction"] = (
        num(room["inactive_room_prior_snap_mass"]) / denom
    )
    room["inactive_snap_complete"] = (
        room["inactive_with_prior_snap"].astype(int)
        == room["inactive_room_player_count"].astype(int)
    )
    return room


def build_state_rows(
    active: pd.DataFrame,
    inactive: pd.DataFrame,
) -> pd.DataFrame:
    room = aggregate_room_state(active, inactive)
    keys = ["season", "week", "team", "family"]
    out = active.merge(room, on=keys, how="left", validate="many_to_one")
    denom = num(out["active_room_prior_snap_mass"]).replace(0, np.nan)
    out["player_active_room_snap_share"] = num(out["prior_snap_mean3"]) / denom
    out["finite_room_state"] = (
        out["prior_snap_mean3"].notna()
        & num(out["active_room_prior_snap_mass"]).gt(0)
        & num(out["inactive_room_prior_snap_mass"]).gt(0)
        & num(out["total_observed_room_prior_snap_mass"]).gt(0)
    )
    return out


def metrics_for(
    season: int,
    family: str,
    inactive: pd.DataFrame,
    state: pd.DataFrame,
) -> dict:
    iq = inactive.loc[(inactive["season"].eq(season)) & (inactive["family"].eq(family))]
    sq = state.loc[(state["season"].eq(season)) & (state["family"].eq(family))]
    event_games = int(iq[["season", "week", "team"]].drop_duplicates().shape[0])
    inactive_identity = float(iq["stable_inactive_identity"].mean()) if len(iq) else 0.0
    inactive_snap = float(iq["prior_snap_mean3"].notna().mean()) if len(iq) else 0.0
    active_identity = float(sq["stable_roster_identity"].mean()) if len(sq) else 0.0
    active_snap = float(sq["prior_snap_mean3"].notna().mean()) if len(sq) else 0.0
    finite = float(sq["finite_room_state"].mean()) if len(sq) else 0.0
    chronology = int(sq["chronology_violations"].sum()) + int(
        iq["chronology_violations"].sum()
    )
    return {
        "season": int(season),
        "family": family,
        "inactive_rows": int(len(iq)),
        "active_candidate_rows": int(len(sq)),
        "event_team_games": event_games,
        "distinct_active_players": int(
            sq.loc[sq["stable_roster_identity"].eq(1), "player_id"].nunique()
        ) if len(sq) else 0,
        "inactive_stable_identity": inactive_identity,
        "active_stable_identity": active_identity,
        "inactive_prior_snap_coverage": inactive_snap,
        "active_prior_snap_coverage": active_snap,
        "finite_room_state_coverage": finite,
        "chronology_violations": chronology,
    }


def gates(m: dict) -> dict[str, bool]:
    fam = str(m["family"])
    return {
        "inactive_identity_ge_90": m["inactive_stable_identity"] >= 0.90,
        "active_identity_ge_90": m["active_stable_identity"] >= 0.90,
        "inactive_prior_snap_ge_85": m["inactive_prior_snap_coverage"] >= 0.85,
        "active_prior_snap_ge_90": m["active_prior_snap_coverage"] >= 0.90,
        "finite_room_state_ge_85": m["finite_room_state_coverage"] >= 0.85,
        "event_support": m["event_team_games"] >= EVENT_MIN[fam],
        "distinct_active_support": m["distinct_active_players"] >= PLAYER_MIN[fam],
        "chronology_zero": m["chronology_violations"] == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teamweeks, inactive_rows = role.parse_inactives(args.inactives)
    source_meta: list[dict] = []
    all_roster = load_all_rosters(source_meta)
    target_roster = all_roster.loc[all_roster["season"].isin(TARGET_SEASONS)].copy()
    snaps = role.load_snaps(source_meta)
    snap_hist = snap_team_history(snaps, all_roster)

    active = role.build_successor_rows(target_roster, inactive_rows)
    active = attach_prior_snap_state(active, snap_hist)
    inactive = map_inactive_players(inactive_rows, target_roster)
    inactive = attach_prior_snap_state(inactive, snap_hist)
    state = build_state_rows(active, inactive)

    metric_rows = []
    family_ready = {}
    position_dispositions = {}
    for fam in ("RB", "WR", "TE"):
        ok = True
        for season in TARGET_SEASONS:
            m = metrics_for(season, fam, inactive, state)
            g = gates(m)
            metric_rows.append({**m, **{f"gate_{k}": bool(v) for k, v in g.items()}})
            ok = ok and all(g.values())
        family_ready[fam] = ok
        position_dispositions[fam] = READY[fam] if ok else BLOCKED[fam]

    if all(family_ready.values()):
        disposition = "PRISM_ROOM_COMPETITION_STATE_SOURCE_READY_ALL"
    elif any(family_ready.values()):
        disposition = "PRISM_ROOM_COMPETITION_STATE_SOURCE_READY_PARTIAL"
    else:
        disposition = "PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED"

    state_payload = state.sort_values(
        ["season", "week", "team", "family", "player_name_key"]
    ).to_csv(index=False, float_format="%.12f", lineterminator="\n").encode()

    result = {
        "version": VERSION,
        "disposition": disposition,
        "position_dispositions": position_dispositions,
        "m78_teamweek_rows": int(len(teamweeks)),
        "inactive_skill_rows": int(len(inactive)),
        "active_candidate_rows": int(len(state)),
        "metric_rows": metric_rows,
        "state_rows_sha256": "sha256:" + hashlib.sha256(state_payload).hexdigest(),
        "target_game_outcomes_used": 0,
        "season_2026_outcomes_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "predictive_models_fit": 0,
        "projection_values_changed": 0,
        "production_changed": False,
    }

    pd.DataFrame(source_meta).to_csv(
        args.out_dir / "prism_room_competition_source_snapshots.csv", index=False
    )
    inactive.to_csv(
        args.out_dir / "prism_room_competition_inactive_rows.csv", index=False
    )
    state.to_csv(
        args.out_dir / "prism_room_competition_state_rows.csv",
        index=False,
        float_format="%.12f",
    )
    pd.DataFrame(metric_rows).to_csv(
        args.out_dir / "prism_room_competition_source_metrics.csv", index=False
    )
    (args.out_dir / "prism_room_competition_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
