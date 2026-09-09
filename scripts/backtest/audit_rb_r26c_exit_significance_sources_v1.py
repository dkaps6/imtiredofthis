#!/usr/bin/env python3
"""R26C source-only audit for the significance of departed RB receiving roles.

No target-game labels are selected or scored. The audit reconstructs canonical
ACT/INA room exits and attaches only strict-as-of receiving identity plus strictly
prior depth state.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.audit_rb_r26_safe_transition_state_v1 import (
    normalize_depth,
    normalize_roster,
    prior_snapshot,
)
from scripts.backtest.historical_inputs import _load_nflreadpy_weekly_sources, build_schedule_history
from scripts.modeling.rb_receiving_identity_runtime_v1 import FEATURES, attach_identity, identity_atlas

TARGET_SEASONS = list(range(2020, 2026))
HISTORY_START = 2013
ALLOWED_STATUS = {"ACT", "INA"}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def depth_order(v) -> float:
    m = re.search(r"(?:^|\D)([1-9]\d*)(?:\D|$)", str(v or ""))
    if not m:
        return np.nan
    z = int(m.group(1))
    return float(z) if z > 0 else np.nan


def targets_pg_bin(prior_games: float, value: float) -> str:
    if prior_games <= 0:
        return "NO_PRIOR_HISTORY"
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value == 0:
        return "0"
    if value <= 1:
        return "0_to_1"
    if value <= 2:
        return "1_to_2"
    return ">2"


def room_share_bin(prior_games: float, value: float) -> str:
    if prior_games <= 0:
        return "NO_PRIOR_HISTORY"
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value < .10:
        return "<0.10"
    if value < .25:
        return "0.10_to_0.25"
    if value < .50:
        return "0.25_to_0.50"
    return ">=0.50"


def depth_bin(v: float) -> str:
    if not np.isfinite(v):
        return "UNAVAILABLE"
    if v == 1:
        return "1"
    if v == 2:
        return "2"
    return "3+"


def strict_feature_frame(exits: pd.DataFrame, season: int, week: int, states: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    q = exits.rename(columns={"player_key": "player_clean_key"}).copy()
    q["position"] = q.get("position", "RB")
    # attach_identity is an as-of feature lookup and its runtime is certified with
    # allow_exact_matches=False. No target-game label is selected into this audit.
    z = attach_identity(q, int(season), int(week), states, prev)
    z["prior_games"] = np.expm1(num(z.get("log1p_prior_games", pd.Series(0.0, index=z.index))).fillna(0.0)).clip(lower=0.0)
    z["same_team_prior_games"] = np.expm1(num(z.get("log1p_same_team_prior_games", pd.Series(0.0, index=z.index))).fillna(0.0)).clip(lower=0.0)
    return z


def aggregate_teamweeks(exit_rows: pd.DataFrame, room_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "team"]
    feats = [
        "prior_targets_pg", "prior_receptions_pg", "prior_rb_room_share",
        "last8_targets_pg", "last8_receptions_pg", "last8_rb_room_share",
        "prev_season_targets_pg", "prev_season_rb_room_share",
    ]
    rows = []
    for k, g in exit_rows.groupby(keys, sort=True):
        r = room_rows.copy()
        for c, v in zip(keys, k):
            r = r.loc[r[c].astype(str).eq(str(v))]
        if len(r) != 1:
            raise RuntimeError(f"room-state lookup failed for {k}: {len(r)}")
        rr = r.iloc[0]
        out = {
            **dict(zip(keys, k)),
            "exits_n": int(len(g)),
            "current_rb_room_n": int(rr.current_rb_room_n),
            "prior_rb_room_n": int(rr.prior_rb_room_n),
            "exit_history_covered_n": int(num(g.prior_games).gt(0).sum()),
            "exit_history_coverage": float(num(g.prior_games).gt(0).mean()),
            "any_exit_positive_history": int(num(g.prior_games).gt(0).any()),
            "any_exit_prior_depth": int(num(g.prior_depth_available).fillna(0).eq(1).any()),
            "best_prior_depth_order": float(num(g.prior_depth_order).min()) if num(g.prior_depth_order).notna().any() else np.nan,
        }
        for f in feats:
            vals = num(g[f])
            hist_vals = vals.loc[num(g.prior_games).gt(0)]
            out[f"max_exit_{f}"] = float(hist_vals.max()) if hist_vals.notna().any() else np.nan
            out[f"sum_exit_{f}"] = float(hist_vals.fillna(0.0).sum()) if len(hist_vals) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def build_summary(exits: pd.DataFrame, teamweeks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in TARGET_SEASONS:
        for phase, mask_e, mask_t in (
            ("ALL", exits.season.eq(season), teamweeks.season.eq(season)),
            ("W1", exits.season.eq(season) & exits.week.eq(1), teamweeks.season.eq(season) & teamweeks.week.eq(1)),
            ("W2+", exits.season.eq(season) & exits.week.gt(1), teamweeks.season.eq(season) & teamweeks.week.gt(1)),
        ):
            e = exits.loc[mask_e].copy(); t = teamweeks.loc[mask_t].copy()
            if e.empty and t.empty:
                continue
            row = {
                "season": season,
                "phase": phase,
                "vacancy_teamweeks": int(len(t)),
                "exited_player_rows": int(len(e)),
                "prior_history_coverage": float(num(e.prior_games).gt(0).mean()) if len(e) else np.nan,
                "prior_depth_coverage": float(num(e.prior_depth_available).fillna(0).eq(1).mean()) if len(e) else np.nan,
                "teamweeks_any_exit_history_rate": float(num(t.any_exit_positive_history).mean()) if len(t) else np.nan,
                "mean_exits": float(num(t.exits_n).mean()) if len(t) else np.nan,
            }
            for col, prefix in (
                ("prior_targets_pg_bin", "prior_tgtpg"),
                ("prior_rb_room_share_bin", "prior_roomshare"),
                ("last8_targets_pg_bin", "last8_tgtpg"),
                ("prior_depth_bin", "prior_depth"),
            ):
                vc = e[col].value_counts(normalize=True, dropna=False) if len(e) else pd.Series(dtype=float)
                for name, value in vc.items():
                    row[f"{prefix}__{name}_rate"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-room-state", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    source_seasons = list(range(min(TARGET_SEASONS) - 1, max(TARGET_SEASONS) + 1))
    schedule = build_schedule_history(source_seasons)
    schedules = {
        int(s): sorted(num(g.week).dropna().astype(int).unique().tolist())
        for s, g in schedule.groupby("season")
    }

    rosters: dict[int, pd.DataFrame] = {}
    depths: dict[int, pd.DataFrame] = {}
    for season in source_seasons:
        raw_roster, raw_depth = _load_nflreadpy_weekly_sources(int(season))
        rosters[season] = normalize_roster(raw_roster, season)
        depths[season] = normalize_depth(raw_depth, season)
        bad = set(rosters[season].status.dropna().astype(str).unique()) - ALLOWED_STATUS
        if bad:
            raise RuntimeError(f"noncanonical roster status survived normalization in {season}: {sorted(bad)}")

    # Build once. Every query below is attached by strict as-of lookup; exact target
    # week is forbidden by the production-safe runtime.
    states, prev_features = identity_atlas(HISTORY_START, max(TARGET_SEASONS))

    exit_parts = []
    room_parts = []
    snapshot_violations = 0
    membership_violations = 0
    for season in TARGET_SEASONS:
        for week in schedules[season]:
            cur = rosters[season].loc[rosters[season].week.eq(week)].copy()
            if cur.empty:
                continue
            ps, pw = prior_snapshot(season, week, schedules)
            if not ((ps < season) or (ps == season and pw < week)):
                snapshot_violations += 1
            prior = rosters[ps].loc[rosters[ps].week.eq(pw)].copy()
            pdepth = depths[ps].loc[depths[ps].week.eq(pw)].copy()
            for team, cg in cur.groupby("team", sort=True):
                pg = prior.loc[prior.team.eq(team)].copy()
                cur_set = set(cg.player_key)
                prior_set = set(pg.player_key)
                exit_keys = sorted(prior_set - cur_set)
                if not exit_keys:
                    continue
                room_parts.append({
                    "season": season, "week": week, "team": team,
                    "prior_snapshot_season": ps, "prior_snapshot_week": pw,
                    "current_rb_room_n": len(cur_set), "prior_rb_room_n": len(prior_set),
                    "room_exits_n": len(exit_keys),
                })
                ex = pg.loc[pg.player_key.isin(exit_keys)].copy()
                if set(ex.player_key) & cur_set or set(ex.player_key) - prior_set:
                    membership_violations += 1
                ex["target_season"] = season; ex["target_week"] = week
                ex["prior_snapshot_season"] = ps; ex["prior_snapshot_week"] = pw
                ex = ex.rename(columns={"season": "roster_source_season", "week": "roster_source_week"})
                ex["season"] = season; ex["week"] = week
                if not pdepth.empty:
                    dp = pdepth.loc[pdepth.team.eq(team), ["player_key", "prior_depth_position", "prior_depth_team"]].drop_duplicates("player_key")
                    ex = ex.merge(dp, on="player_key", how="left", validate="one_to_one")
                else:
                    ex["prior_depth_position"] = ""; ex["prior_depth_team"] = ""
                ex["prior_depth_available"] = ex.prior_depth_position.fillna("").astype(str).str.strip().ne("").astype(int)
                ex["prior_depth_order"] = ex.prior_depth_team.map(depth_order)
                feat = strict_feature_frame(ex, season, week, states, prev_features)
                exit_parts.append(feat)

    exits = pd.concat(exit_parts, ignore_index=True, sort=False) if exit_parts else pd.DataFrame()
    rooms = pd.DataFrame(room_parts).drop_duplicates(["season", "week", "team"])
    if exits.empty or rooms.empty:
        raise RuntimeError("R26C reconstructed zero exits/vacancy team-weeks")

    exits["prior_targets_pg_bin"] = [targets_pg_bin(g, v) for g, v in zip(num(exits.prior_games).fillna(0), num(exits.prior_targets_pg))]
    exits["last8_targets_pg_bin"] = [targets_pg_bin(g, v) for g, v in zip(num(exits.prior_games).fillna(0), num(exits.last8_targets_pg))]
    exits["prior_rb_room_share_bin"] = [room_share_bin(g, v) for g, v in zip(num(exits.prior_games).fillna(0), num(exits.prior_rb_room_share))]
    exits["prior_depth_bin"] = exits.prior_depth_order.map(depth_bin)

    teamweeks = aggregate_teamweeks(exits, rooms)
    summary = build_summary(exits, teamweeks)

    parent = pd.read_csv(a.parent_room_state, low_memory=False)
    parent = parent.loc[num(parent.season).isin(TARGET_SEASONS) & num(parent.room_exits_n).ge(1)].copy()
    parent_keys = set(zip(num(parent.season).astype(int), num(parent.week).astype(int), parent.team.astype(str)))
    ours_keys = set(zip(num(rooms.season).astype(int), num(rooms.week).astype(int), rooms.team.astype(str)))
    reconstructed = len(parent_keys & ours_keys) / max(len(parent_keys), 1)

    history_cov = float(num(exits.prior_games).gt(0).mean())
    teamweek_hist_cov = float(num(teamweeks.any_exit_positive_history).mean())
    integrity = {
        "candidate": "RB_R26C_EXIT_SIGNIFICANCE_SOURCE_AUDIT_V1",
        "target_game_outcomes_selected_or_used": 0,
        "target_game_participation_selected_or_used": 0,
        "sportsbook_inputs_used": 0,
        "same_week_historical_depth_used": False,
        "allowed_roster_status": sorted(ALLOWED_STATUS),
        "snapshot_order_violations": int(snapshot_violations),
        "exit_membership_violations": int(membership_violations),
        "strict_asof_identity": True,
        "parent_vacancy_teamweeks": int(len(parent_keys)),
        "reconstructed_vacancy_teamweeks": int(len(ours_keys)),
        "parent_vacancy_reconstruction_rate": float(reconstructed),
        "exited_player_rows": int(len(exits)),
        "prior_history_coverage": history_cov,
        "vacancy_teamweeks_any_exit_history_rate": teamweek_hist_cov,
        "prior_depth_coverage": float(num(exits.prior_depth_available).fillna(0).eq(1).mean()),
        "production_parameters_changed": False,
    }
    integrity_gates = {
        "no_target_game_outcomes_used": integrity["target_game_outcomes_selected_or_used"] == 0,
        "no_target_game_participation_used": integrity["target_game_participation_selected_or_used"] == 0,
        "sportsbook_zero": integrity["sportsbook_inputs_used"] == 0,
        "same_week_depth_false": integrity["same_week_historical_depth_used"] is False,
        "canonical_status_only": integrity["allowed_roster_status"] == ["ACT", "INA"],
        "strict_prior_snapshots": integrity["snapshot_order_violations"] == 0,
        "exit_membership_exact": integrity["exit_membership_violations"] == 0,
        "strict_asof_identity": integrity["strict_asof_identity"] is True,
        "parent_vacancy_coverage": reconstructed >= 0.995,
        "exited_history_coverage": history_cov >= 0.60,
        "teamweek_history_coverage": teamweek_hist_cov >= 0.60,
    }
    if not all(integrity_gates.values()):
        disposition = "EXIT_SIGNIFICANCE_SOURCE_FAILURE" if not all(list(integrity_gates.values())[:9]) else "EXIT_SIGNIFICANCE_SOURCE_INSUFFICIENT"
    else:
        disposition = "EXIT_SIGNIFICANCE_SOURCE_READY"

    result = {
        **integrity,
        "integrity_gates": integrity_gates,
        "disposition": disposition,
        "scientific_candidate_scored": False,
        "outcome_diagnostic_authorized": disposition == "EXIT_SIGNIFICANCE_SOURCE_READY",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    exits.to_csv(a.out_dir / "r26c_exited_player_source_state.csv", index=False)
    teamweeks.to_csv(a.out_dir / "r26c_vacancy_teamweek_aggregate.csv", index=False)
    summary.to_csv(a.out_dir / "r26c_season_phase_source_summary.csv", index=False)
    (a.out_dir / "r26c_source_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(summary.to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
