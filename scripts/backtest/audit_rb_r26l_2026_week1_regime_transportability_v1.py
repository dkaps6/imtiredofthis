#!/usr/bin/env python3
"""R26L prospective/source-only 2026 Week-1 RB regime transportability audit.

No 2026 outcomes, no prediction generation, no sportsbook football inputs, no
same-week depth. Uses canonical weekly ACT/INA roster state for source alignment
with R26/R26J and immutable post-merge production files only for secondary
current-football diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.audit_rb_r26_safe_transition_state_v1 import normalize_roster, prior_snapshot
from scripts.backtest.audit_rb_r26c_exit_significance_sources_v1 import strict_feature_frame
from scripts.backtest.historical_inputs import _load_nflreadpy_weekly_sources, build_schedule_history
from scripts.modeling.rb_receiving_identity_runtime_v1 import identity_atlas
from scripts._opponent_map import canon_team

TARGET_SEASON = 2026
TARGET_WEEK = 1
PRIOR_SEASON = 2025
HISTORY_START = 2013
ALLOWED_STATUS = {"ACT", "INA"}
EPS = 1e-12

PRIMARY_FEATURES = [
    "current_room_n",
    "continuing_n",
    "entrants_n",
    "veteran_entry_n",
    "veteran_entry_share",
    "exit_history_coverage",
    "sum_exit_last8_targets_pg",
]
COUNT_FEATURES = {"current_room_n", "continuing_n", "entrants_n", "veteran_entry_n"}
RATE_FEATURES = {"veteran_entry_share", "exit_history_coverage"}
FLOORS = {**{k: 0.25 for k in COUNT_FEATURES}, **{k: 0.05 for k in RATE_FEATURES}, "sum_exit_last8_targets_pg": 0.25}
RB_POS = {"RB", "HB", "TB", "FB"}


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_one_csv(root: Path, name: str) -> pd.DataFrame:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def read_one_json(root: Path, name: str) -> dict:
    hits = sorted(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return json.loads(hits[0].read_text())


def build_current_source_state() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    schedule = build_schedule_history([PRIOR_SEASON, TARGET_SEASON])
    schedules = {
        int(s): sorted(num(g.week).dropna().astype(int).unique().tolist())
        for s, g in schedule.groupby("season")
    }
    if TARGET_SEASON not in schedules or TARGET_WEEK not in schedules[TARGET_SEASON]:
        raise RuntimeError("2026 Week 1 absent from authoritative schedule")
    ps, pw = prior_snapshot(TARGET_SEASON, TARGET_WEEK, schedules)
    if (ps, pw) != (PRIOR_SEASON, 18):
        raise RuntimeError(f"unexpected 2026 W1 prior snapshot {(ps, pw)}; expected (2025,18)")

    raw25, _ = _load_nflreadpy_weekly_sources(PRIOR_SEASON)
    raw26, _ = _load_nflreadpy_weekly_sources(TARGET_SEASON)
    r25 = normalize_roster(raw25, PRIOR_SEASON)
    r26 = normalize_roster(raw26, TARGET_SEASON)

    cur = r26.loc[r26.week.eq(TARGET_WEEK)].copy()
    prior = r25.loc[r25.week.eq(pw)].copy()
    status25 = set(prior.status.dropna().astype(str).unique()) - {""}
    status26 = set(cur.status.dropna().astype(str).unique()) - {""}

    source_ready_reasons: list[str] = []
    if cur.team.nunique() < 30:
        source_ready_reasons.append(f"2026_week1_teams={cur.team.nunique()}<30")
    if prior.team.nunique() != 32:
        source_ready_reasons.append(f"2025_week18_teams={prior.team.nunique()}!=32")
    if status25 - ALLOWED_STATUS:
        source_ready_reasons.append(f"2025_noncanonical_status={sorted(status25-ALLOWED_STATUS)}")
    if status26 - ALLOWED_STATUS:
        source_ready_reasons.append(f"2026_noncanonical_status={sorted(status26-ALLOWED_STATUS)}")

    if source_ready_reasons:
        return pd.DataFrame(), pd.DataFrame(), {
            "source_ready": False,
            "source_ready_reasons": source_ready_reasons,
            "canonical_2026_week1_teams": int(cur.team.nunique()),
            "canonical_2025_week18_teams": int(prior.team.nunique()),
            "allowed_status_2025": sorted(status25),
            "allowed_status_2026": sorted(status26),
            "prior_snapshot_season": ps,
            "prior_snapshot_week": pw,
        }

    prior_any = prior[["player_key", "team"]].drop_duplicates("player_key", keep="last")
    prior_any_map = dict(zip(prior_any.player_key.astype(str), prior_any.team.astype(str)))

    states, prev_features = identity_atlas(HISTORY_START, TARGET_SEASON)
    room_rows: list[dict] = []
    exit_parts: list[pd.DataFrame] = []
    membership_violations = 0

    for team, cg in cur.groupby("team", sort=True):
        pg = prior.loc[prior.team.eq(team)].copy()
        cur_set = set(cg.player_key.astype(str))
        prior_set = set(pg.player_key.astype(str))
        entrants = sorted(cur_set - prior_set)
        exits = sorted(prior_set - cur_set)
        if not exits:
            continue
        continuing = cur_set & prior_set
        veteran_entries = [p for p in entrants if p in prior_any_map]
        room_rows.append({
            "season": TARGET_SEASON,
            "week": TARGET_WEEK,
            "team": str(team),
            "prior_snapshot_season": ps,
            "prior_snapshot_week": pw,
            "current_room_n": int(len(cur_set)),
            "prior_room_n": int(len(prior_set)),
            "continuing_n": int(len(continuing)),
            "entrants_n": int(len(entrants)),
            "exits_n": int(len(exits)),
            "veteran_entry_n": int(len(veteran_entries)),
            "veteran_entry_share": float(len(veteran_entries) / max(len(cur_set), 1)),
        })
        ex = pg.loc[pg.player_key.astype(str).isin(exits)].copy()
        if set(ex.player_key.astype(str)) & cur_set or set(ex.player_key.astype(str)) - prior_set:
            membership_violations += 1
        ex["season"] = TARGET_SEASON
        ex["week"] = TARGET_WEEK
        ex["prior_depth_position"] = ""
        ex["prior_depth_team"] = ""
        ex["prior_depth_available"] = 0
        feat = strict_feature_frame(ex, TARGET_SEASON, TARGET_WEEK, states, prev_features)
        feat["source_team"] = str(team)
        exit_parts.append(feat)

    rooms = pd.DataFrame(room_rows)
    exits = pd.concat(exit_parts, ignore_index=True, sort=False) if exit_parts else pd.DataFrame()
    if rooms.empty or exits.empty:
        return rooms, exits, {
            "source_ready": False,
            "source_ready_reasons": ["zero_2026_week1_vacancy_rooms_or_exits"],
            "canonical_2026_week1_teams": int(cur.team.nunique()),
            "canonical_2025_week18_teams": int(prior.team.nunique()),
            "allowed_status_2025": sorted(status25),
            "allowed_status_2026": sorted(status26),
            "prior_snapshot_season": ps,
            "prior_snapshot_week": pw,
        }

    # attach_identity preserves the input team field in the feature frame; source_team
    # is retained as a fail-closed backup for aggregation semantics.
    if "team" not in exits.columns:
        exits["team"] = exits["source_team"]
    exits["team"] = exits["team"].fillna(exits["source_team"]).astype(str)
    exits["prior_games"] = num(exits.get("prior_games", 0)).fillna(0.0)
    exits["last8_targets_pg"] = num(exits.get("last8_targets_pg", np.nan))

    agg_rows = []
    for (season, week, team), g in exits.groupby(["season", "week", "team"], sort=True):
        hist = g.loc[g.prior_games.gt(0)].copy()
        agg_rows.append({
            "season": int(season), "week": int(week), "team": str(team),
            "exit_history_coverage": float(g.prior_games.gt(0).mean()),
            "sum_exit_last8_targets_pg": float(num(hist.last8_targets_pg).fillna(0.0).sum()) if len(hist) else np.nan,
            "exit_player_rows": int(len(g)),
            "exit_positive_history_n": int(g.prior_games.gt(0).sum()),
        })
    exagg = pd.DataFrame(agg_rows)
    rooms = rooms.merge(exagg, on=["season", "week", "team"], how="left", validate="one_to_one")
    if rooms.exit_history_coverage.isna().any():
        raise RuntimeError("R26L missing exit aggregate for current vacancy room")

    meta = {
        "source_ready": True,
        "source_ready_reasons": [],
        "canonical_2026_week1_teams": int(cur.team.nunique()),
        "canonical_2025_week18_teams": int(prior.team.nunique()),
        "canonical_2026_week1_rb_rows": int(len(cur)),
        "canonical_2025_week18_rb_rows": int(len(prior)),
        "allowed_status_2025": sorted(status25),
        "allowed_status_2026": sorted(status26),
        "prior_snapshot_season": ps,
        "prior_snapshot_week": pw,
        "vacancy_rooms": int(len(rooms)),
        "exited_player_rows": int(len(exits)),
        "departed_history_coverage": float(exits.prior_games.gt(0).mean()),
        "membership_violations": int(membership_violations),
    }
    return rooms, exits, meta


def secondary_production_diagnostics(prod_root: Path) -> pd.DataFrame:
    roles = read_one_csv(prod_root, "roles_ourlads.csv")
    form = read_one_csv(prod_root, "player_form_consensus.csv")
    roles.columns = [str(c).strip().lower() for c in roles.columns]
    form.columns = [str(c).strip().lower() for c in form.columns]
    roles["team"] = roles.team.map(canon_team)
    form["team"] = form.team.map(canon_team)
    pos = roles.get("position", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    grp = roles.get("position_group", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    rbroles = roles.loc[pos.isin(RB_POS) | grp.eq("RB")].copy()
    if "player_clean_key" not in rbroles.columns or "player_clean_key" not in form.columns:
        raise RuntimeError("R26L production secondary diagnostics missing player_clean_key")
    keep = rbroles[["team", "player_clean_key"]].drop_duplicates()
    f = form.merge(keep, on=["team", "player_clean_key"], how="inner", validate="many_to_one")
    share_col = "tgt_share" if "tgt_share" in f.columns else "target_share"
    f["prior_target_share"] = num(f[share_col]).fillna(0.0).clip(lower=0.0)
    rows = []
    for team, g in f.groupby("team", sort=True):
        v = g.prior_target_share.to_numpy(float)
        s = float(v.sum())
        w = v / s if s > 0 else np.zeros_like(v)
        rows.append({
            "team": str(team),
            "production_ourlads_rb_room_n": int(len(g)),
            "production_prior_target_share_sum": s,
            "production_normalized_rb_hhi": float(np.square(w).sum()) if len(w) else np.nan,
            "production_normalized_top_rb_share": float(w.max()) if len(w) else np.nan,
            "classification_authority": "SECONDARY_ONLY_SOURCE_SEMANTICS_DIFFER_FROM_R26J",
        })
    return pd.DataFrame(rows)


def compare_regime(current: pd.DataFrame, historical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    historical = historical.copy()
    historical["season"] = num(historical.season).astype(int)
    historical["week"] = num(historical.week).astype(int)
    historical = historical.loc[historical.season.between(2020, 2025) & historical.week.eq(1)].copy()
    rows = []
    modern_closer = 0
    y2020_closer = 0
    beyond_2020 = 0
    d20s = []
    dmods = []

    for feature in PRIMARY_FEATURES:
        if feature not in current.columns or feature not in historical.columns:
            raise RuntimeError(f"R26L missing primary feature {feature}")
        v26 = float(num(current[feature]).mean())
        season_means = historical.groupby("season")[feature].apply(lambda s: float(num(s).mean()))
        if 2020 not in season_means.index or not set(range(2021, 2026)).issubset(set(season_means.index)):
            raise RuntimeError(f"R26L incomplete historical season means for {feature}")
        v20 = float(season_means.loc[2020])
        modern_vals = np.array([float(season_means.loc[s]) for s in range(2021, 2026)], dtype=float)
        modern_mean = float(modern_vals.mean())
        modern_min = float(modern_vals.min())
        modern_max = float(modern_vals.max())
        lo = min(v20, modern_min)
        hi = max(v20, modern_max)
        scale = max(hi - lo, FLOORS[feature])
        d20 = abs(v26 - v20) / scale
        dmod = abs(v26 - modern_mean) / scale
        if abs(d20 - dmod) <= EPS:
            pref = "TIE"
        elif dmod < d20:
            pref = "MODERN_CLOSER"; modern_closer += 1
        else:
            pref = "2020_CLOSER"; y2020_closer += 1
        anomaly_sign = np.sign(v20 - modern_mean)
        beyond = bool((anomaly_sign > 0 and v26 > v20 + EPS) or (anomaly_sign < 0 and v26 < v20 - EPS))
        beyond_2020 += int(beyond)
        d20s.append(d20); dmods.append(dmod)
        rows.append({
            "feature": feature,
            "v2026": v26,
            "v2020": v20,
            "modern_mean_2021_2025": modern_mean,
            "modern_min_2021_2025": modern_min,
            "modern_max_2021_2025": modern_max,
            "historical_scale": scale,
            "normalized_distance_to_2020": d20,
            "normalized_distance_to_modern": dmod,
            "preference": pref,
            "beyond_2020_in_2020_anomalous_direction": beyond,
        })

    mean20 = float(np.mean(d20s))
    meanmod = float(np.mean(dmods))
    summary = {
        "modern_closer_features": modern_closer,
        "2020_closer_features": y2020_closer,
        "tie_features": len(PRIMARY_FEATURES) - modern_closer - y2020_closer,
        "mean_normalized_distance_to_2020": mean20,
        "mean_normalized_distance_to_modern": meanmod,
        "beyond_2020_anomalous_direction_features": beyond_2020,
    }
    return pd.DataFrame(rows), summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26j-root", type=Path, required=True)
    ap.add_argument("--production-root", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26L protected-production marker missing")
    jdisp = read_one_json(a.r26j_root, "r26j_source_disposition.json")
    if jdisp.get("disposition") != "2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP" or jdisp.get("all_integrity_gates_pass") is not True:
        raise RuntimeError("R26L R26J parent contract mismatch")

    rooms, exits, source_meta = build_current_source_state()
    historical = read_one_csv(a.r26j_root, "r26j_week1_vacancy_room_source_state.csv")
    secondary = secondary_production_diagnostics(a.production_root)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    rooms.to_csv(a.out_dir / "r26l_2026_week1_room_state.csv", index=False)
    exits.to_csv(a.out_dir / "r26l_2026_exited_player_state.csv", index=False)
    secondary.to_csv(a.out_dir / "r26l_secondary_production_diagnostics.csv", index=False)

    integrity = {
        "parent_digests_verified_by_workflow": True,
        "protected_production_files_clean": True,
        "canonical_2026_week1_teams_ge30": source_meta.get("canonical_2026_week1_teams", 0) >= 30,
        "canonical_2025_week18_teams_eq32": source_meta.get("canonical_2025_week18_teams", 0) == 32,
        "allowed_roster_status_only": set(source_meta.get("allowed_status_2025", [])) <= ALLOWED_STATUS and set(source_meta.get("allowed_status_2026", [])) <= ALLOWED_STATUS,
        "prior_snapshot_exact_2025_w18": source_meta.get("prior_snapshot_season") == 2025 and source_meta.get("prior_snapshot_week") == 18,
        "vacancy_rooms_ge20": source_meta.get("vacancy_rooms", 0) >= 20,
        "departed_history_coverage_ge60pct": source_meta.get("departed_history_coverage", 0.0) >= .60,
        "membership_violations_zero": source_meta.get("membership_violations", 0) == 0,
        "target_game_outcomes_used_zero": True,
        "target_game_participation_used_zero": True,
        "sportsbook_football_inputs_zero": True,
        "same_week_depth_used_false": True,
        "production_parameters_changed_false": True,
    }
    source_ready = bool(source_meta.get("source_ready", False))
    integrity_ok = bool(source_ready and all(integrity.values()))

    if integrity_ok:
        cmp, distances = compare_regime(rooms, historical)
        finite_features = bool(np.isfinite(cmp.v2026).all())
        modern = bool(
            finite_features
            and distances["modern_closer_features"] >= 5
            and distances["mean_normalized_distance_to_modern"] <= .75 * distances["mean_normalized_distance_to_2020"]
            and distances["beyond_2020_anomalous_direction_features"] <= 2
        )
        old = bool(
            finite_features
            and distances["2020_closer_features"] >= 5
            and distances["mean_normalized_distance_to_2020"] <= .75 * distances["mean_normalized_distance_to_modern"]
        )
        if modern:
            disposition = "2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION"
        elif old:
            disposition = "2026_SOURCE_REGIME_2020_LIKE_NO_MODERN_QUALIFICATION"
        else:
            disposition = "2026_SOURCE_REGIME_MIXED_NO_TRANSPORTABILITY_CONCLUSION"
    else:
        cmp = pd.DataFrame(columns=["feature"])
        distances = {
            "modern_closer_features": 0,
            "2020_closer_features": 0,
            "tie_features": 0,
            "mean_normalized_distance_to_2020": np.nan,
            "mean_normalized_distance_to_modern": np.nan,
            "beyond_2020_anomalous_direction_features": 0,
        }
        finite_features = False
        disposition = "2026_CURRENT_SOURCE_NOT_READY"

    current_summary = pd.DataFrame([{
        "season": 2026,
        "week": 1,
        "vacancy_rooms": int(len(rooms)),
        **{f: float(num(rooms[f]).mean()) if len(rooms) and f in rooms.columns else np.nan for f in PRIMARY_FEATURES},
    }])
    current_summary.to_csv(a.out_dir / "r26l_2026_source_summary.csv", index=False)
    cmp.to_csv(a.out_dir / "r26l_transportability_feature_comparison.csv", index=False)

    result = {
        "candidate": "RB_R26L_2026_WEEK1_REGIME_TRANSPORTABILITY_V1",
        "scientific_label": "PROSPECTIVE_SOURCE_ONLY_TRANSPORTABILITY_AUDIT",
        "disposition": disposition,
        "source_meta": source_meta,
        "integrity_gates": integrity,
        "all_integrity_gates_pass": integrity_ok,
        "primary_features": PRIMARY_FEATURES,
        "all_primary_2026_features_finite": finite_features,
        **distances,
        "prospective_qualification_design_authorized": disposition == "2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION",
        "exclude_2020_authorized": False,
        "prospective_shadow_authorized": False,
        "production_promotion_authorized": False,
        "r9_refit": False,
        "predictions_regenerated": False,
        "sportsbook_inputs_used": 0,
        "same_week_depth_used": False,
        "production_parameters_changed": False,
        "r22_changed": False,
        "receiving_yard_means_changed": False,
        "secondary_production_diagnostics_decisive": False,
    }
    (a.out_dir / "r26l_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(current_summary.to_csv(index=False))
    print(cmp.to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
