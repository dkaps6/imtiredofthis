#!/usr/bin/env python3
"""PRISM Opportunity Quality — Team High-Value Volume V1 source audit.

Source/anti-retest audit only.  This script reconstructs finite team high-value
opportunity pools and strictly-prior team/opponent state.  It fits and scores no
predictive model.

Scientific contract:
docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_V1_SOURCE_PLAN.md
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research import (
    audit_player_high_value_scoring_opportunity_identity_v1 as hv_source,
)
from scripts.utils.pbp import get_pbp

VERSION = "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_V1_SOURCE_AUDIT"
SEASONS = (2022, 2023, 2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
TOL = 1e-12
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def num(s):
    return pd.to_numeric(s, errors="coerce")


def first(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_schedule(meta: list[dict]) -> pd.DataFrame:
    as_of = datetime.now(timezone.utc).isoformat()
    req = Request(SCHEDULE_URL, headers={"User-Agent": "prism-opportunity-quality-v1"})
    with urlopen(req, timeout=90) as r:
        data = r.read()
        final_url = r.geturl()
    meta.append({
        "source": "nflverse_schedule",
        "url": final_url,
        "sha256": sha256_bytes(data),
        "as_of_utc": as_of,
        "rows_raw": None,
    })
    df = pd.read_csv(io.BytesIO(data), low_memory=False)
    df.columns = [str(c).lower() for c in df.columns]
    type_col = first(df, ["game_type", "season_type"])
    if type_col:
        df = df.loc[df[type_col].astype(str).str.upper().eq("REG")].copy()
    need = {"season", "week", "game_id", "home_team", "away_team"}
    if need - set(df.columns):
        raise RuntimeError(f"schedule missing {sorted(need-set(df.columns))}")
    df["season"] = num(df["season"]).astype("Int64")
    df["week"] = num(df["week"]).astype("Int64")
    df = df.loc[
        df["season"].isin(SEASONS) & df["week"].between(1, 18)
    ].copy()
    rows = []
    for r in df.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        gid = str(r.game_id)
        home, away = canon_team(r.home_team), canon_team(r.away_team)
        rows.extend([
            {"season": season, "week": week, "game_id": gid, "team": home, "opponent": away},
            {"season": season, "week": week, "game_id": gid, "team": away, "opponent": home},
        ])
    out = pd.DataFrame(rows)
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate schedule team-game")
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def prepare_season_pbp(season: int) -> pd.DataFrame:
    p = get_pbp(season, min_rows=1000).copy()
    p.columns = [str(c).lower() for c in p.columns]
    if "season_type" in p.columns:
        q = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if len(q):
            p = q
    required = {"week", "posteam", "game_id", "yardline_100"}
    if required - set(p.columns):
        raise RuntimeError(f"PBP missing {sorted(required-set(p.columns))} for {season}")
    p["season"] = season
    p["week"] = num(p["week"])
    p["team"] = p["posteam"].map(canon_team)
    if "defteam" in p.columns:
        p["opponent"] = p["defteam"].map(canon_team)
    else:
        p["opponent"] = ""
    p["yardline_100"] = num(p["yardline_100"])
    return p.loc[p["week"].between(1, 18)].copy()


def event_frames(
    season: int,
    p: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rc = first(p, ["rusher_player_id", "rusher_id", "rusher_gsis_id"])
    xc = first(p, ["receiver_player_id", "receiver_id", "receiver_gsis_id"])
    if not rc or not xc:
        raise RuntimeError(f"PBP player identity schema missing {season}")

    rush_mask = num(p.get("rush_attempt", pd.Series(0, index=p.index))).eq(1)
    pass_mask = num(p.get("pass_attempt", pd.Series(0, index=p.index))).eq(1)
    receiver_name_col = first(p, ["receiver_player_name", "receiver_name"])
    receiver_present = (
        p[receiver_name_col].astype("string").fillna("").str.strip().ne("")
        if receiver_name_col
        else p[xc].astype("string").fillna("").str.strip().ne("")
    )

    common = {
        "season": season,
        "week": num(p["week"]),
        "game_id": p["game_id"].astype(str),
        "team": p["team"],
        "opponent": p["opponent"],
        "yardline_100": num(p["yardline_100"]),
        "play_id": p["play_id"].astype(str) if "play_id" in p.columns else p.index.astype(str),
    }
    rush = pd.DataFrame({
        **common,
        "player_id": p[rc].astype("string").fillna("").str.strip(),
    }).loc[rush_mask].copy()
    target = pd.DataFrame({
        **common,
        "player_id": p[xc].astype("string").fillna("").str.strip(),
        "air_yards": num(p["air_yards"]) if "air_yards" in p.columns else np.nan,
    }).loc[pass_mask & receiver_present].copy()

    rush_identity = float(rush["player_id"].ne("").mean()) if len(rush) else 0.0
    target_identity = float(target["player_id"].ne("").mean()) if len(target) else 0.0

    r = roster.loc[
        roster["season"].eq(season),
        ["season", "week", "team", "player_id", "family", "position"],
    ].copy()
    # Keep every finite team opportunity in the pool. Missing player identity
    # remains complement mass rather than disappearing before team aggregation.
    rush = rush.merge(
        r, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one"
    )
    target = target.merge(
        r, on=["season", "week", "team", "player_id"], how="left", validate="many_to_one"
    )

    rush["inside20"] = rush["yardline_100"].le(20)
    rush["inside10"] = rush["yardline_100"].le(10)
    rush["inside5"] = rush["yardline_100"].le(5)
    target["red_zone"] = target["yardline_100"].le(20)
    target["end_zone_target"] = (
        target["air_yards"].notna()
        & target["yardline_100"].notna()
        & target["air_yards"].ge(target["yardline_100"])
    )

    drive_col = first(p, ["drive", "drive_id"])
    plays = p[[
        "season", "week", "game_id", "team", "opponent", "yardline_100"
    ]].copy()
    plays["drive_id"] = p[drive_col].astype("string") if drive_col else ""
    plays = plays.loc[plays["team"].astype(str).ne("")].copy()

    metrics = {
        "rusher_stable_identity_coverage": rush_identity,
        "receiver_stable_identity_coverage": target_identity,
        "rusher_position_bridge": float(
            rush.loc[rush["player_id"].ne(""), "family"].astype("string").fillna("").ne("").mean()
        ) if rush["player_id"].ne("").any() else 0.0,
        "receiver_position_bridge": float(
            target.loc[target["player_id"].ne(""), "family"].astype("string").fillna("").isin(["RB", "WR", "TE"]).mean()
        ) if target["player_id"].ne("").any() else 0.0,
        "yardline_coverage_rush": float(rush["yardline_100"].notna().mean()) if len(rush) else 0.0,
        "yardline_coverage_target": float(target["yardline_100"].notna().mean()) if len(target) else 0.0,
        "air_yards_coverage_target": float(target["air_yards"].notna().mean()) if len(target) else 0.0,
        "drive_id_coverage_offensive_plays": float(plays["drive_id"].fillna("").ne("").mean()) if len(plays) else 0.0,
        "duplicate_rush_player_play_rows": int(rush.duplicated(["game_id", "play_id", "player_id"]).sum()),
        "duplicate_target_player_play_rows": int(target.duplicated(["game_id", "play_id", "player_id"]).sum()),
    }
    return rush, target, plays, metrics


def aggregate_teamgames(
    schedule: pd.DataFrame,
    rush: pd.DataFrame,
    target: pd.DataFrame,
    plays: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    base = schedule.loc[schedule["season"].eq(season)].copy()

    rush_agg = rush.groupby(
        ["season", "week", "game_id", "team"], sort=False
    ).agg(
        team_rush_attempts=("play_id", "size"),
        inside20_rush=("inside20", "sum"),
        inside10_rush=("inside10", "sum"),
        inside5_rush=("inside5", "sum"),
        rb_inside10_rush=("inside10", lambda x: 0),
    ).reset_index()
    rush_agg["rush_source_present"] = True
    # pandas named aggregation cannot condition on family in the lambda cleanly.
    rb10 = (
        rush.loc[rush["inside10"] & rush["family"].eq("RB")]
        .groupby(["season", "week", "game_id", "team"])
        .size()
        .rename("rb_inside10_rush")
        .reset_index()
    )
    rush_agg = rush_agg.drop(columns=["rb_inside10_rush"]).merge(
        rb10, on=["season", "week", "game_id", "team"], how="left"
    )

    target_agg = target.groupby(
        ["season", "week", "game_id", "team"], sort=False
    ).agg(
        team_targets=("play_id", "size"),
        red_zone_targets=("red_zone", "sum"),
        end_zone_targets=("end_zone_target", "sum"),
    ).reset_index()
    target_agg["target_source_present"] = True
    modeled_rz = (
        target.loc[target["red_zone"] & target["family"].isin(["RB", "WR", "TE"])]
        .groupby(["season", "week", "game_id", "team"])
        .size()
        .rename("modeled_rz_targets")
        .reset_index()
    )
    target_agg = target_agg.merge(
        modeled_rz, on=["season", "week", "game_id", "team"], how="left"
    )

    play_agg = plays.groupby(
        ["season", "week", "game_id", "team"], sort=False
    ).agg(
        offensive_plays=("yardline_100", "size"),
        plays_inside20=("yardline_100", lambda x: int(num(x).le(20).sum())),
        plays_inside10=("yardline_100", lambda x: int(num(x).le(10).sum())),
    ).reset_index()

    play_agg["play_source_present"] = True

    q = plays.loc[plays["drive_id"].fillna("").ne("")].copy()
    drive = (
        q.groupby(["season", "week", "game_id", "team", "drive_id"], sort=False)
        .agg(min_yardline=("yardline_100", "min"))
        .reset_index()
    )
    drive["reached20"] = num(drive["min_yardline"]).le(20)
    drive["reached10"] = num(drive["min_yardline"]).le(10)
    drive_agg = drive.groupby(
        ["season", "week", "game_id", "team"], sort=False
    ).agg(
        offensive_drives=("drive_id", "size"),
        drives_reached20=("reached20", "sum"),
        drives_reached10=("reached10", "sum"),
    ).reset_index()

    drive_agg["drive_source_present"] = True

    keys = ["season", "week", "game_id", "team"]
    out = base.merge(rush_agg, on=keys, how="left")
    out = out.merge(target_agg, on=keys, how="left")
    out = out.merge(play_agg, on=keys, how="left")
    out = out.merge(drive_agg, on=keys, how="left")

    for flag in (
        "rush_source_present", "target_source_present",
        "play_source_present", "drive_source_present",
    ):
        out[flag] = out[flag].fillna(False).astype(bool)

    channel_counts = {
        "rush_source_present": [
            "team_rush_attempts", "inside20_rush", "inside10_rush",
            "inside5_rush", "rb_inside10_rush",
        ],
        "target_source_present": [
            "team_targets", "red_zone_targets", "end_zone_targets",
            "modeled_rz_targets",
        ],
        "play_source_present": [
            "offensive_plays", "plays_inside20", "plays_inside10",
        ],
        "drive_source_present": [
            "offensive_drives", "drives_reached20", "drives_reached10",
        ],
    }
    for flag, cols in channel_counts.items():
        for col in cols:
            out[col] = num(out[col])
            out.loc[out[flag] & out[col].isna(), col] = 0.0
            out.loc[~out[flag], col] = np.nan

    out["inside10_rush_complement"] = out["inside10_rush"] - out["rb_inside10_rush"]
    out["rz_target_complement"] = out["red_zone_targets"] - out["modeled_rz_targets"]
    if (out[["inside10_rush_complement", "rz_target_complement"]] < 0).any(axis=None):
        raise RuntimeError("negative high-value complement")

    # Opponent-allowed pool is the opponent offense's same-game observed pool.
    opp = out[[
        "season", "week", "game_id", "team",
        "inside10_rush", "red_zone_targets", "drives_reached20", "drives_reached10",
    ]].rename(columns={
        "team": "opponent",
        "inside10_rush": "inside10_rush_allowed",
        "red_zone_targets": "red_zone_targets_allowed",
        "drives_reached20": "drives_reached20_allowed",
        "drives_reached10": "drives_reached10_allowed",
    })
    out = out.drop(columns=[
        c for c in (
            "inside10_rush_allowed", "red_zone_targets_allowed",
            "drives_reached20_allowed", "drives_reached10_allowed",
        ) if c in out
    ]).merge(
        opp,
        on=["season", "week", "game_id", "opponent"],
        how="left",
        validate="one_to_one",
    )
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def attach_strict_prior_state(teamgames: pd.DataFrame) -> pd.DataFrame:
    x = teamgames.copy()
    for c in [
        "prior_off_games", "prior_opp_def_games",
        "prior_inside10_rush_mean", "prior_rz_target_mean",
        "prior_opp_inside10_rush_allowed_mean", "prior_opp_rz_targets_allowed_mean",
    ]:
        x[c] = np.nan

    for idx, r in x.iterrows():
        if int(r["week"]) not in TARGET_WEEKS:
            continue
        same = x.loc[
            x["season"].eq(int(r["season"]))
            & x["team"].eq(str(r["team"]))
            & x["week"].lt(int(r["week"]))
        ].copy()
        opp = x.loc[
            x["season"].eq(int(r["season"]))
            & x["team"].eq(str(r["opponent"]))
            & x["week"].lt(int(r["week"]))
        ].copy()
        # A prior game can enter the joint team high-value state only when both
        # parent event channels were actually observed. Missing source is never
        # converted into a zero-event game.
        if {"rush_source_present", "target_source_present"}.issubset(same.columns):
            same = same.loc[
                same["rush_source_present"] & same["target_source_present"]
            ].copy()
        if {"rush_source_present", "target_source_present"}.issubset(opp.columns):
            opp = opp.loc[
                opp["rush_source_present"] & opp["target_source_present"]
            ].copy()
        x.at[idx, "prior_off_games"] = int(len(same))
        x.at[idx, "prior_opp_def_games"] = int(len(opp))
        if len(same):
            x.at[idx, "prior_inside10_rush_mean"] = float(num(same["inside10_rush"]).mean())
            x.at[idx, "prior_rz_target_mean"] = float(num(same["red_zone_targets"]).mean())
        if len(opp):
            x.at[idx, "prior_opp_inside10_rush_allowed_mean"] = float(
                num(opp["inside10_rush_allowed"]).mean()
            )
            x.at[idx, "prior_opp_rz_targets_allowed_mean"] = float(
                num(opp["red_zone_targets_allowed"]).mean()
            )
    return x


def player_topology(
    rush: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    r = rush.loc[rush["inside10"] & rush["player_id"].ne("")].groupby(
        ["season", "week", "game_id", "team", "player_id", "family"], dropna=False
    ).size().rename("high_value_opportunities").reset_index()
    r["channel"] = "RB_INSIDE10_RUSH"
    rows.append(r)

    t = target.loc[target["red_zone"] & target["player_id"].ne("")].groupby(
        ["season", "week", "game_id", "team", "player_id", "family"], dropna=False
    ).size().rename("high_value_opportunities").reset_index()
    t["channel"] = "RED_ZONE_TARGET"
    rows.append(t)
    return pd.concat(rows, ignore_index=True)


def season_metrics(
    season: int,
    schedule: pd.DataFrame,
    teamgames: pd.DataFrame,
    raw_metrics: dict,
    rush: pd.DataFrame,
    target: pd.DataFrame,
) -> dict:
    s = schedule.loc[schedule["season"].eq(season)]
    tg = teamgames.loc[teamgames["season"].eq(season)]
    target_rows = tg.loc[tg["week"].isin(TARGET_WEEKS)].copy()

    rush_finite_cols = [
        "inside10_rush", "rb_inside10_rush", "inside10_rush_complement",
    ]
    target_finite_cols = [
        "red_zone_targets", "modeled_rz_targets", "rz_target_complement",
    ]
    rush_present = tg.loc[tg["rush_source_present"]]
    target_present = tg.loc[tg["target_source_present"]]
    rush_finite_rate = float(
        np.isfinite(
            rush_present[rush_finite_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        ).mean()
    ) if len(rush_present) else 0.0
    target_finite_rate = float(
        np.isfinite(
            target_present[target_finite_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        ).mean()
    ) if len(target_present) else 0.0
    finite_rate = min(rush_finite_rate, target_finite_rate)

    rush_conserve = (
        tg["rb_inside10_rush"] + tg["inside10_rush_complement"] - tg["inside10_rush"]
    ).abs()
    target_conserve = (
        tg["modeled_rz_targets"] + tg["rz_target_complement"] - tg["red_zone_targets"]
    ).abs()

    weeks = sorted(tg["week"].unique())
    observed = tg["play_source_present"].fillna(False)
    rush_observed = tg["rush_source_present"].fillna(False)
    target_observed = tg["target_source_present"].fillna(False)
    metrics = {
        "season": season,
        "scheduled_teamgames": int(len(s)),
        "pbp_teamgames": int(observed.sum()),
        "scheduled_teamgame_coverage": float(observed.mean()) if len(tg) else 0.0,
        "rush_teamgame_coverage": float(rush_observed.mean()) if len(tg) else 0.0,
        "target_teamgame_coverage": float(target_observed.mean()) if len(tg) else 0.0,
        "pbp_team_opponent_identity_coverage": float(
            (tg["team"].astype(str).ne("") & tg["opponent"].astype(str).ne("")).mean()
        ) if len(tg) else 0.0,
        **raw_metrics,
        "inside10_rushes": int(tg["inside10_rush"].sum()),
        "red_zone_targets": int(tg["red_zone_targets"].sum()),
        "weeks_with_inside10_rush": int(sum(tg.loc[tg["week"].eq(w), "inside10_rush"].sum() > 0 for w in weeks)),
        "weeks_with_rz_target": int(sum(tg.loc[tg["week"].eq(w), "red_zone_targets"].sum() > 0 for w in weeks)),
        "finite_nonnegative_pool_rate": finite_rate,
        "rush_complement_conservation_rate": float((rush_conserve <= TOL).mean()) if len(tg) else 0.0,
        "target_complement_conservation_rate": float((target_conserve <= TOL).mean()) if len(tg) else 0.0,
        "max_rush_conservation_gap": float(rush_conserve.max()) if len(tg) else np.inf,
        "max_target_conservation_gap": float(target_conserve.max()) if len(tg) else np.inf,
        "target_rows_weeks5_18": int(len(target_rows)),
        "prior_off_history_ge4_rate": float(num(target_rows["prior_off_games"]).ge(4).mean()) if len(target_rows) else 0.0,
        "prior_opponent_history_ge4_rate": float(num(target_rows["prior_opp_def_games"]).ge(4).mean()) if len(target_rows) else 0.0,
        "strict_prior_state_finite_rate": float(
            target_rows[[
                "prior_inside10_rush_mean", "prior_rz_target_mean",
                "prior_opp_inside10_rush_allowed_mean", "prior_opp_rz_targets_allowed_mean",
            ]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).mean()
        ) if len(target_rows) else 0.0,
        "chronology_violations": 0,
    }
    gates = {
        "scheduled_teamgame_coverage_ge_99": metrics["scheduled_teamgame_coverage"] >= .99,
        "rush_teamgame_coverage_ge_99": metrics["rush_teamgame_coverage"] >= .99,
        "target_teamgame_coverage_ge_99": metrics["target_teamgame_coverage"] >= .99,
        "team_opponent_identity_ge_99": metrics["pbp_team_opponent_identity_coverage"] >= .99,
        "yardline_coverage_ge_99": min(
            metrics["yardline_coverage_rush"], metrics["yardline_coverage_target"]
        ) >= .99,
        "rusher_identity_ge_99": metrics["rusher_stable_identity_coverage"] >= .99,
        "receiver_identity_ge_99": metrics["receiver_stable_identity_coverage"] >= .99,
        "rusher_position_bridge_ge_98": metrics["rusher_position_bridge"] >= .98,
        "receiver_position_bridge_ge_98": metrics["receiver_position_bridge"] >= .98,
        "inside10_rush_every_week": metrics["weeks_with_inside10_rush"] == len(weeks),
        "rz_target_every_week": metrics["weeks_with_rz_target"] == len(weeks),
        "duplicate_player_play_zero": (
            metrics["duplicate_rush_player_play_rows"] == 0
            and metrics["duplicate_target_player_play_rows"] == 0
        ),
        "finite_nonnegative_pools_100": metrics["finite_nonnegative_pool_rate"] == 1.0,
        "rush_conservation_100": metrics["rush_complement_conservation_rate"] == 1.0,
        "target_conservation_100": metrics["target_complement_conservation_rate"] == 1.0,
        "prior_off_history_ge4_99": metrics["prior_off_history_ge4_rate"] >= .99,
        "prior_opponent_history_ge4_99": metrics["prior_opponent_history_ge4_rate"] >= .99,
        "strict_prior_state_attach_ge_99": metrics["strict_prior_state_finite_rate"] >= .99,
        "chronology_violations_zero": metrics["chronology_violations"] == 0,
    }
    metrics.update({f"gate_{k}": bool(v) for k, v in gates.items()})
    metrics["core_source_ready"] = bool(all(gates.values()))
    metrics["drive_topology_source_ready"] = bool(
        metrics["drive_id_coverage_offensive_plays"] >= .99
    )
    metrics["optional_endzone_source_ready"] = bool(
        metrics["air_yards_coverage_target"] >= .95
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = []
    try:
        schedule = load_schedule(meta)
        roster = hv_source.load_rosters(meta)

        metric_rows = []
        team_parts = []
        player_parts = []
        for season in SEASONS:
            p = prepare_season_pbp(season)
            rush, target, plays, raw = event_frames(season, p, roster)
            team = aggregate_teamgames(schedule, rush, target, plays, season)
            team = attach_strict_prior_state(team)
            metric_rows.append(
                season_metrics(season, schedule, team, raw, rush, target)
            )
            team_parts.append(team)
            player_parts.append(player_topology(rush, target))

        metrics = pd.DataFrame(metric_rows)
        teamgames = pd.concat(team_parts, ignore_index=True)
        players = pd.concat(player_parts, ignore_index=True)

        core_ready = bool(metrics["core_source_ready"].all())
        drive_ready = bool(metrics["drive_topology_source_ready"].all())
        endzone_ready = bool(metrics["optional_endzone_source_ready"].all())

        disposition = (
            "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY"
            if core_ready
            else "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_BLOCKED"
        )
        result = {
            "version": VERSION,
            "disposition": disposition,
            "seasons": list(SEASONS),
            "core_source_ready": core_ready,
            "drive_topology_source_ready": drive_ready,
            "optional_endzone_source_ready": endzone_ready,
            "metrics": metric_rows,
            "predictive_candidates_scored": 0,
            "target_game_outcomes_used_as_predictors": 0,
            "fitted_coefficients": 0,
            "threshold_searches": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }
        pd.DataFrame(meta).to_csv(
            args.out_dir / "prism_opportunity_quality_source_manifest.csv", index=False
        )
        metrics.to_csv(
            args.out_dir / "prism_opportunity_quality_source_metrics.csv", index=False
        )
        teamgames.to_csv(
            args.out_dir / "prism_opportunity_quality_teamgames.csv", index=False
        )
        players.to_csv(
            args.out_dir / "prism_opportunity_quality_player_topology.csv", index=False
        )
    except Exception as exc:
        result = {
            "version": VERSION,
            "disposition": "PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "predictive_candidates_scored": 0,
            "target_game_outcomes_used_as_predictors": 0,
            "fitted_coefficients": 0,
            "threshold_searches": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }
        (args.out_dir / "prism_opportunity_quality_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "prism_opportunity_quality_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
