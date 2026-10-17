#!/usr/bin/env python3
"""PRISM opportunity-quality team high-value volume V1 — source audit.

Frozen source contract:
docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_HIGH_VALUE_VOLUME_V1_SOURCE_PLAN.md

This audit does not score a predictive candidate.  It reconciles the already-
qualified high-value PBP topology source with the already-qualified PRISM
Stage-2 historical entitlement source, and proves whether strictly-prior team
high-value volume state can be constructed without target-game leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.historical_inputs import build_schedule_history

VERSION = "PRISM_OPPORTUNITY_QUALITY_TEAM_HIGH_VALUE_VOLUME_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "48d9331f86aa052c90bc48816b2f031b84900c4b"
HIGH_VALUE_RUN = 38848080453
HIGH_VALUE_ARTIFACT = 12023852453
HIGH_VALUE_DIGEST = "sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda"
STAGE2_RUN = 38955368643
STAGE2_ARTIFACT = 12070806828
STAGE2_DIGEST = "sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83"

SOURCE_SEASONS = (2022, 2023, 2024, 2025)
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
HISTORY_WINDOW = 4
HISTORY_MIN = 3
TOL = 1e-12

CELL_MAP = {
    "RB_RUSH_INSIDE10": {
        "stage2_cell": "RB_RUSHING",
        "family": "RB",
        "channel": "rush",
        "high_value_col": "inside10",
        "pool_col": "team_inside10_rushes",
        "parent_col": "team_rushes",
        "support": {"rows": 700, "players": 80, "positive_teamgames": 300},
    },
    "RB_RECEIVING_REDZONE": {
        "stage2_cell": "RB_RECEIVING",
        "family": "RB",
        "channel": "target",
        "high_value_col": "inside20",
        "pool_col": "team_redzone_targets",
        "parent_col": "team_targets",
        "support": {"rows": 700, "players": 80, "positive_teamgames": 300},
    },
    "TE_RECEIVING_REDZONE": {
        "stage2_cell": "TE_RECEIVING",
        "family": "TE",
        "channel": "target",
        "high_value_col": "inside20",
        "pool_col": "team_redzone_targets",
        "parent_col": "team_targets",
        "support": {"rows": 700, "players": 75, "positive_teamgames": 300},
    },
}


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def read_json(path: Path, label: str) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_schedule() -> pd.DataFrame:
    s = build_schedule_history(TARGET_SEASONS).copy()
    required = {"season", "week", "team", "opponent"}
    if required - set(s.columns):
        raise RuntimeError(f"schedule missing {sorted(required-set(s.columns))}")
    s["season"] = num(s["season"]).astype(int)
    s["week"] = num(s["week"]).astype(int)
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    s = s.loc[
        s["season"].isin(TARGET_SEASONS)
        & s["week"].between(1, 18)
    ].copy()
    if s.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate schedule team-game")
    return s[["season", "week", "team", "opponent"]].reset_index(drop=True)


def build_team_topology(playerweeks: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    p = playerweeks.copy()
    required = {
        "season", "week", "team", "player_id", "family",
        "opportunities", "inside20", "inside10", "inside5", "channel",
    }
    if required - set(p.columns):
        raise RuntimeError(f"high-value playerweeks missing {sorted(required-set(p.columns))}")
    p["season"] = num(p["season"]).astype(int)
    p["week"] = num(p["week"]).astype(int)
    p["team"] = p["team"].map(canon_team)
    for c in ("opportunities", "inside20", "inside10", "inside5"):
        p[c] = num(p[c]).fillna(0.0)
    if (p[["opportunities", "inside20", "inside10", "inside5"]] < 0).any().any():
        raise RuntimeError("negative high-value opportunity count")
    if ((p["inside10"] - p["inside20"]) > TOL).any():
        raise RuntimeError("inside10 exceeds inside20")
    if ((p["inside5"] - p["inside10"]) > TOL).any():
        raise RuntimeError("inside5 exceeds inside10")

    rush = (
        p.loc[p["channel"].eq("rush")]
        .groupby(["season", "week", "team"], sort=False)
        .agg(
            team_rushes=("opportunities", "sum"),
            team_inside20_rushes=("inside20", "sum"),
            team_inside10_rushes=("inside10", "sum"),
            team_inside5_rushes=("inside5", "sum"),
        )
        .reset_index()
    )
    target = (
        p.loc[p["channel"].eq("target")]
        .groupby(["season", "week", "team"], sort=False)
        .agg(
            team_targets=("opportunities", "sum"),
            team_redzone_targets=("inside20", "sum"),
            team_inside10_targets=("inside10", "sum"),
        )
        .reset_index()
    )

    t = schedule.merge(
        rush, on=["season", "week", "team"], how="left", validate="one_to_one"
    ).merge(
        target, on=["season", "week", "team"], how="left", validate="one_to_one"
    )
    count_cols = [
        "team_rushes", "team_inside20_rushes", "team_inside10_rushes",
        "team_inside5_rushes", "team_targets", "team_redzone_targets",
        "team_inside10_targets",
    ]
    for c in count_cols:
        t[c] = num(t[c])
    t["rush_source_present"] = t["team_rushes"].notna()
    t["target_source_present"] = t["team_targets"].notna()

    # Zero high-value count is a valid football outcome only when the parent
    # channel itself is present for that team-game.
    for c in ("team_inside20_rushes", "team_inside10_rushes", "team_inside5_rushes"):
        # Preserve NaN when the parent rushing channel itself is absent.
        # A genuine zero high-value count is only legal inside a source-present
        # team-game; missing source may never be relabeled as zero.
        t.loc[~t["rush_source_present"], c] = np.nan
        t.loc[t["rush_source_present"] & t[c].isna(), c] = 0.0
    for c in ("team_redzone_targets", "team_inside10_targets"):
        t.loc[~t["target_source_present"], c] = np.nan
        t.loc[t["target_source_present"] & t[c].isna(), c] = 0.0

    if (t.loc[t["rush_source_present"], "team_inside10_rushes"] > t.loc[t["rush_source_present"], "team_rushes"] + TOL).any():
        raise RuntimeError("inside10 rushing pool exceeds team rushes")
    if (t.loc[t["target_source_present"], "team_redzone_targets"] > t.loc[t["target_source_present"], "team_targets"] + TOL).any():
        raise RuntimeError("red-zone target pool exceeds team targets")
    return t


def attach_strict_prior_state(team: pd.DataFrame) -> pd.DataFrame:
    x = team.copy().sort_values(["season", "week", "team"]).reset_index(drop=True)
    metric_cols = [
        "team_inside10_rushes",
        "team_redzone_targets",
        "team_rushes",
        "team_targets",
    ]

    rows: list[dict] = []
    for r in x.itertuples(index=False):
        season = int(r.season)
        week = int(r.week)
        team_name = str(r.team)
        opp = str(r.opponent)

        prior_off = x.loc[
            x["season"].eq(season)
            & x["team"].eq(team_name)
            & x["week"].lt(week)
        ].sort_values("week").tail(HISTORY_WINDOW)

        prior_def = x.loc[
            x["season"].eq(season)
            & x["opponent"].eq(opp)
            & x["week"].lt(week)
        ].sort_values("week").tail(HISTORY_WINDOW)

        league = x.loc[
            x["season"].eq(season)
            & x["week"].lt(week)
        ]

        rec = {
            "season": season,
            "week": week,
            "team": team_name,
            "opponent": opp,
            "prior_off_games": int(len(prior_off)),
            "prior_opp_allowed_games": int(len(prior_def)),
            "league_prior_teamgames": int(len(league)),
            "off_max_prior_week": int(prior_off["week"].max()) if len(prior_off) else 0,
            "opp_max_prior_week": int(prior_def["week"].max()) if len(prior_def) else 0,
        }
        for c in metric_cols:
            rec[f"prior_off_{c}_pg"] = (
                float(num(prior_off[c]).mean()) if len(prior_off) else np.nan
            )
            rec[f"prior_opp_allowed_{c}_pg"] = (
                float(num(prior_def[c]).mean()) if len(prior_def) else np.nan
            )
            rec[f"league_prior_{c}_pg"] = (
                float(num(league[c]).mean()) if len(league) else np.nan
            )
        rec["chronology_violation"] = int(
            rec["off_max_prior_week"] >= week
            or rec["opp_max_prior_week"] >= week
        )
        rows.append(rec)

    return x.merge(
        pd.DataFrame(rows),
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )


def player_high_value_labels(playerweeks: pd.DataFrame, cell: str) -> pd.DataFrame:
    spec = CELL_MAP[cell]
    q = playerweeks.loc[
        num(playerweeks["season"]).isin(TARGET_SEASONS)
        & num(playerweeks["week"]).isin(TARGET_WEEKS)
        & playerweeks["channel"].astype(str).eq(spec["channel"])
        & playerweeks["family"].astype(str).eq(spec["family"])
    ].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_id"] = q["player_id"].astype("string").fillna("").str.strip()
    q["actual_high_value_opportunities"] = num(q[spec["high_value_col"]]).fillna(0.0)
    if q.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError(f"duplicate player high-value label rows cell={cell}")
    return q[
        ["season", "week", "team", "player_id", "actual_high_value_opportunities"]
    ]


def evaluate_cell(
    *,
    cell: str,
    season: int,
    stage2: pd.DataFrame,
    playerweeks: pd.DataFrame,
    team_state: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    spec = CELL_MAP[cell]
    q = stage2.loc[
        num(stage2["season"]).eq(season)
        & num(stage2["week"]).isin(TARGET_WEEKS)
        & stage2["cell"].astype(str).eq(spec["stage2_cell"])
    ].copy()
    q["season"] = num(q["season"]).astype(int)
    q["week"] = num(q["week"]).astype(int)
    q["team"] = q["team"].map(canon_team)
    q["player_id"] = q["player_id"].astype("string").fillna("").str.strip()
    if q.duplicated(["season", "week", "team", "player_id"]).any():
        raise RuntimeError(f"duplicate Stage-2 identity cell={cell} season={season}")

    labels = player_high_value_labels(playerweeks, cell)
    labels = labels.loc[labels["season"].eq(season)].copy()
    q = q.merge(
        labels,
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="one_to_one",
    )
    q["actual_high_value_opportunities"] = num(
        q["actual_high_value_opportunities"]
    ).fillna(0.0)

    ts = team_state.loc[
        team_state["season"].eq(season)
        & team_state["week"].isin(TARGET_WEEKS)
    ].copy()
    keep = [
        "season", "week", "team", "opponent",
        spec["pool_col"], spec["parent_col"],
        "rush_source_present", "target_source_present",
        "prior_off_games", "prior_opp_allowed_games",
        "league_prior_teamgames", "chronology_violation",
        f"prior_off_{spec['pool_col']}_pg",
        f"prior_opp_allowed_{spec['pool_col']}_pg",
        f"league_prior_{spec['pool_col']}_pg",
        f"prior_off_{spec['parent_col']}_pg",
        f"prior_opp_allowed_{spec['parent_col']}_pg",
        f"league_prior_{spec['parent_col']}_pg",
    ]
    q = q.merge(
        ts[keep],
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )

    pool = num(q[spec["pool_col"]])
    if (num(q["actual_high_value_opportunities"]) > pool + TOL).any():
        raise RuntimeError(f"player high-value count exceeds finite pool cell={cell}")

    room = (
        q.groupby(["season", "week", "team"], sort=False)
        .agg(
            explicit_player_high_value=("actual_high_value_opportunities", "sum"),
            team_high_value_pool=(spec["pool_col"], "first"),
            parent_pool=(spec["parent_col"], "first"),
            prior_off_games=("prior_off_games", "first"),
            prior_opp_allowed_games=("prior_opp_allowed_games", "first"),
            chronology_violation=("chronology_violation", "max"),
        )
        .reset_index()
    )
    room["complement_high_value"] = (
        num(room["team_high_value_pool"])
        - num(room["explicit_player_high_value"])
    )
    if (room["complement_high_value"] < -TOL).any():
        raise RuntimeError(f"negative high-value complement cell={cell}")

    scheduled_teamgames = int(
        ts[["season", "week", "team"]].drop_duplicates().shape[0]
    )
    topology_teamgames = int(
        ts.loc[num(ts[spec["pool_col"]]).notna(), ["season", "week", "team"]]
        .drop_duplicates().shape[0]
    )
    positive_teamgames = int(
        ts.loc[num(ts[spec["pool_col"]]).gt(0), ["season", "week", "team"]]
        .drop_duplicates().shape[0]
    )

    stable_identity = float(
        num(q["stable_roster_identity"]).fillna(0).eq(1).mean()
    ) if len(q) else 0.0
    authority = num(q["authority_covered"]).fillna(0).eq(1)
    denom = num(q["denominator_finite"]).fillna(0).eq(1)
    ambiguous = num(q["ambiguous_identity"]).fillna(0).eq(1)

    team_keys = q[["season", "week", "team"]].drop_duplicates()
    prior_off = ts[["season", "week", "team", "prior_off_games"]].merge(
        team_keys, on=["season", "week", "team"], how="inner"
    )
    prior_def = ts[["season", "week", "team", "prior_opp_allowed_games"]].merge(
        team_keys, on=["season", "week", "team"], how="inner"
    )
    league_available = ts.merge(
        team_keys,
        on=["season", "week", "team"],
        how="inner",
        validate="one_to_one",
    )

    metrics = {
        "cell": cell,
        "season": season,
        "player_games": int(len(q)),
        "players": int(q["player_id"].nunique()),
        "scheduled_teamgames": scheduled_teamgames,
        "topology_teamgames": topology_teamgames,
        "scheduled_teamgame_coverage": (
            topology_teamgames / max(scheduled_teamgames, 1)
        ),
        "positive_high_value_teamgames": positive_teamgames,
        "stable_identity_coverage": stable_identity,
        "canonical_entitlement_coverage": float(authority.mean()) if len(q) else 0.0,
        "finite_parent_denominator_coverage": float(denom.mean()) if len(q) else 0.0,
        "ambiguous_identity_rate": float(ambiguous.mean()) if len(q) else 1.0,
        "strict_prior_offense_attach_rate": float(
            num(prior_off["prior_off_games"]).ge(HISTORY_MIN).mean()
        ) if len(prior_off) else 0.0,
        "strict_prior_opponent_attach_rate": float(
            num(prior_def["prior_opp_allowed_games"]).ge(HISTORY_MIN).mean()
        ) if len(prior_def) else 0.0,
        "strict_prior_league_environment_rate": float(
            num(league_available["league_prior_teamgames"]).gt(0).mean()
        ) if len(league_available) else 0.0,
        "chronology_violations": int(
            num(q["chronology_violation"]).fillna(1).sum()
        ),
        "negative_complement_rooms": int(
            (num(room["complement_high_value"]) < -TOL).sum()
        ),
        "finite_room_pool_rate": float(
            np.isfinite(num(room["team_high_value_pool"])).mean()
        ) if len(room) else 0.0,
        "integer_team_pool_rate": float(
            np.isclose(
                num(room["team_high_value_pool"]),
                np.rint(num(room["team_high_value_pool"])),
                atol=TOL,
            ).mean()
        ) if len(room) else 0.0,
        "max_player_over_pool": float(
            (
                num(q["actual_high_value_opportunities"])
                - num(q[spec["pool_col"]])
            ).max()
        ) if len(q) else np.nan,
        "mean_team_high_value_pool": float(
            num(ts[spec["pool_col"]]).mean()
        ) if len(ts) else np.nan,
        "zero_high_value_teamgame_rate": float(
            num(ts[spec["pool_col"]]).eq(0).mean()
        ) if len(ts) else np.nan,
    }

    support = spec["support"]
    gates = {
        "scheduled_teamgame_coverage_ge_99": metrics["scheduled_teamgame_coverage"] >= 0.99,
        "stable_identity_ge_99": metrics["stable_identity_coverage"] >= 0.99,
        "canonical_entitlement_attach_ge_95": metrics["canonical_entitlement_coverage"] >= 0.95,
        "finite_parent_denominator_ge_99": metrics["finite_parent_denominator_coverage"] >= 0.99,
        "ambiguous_identity_le_0_5pct": metrics["ambiguous_identity_rate"] <= 0.005,
        "strict_prior_offense_attach_ge_95": metrics["strict_prior_offense_attach_rate"] >= 0.95,
        "strict_prior_opponent_attach_ge_95": metrics["strict_prior_opponent_attach_rate"] >= 0.95,
        "strict_prior_league_environment_100": metrics["strict_prior_league_environment_rate"] >= 1.0 - TOL,
        "chronology_violations_zero": metrics["chronology_violations"] == 0,
        "finite_pool_conservation": (
            metrics["negative_complement_rooms"] == 0
            and metrics["max_player_over_pool"] <= TOL
            and metrics["finite_room_pool_rate"] >= 0.99
            and metrics["integer_team_pool_rate"] >= 0.99
        ),
        "support_rows": metrics["player_games"] >= support["rows"],
        "support_players": metrics["players"] >= support["players"],
        "support_positive_teamgames": metrics["positive_high_value_teamgames"] >= support["positive_teamgames"],
    }
    metrics["gates"] = gates
    metrics["source_ready"] = bool(all(gates.values()))
    metrics["disposition"] = (
        f"{cell}_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY"
        if metrics["source_ready"]
        else f"{cell}_PRISM_OPPORTUNITY_QUALITY_SOURCE_BLOCKED"
    )
    q["quality_cell"] = cell
    q["team_high_value_pool"] = num(q[spec["pool_col"]])
    q["parent_opportunity_pool"] = num(q[spec["parent_col"]])
    return metrics, q


def validate_parent_source(high_value_result: dict, stage2_result: dict) -> dict:
    if high_value_result.get("disposition") != "PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY":
        raise RuntimeError("high-value parent is not source-ready")
    if stage2_result.get("disposition") != "PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL":
        raise RuntimeError("Stage-2 parent source disposition drift")

    parent_metrics = {
        int(m["season"]): m for m in high_value_result.get("metrics", [])
    }
    rows = {}
    for season in TARGET_SEASONS:
        m = parent_metrics.get(season)
        if not m:
            raise RuntimeError(f"missing high-value parent season={season}")
        rows[str(season)] = {
            "week_team_coverage": min(
                float(m["week_team_coverage_rush"]),
                float(m["week_team_coverage_target"]),
            ),
            "yardline_coverage": min(
                float(m["yardline_coverage_rush"]),
                float(m["yardline_coverage_target"]),
            ),
            "rusher_identity": float(m["rusher_stable_identity_coverage"]),
            "receiver_identity": float(m["receiver_stable_identity_coverage"]),
            "skill_position_bridge": float(m["skill_position_bridge"]),
            "duplicate_rush_player_play_rows": int(m["duplicate_rush_player_play_rows"]),
            "duplicate_target_player_play_rows": int(m["duplicate_target_player_play_rows"]),
            "endzone_air_yards_input_coverage": float(m["air_yards_coverage_target"]),
        }
        if rows[str(season)]["week_team_coverage"] < 0.99:
            raise RuntimeError("parent week/team source gate drift")
        if rows[str(season)]["yardline_coverage"] < 0.99:
            raise RuntimeError("parent yardline gate drift")
        if rows[str(season)]["rusher_identity"] < 0.99 or rows[str(season)]["receiver_identity"] < 0.99:
            raise RuntimeError("parent player identity gate drift")
        if rows[str(season)]["skill_position_bridge"] < 0.98:
            raise RuntimeError("parent position bridge gate drift")
        if rows[str(season)]["duplicate_rush_player_play_rows"] != 0 or rows[str(season)]["duplicate_target_player_play_rows"] != 0:
            raise RuntimeError("parent duplicate player-play drift")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--high-value-playerweeks", type=Path, required=True)
    ap.add_argument("--high-value-result", type=Path, required=True)
    ap.add_argument("--stage2-detail", type=Path, required=True)
    ap.add_argument("--stage2-result", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        hv = read_csv(args.high_value_playerweeks, "high-value playerweeks")
        hv_result = read_json(args.high_value_result, "high-value parent result")
        s2 = read_csv(args.stage2_detail, "Stage-2 source detail")
        s2_result = read_json(args.stage2_result, "Stage-2 source result")

        parent_source = validate_parent_source(hv_result, s2_result)
        schedule = prepare_schedule()
        team = build_team_topology(hv, schedule)
        team_state = attach_strict_prior_state(team)

        season_cells: list[dict] = []
        joined_parts: list[pd.DataFrame] = []
        dispositions: dict[str, str] = {}

        for cell in CELL_MAP:
            cell_ready = True
            for season in TARGET_SEASONS:
                metrics, detail = evaluate_cell(
                    cell=cell,
                    season=season,
                    stage2=s2,
                    playerweeks=hv,
                    team_state=team_state,
                )
                season_cells.append(metrics)
                joined_parts.append(detail)
                cell_ready = cell_ready and bool(metrics["source_ready"])
            dispositions[cell] = (
                f"{cell}_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY"
                if cell_ready
                else f"{cell}_PRISM_OPPORTUNITY_QUALITY_SOURCE_BLOCKED"
            )

        ready = sum(v.endswith("_SOURCE_READY") for v in dispositions.values())
        if ready == len(CELL_MAP):
            overall = "PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL"
        elif ready:
            overall = "PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_PARTIAL"
        else:
            overall = "PRISM_OPPORTUNITY_QUALITY_SOURCE_BLOCKED"

        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": overall,
            "position_dispositions": dispositions,
            "source_seasons": list(SOURCE_SEASONS),
            "target_seasons": list(TARGET_SEASONS),
            "target_weeks": [min(TARGET_WEEKS), max(TARGET_WEEKS)],
            "history_window": HISTORY_WINDOW,
            "history_min": HISTORY_MIN,
            "high_value_parent": {
                "run": HIGH_VALUE_RUN,
                "artifact": HIGH_VALUE_ARTIFACT,
                "digest": HIGH_VALUE_DIGEST,
            },
            "stage2_parent": {
                "run": STAGE2_RUN,
                "artifact": STAGE2_ARTIFACT,
                "digest": STAGE2_DIGEST,
            },
            "parent_source_metrics": parent_source,
            "season_cells": season_cells,
            "optional_extensions": {
                "drive_state": "NOT_ADJUDICATED__NOT_PERSISTED_IN_QUALIFIED_PARENT_ARTIFACT",
                "endzone_target": (
                    "INPUT_SOURCE_READY_PARENT__ROW_LEVEL_DERIVATION_NOT_PERSISTED"
                    if all(
                        parent_source[str(s)]["endzone_air_yards_input_coverage"] >= 0.95
                        for s in TARGET_SEASONS
                    )
                    else "SOURCE_BLOCKED"
                ),
            },
            "closed_persistence_v1_reused_as_candidate": False,
            "player_high_value_history_used_as_predictor": False,
            "target_game_high_value_counts_used_as_predictors": 0,
            "predictive_models_fit": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }

        pd.DataFrame(season_cells).to_csv(
            args.out_dir / "prism_opportunity_quality_source_season_cells.csv",
            index=False,
        )
        team_state.to_csv(
            args.out_dir / "prism_opportunity_quality_team_state.csv",
            index=False,
        )
        joined = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()
        joined.to_csv(
            args.out_dir / "prism_opportunity_quality_player_join_detail.csv",
            index=False,
        )
        (args.out_dir / "prism_opportunity_quality_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result = {
            "version": VERSION,
            "freeze_commit": FREEZE_COMMIT,
            "disposition": "PRISM_OPPORTUNITY_QUALITY_SOURCE_INTEGRITY_FAILURE",
            "error": f"{type(exc).__name__}: {exc}",
            "closed_persistence_v1_reused_as_candidate": False,
            "player_high_value_history_used_as_predictor": False,
            "target_game_high_value_counts_used_as_predictors": 0,
            "predictive_models_fit": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "production_changed": False,
        }
        (args.out_dir / "prism_opportunity_quality_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
