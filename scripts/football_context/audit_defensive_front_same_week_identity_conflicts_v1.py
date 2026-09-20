#!/usr/bin/env python3
"""Forensic, outcome-free audit of same-week defensive-front GSIS/team conflicts.

This audit does not change the frozen Defensive Front Pairwise Cohesion V1 candidate.
It only exposes source semantics needed to decide whether the original
REJECTED_INTEGRITY disposition is mechanically repairable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.football_context import qualify_ol_roster_continuity_v1 as ol
from scripts.football_context.qualify_defensive_front_pairwise_cohesion_v1 import (
    FRONT_POSITIONS,
    SEASONS,
    _first_col,
    _lower,
)

EXPECTED_WEEKLY_ROSTER_SHA256 = "f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a"

DIAGNOSTIC_CANDIDATES = [
    "full_name", "football_name", "player_name", "player", "name",
    "team", "team_abbr", "club_code",
    "position", "pos", "depth_chart_position", "depth_position",
    "status", "status_description", "roster_status",
    "date", "game_date", "timestamp", "updated_at", "last_modified",
    "entry_year", "rookie_year", "years_exp", "birth_date",
    "draft_club", "draft_number",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(v):
    if pd.isna(v):
        return ""
    return str(v)


def build_conflict_audit(roster: pd.DataFrame, schedule: pd.DataFrame):
    raw = _lower(roster)
    source_columns = list(raw.columns)

    team_col = _first_col(raw, ["team", "team_abbr", "club_code"])
    gsis_col = _first_col(raw, ["gsis_id", "player_id", "player_gsis_id"])
    pos_col = _first_col(raw, ["position", "pos"])
    depth_col = _first_col(raw, ["depth_chart_position", "depth_position"])
    if not {"season", "week"}.issubset(raw.columns) or not team_col or not gsis_col or not pos_col:
        raise RuntimeError("weekly roster missing season/week/team/GSIS/position")

    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
    raw["_canon_team"] = raw[team_col].map(canon_team)
    raw["_norm_gsis"] = raw[gsis_col].map(ol._norm_gsis)
    raw["_position_norm"] = raw[pos_col].map(ol._norm_pos)
    raw["_depth_position_norm"] = raw[depth_col].map(ol._norm_pos) if depth_col else ""

    eligible = (
        raw["_position_norm"].isin(FRONT_POSITIONS)
        | raw["_depth_position_norm"].isin(FRONT_POSITIONS)
    )
    q = raw.loc[
        raw["season"].isin(SEASONS)
        & raw["week"].gt(0)
        & eligible
        & raw["_norm_gsis"].ne("")
    ].copy()

    sched = schedule[["season", "week", "team"]].drop_duplicates().copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched["team"] = sched["team"].map(canon_team)
    q = q.merge(
        sched.rename(columns={"team": "_canon_team"}),
        on=["season", "week", "_canon_team"],
        how="inner",
        validate="many_to_one",
    )

    counts = q.groupby(["season", "week", "_norm_gsis"])["_canon_team"].nunique()
    bad_index = counts.loc[counts.gt(1)].index
    if len(bad_index) == 0:
        conflicts = q.iloc[0:0].copy()
    else:
        keys = pd.DataFrame(list(bad_index), columns=["season", "week", "_norm_gsis"])
        conflicts = q.merge(keys, on=["season", "week", "_norm_gsis"], how="inner", validate="many_to_one")

    chosen = [c for c in DIAGNOSTIC_CANDIDATES if c in conflicts.columns]
    identity_cols = ["season", "week", "_norm_gsis", "_canon_team", "_position_norm", "_depth_position_norm"]
    export_cols = []
    for c in identity_cols + chosen:
        if c not in export_cols:
            export_cols.append(c)

    conflict_rows = conflicts[export_cols].copy() if len(conflicts) else pd.DataFrame(columns=export_cols)
    conflict_rows = conflict_rows.sort_values(
        ["season", "week", "_norm_gsis", "_canon_team"]
    ).reset_index(drop=True)

    groups = []
    for (season, week, gsis), g in conflict_rows.groupby(["season", "week", "_norm_gsis"], sort=True):
        prior = q.loc[
            q["_norm_gsis"].eq(gsis)
            & (
                q["season"].lt(season)
                | (q["season"].eq(season) & q["week"].lt(week))
            )
        ].sort_values(["season", "week"]).tail(4)
        same_season_prior = prior.loc[prior["season"].eq(season)]
        prior_team = str(same_season_prior.iloc[-1]["_canon_team"]) if len(same_season_prior) else (
            str(prior.iloc[-1]["_canon_team"]) if len(prior) else ""
        )

        teams = sorted(g["_canon_team"].astype(str).unique().tolist())
        statuses = sorted({
            _scalar(x) for x in g["status"].tolist()
        }) if "status" in g.columns else []
        dates = sorted({
            _scalar(x) for c in ["date", "game_date", "timestamp", "updated_at", "last_modified"]
            if c in g.columns for x in g[c].tolist() if _scalar(x)
        })
        names = sorted({
            _scalar(x) for c in ["full_name", "football_name", "player_name", "player", "name"]
            if c in g.columns for x in g[c].tolist() if _scalar(x)
        })
        groups.append({
            "season": int(season),
            "week": int(week),
            "gsis_id": str(gsis),
            "teams": "|".join(teams),
            "team_count": int(len(teams)),
            "source_row_count": int(len(g)),
            "prior_roster_team_strictly_before_week": prior_team,
            "source_status_values": "|".join(statuses),
            "source_time_values": "|".join(dates),
            "source_names": "|".join(names),
        })
    summary = pd.DataFrame(groups)

    metadata = {
        "source_columns": source_columns,
        "diagnostic_columns_present": [c for c in DIAGNOSTIC_CANDIDATES if c in source_columns],
        "timing_columns_present": [
            c for c in ["date", "game_date", "timestamp", "updated_at", "last_modified"]
            if c in source_columns
        ],
        "status_columns_present": [
            c for c in ["status", "status_description", "roster_status"]
            if c in source_columns
        ],
        "conflict_group_count": int(len(summary)),
        "conflict_source_row_count": int(len(conflict_rows)),
        "target_game_outcomes_read": False,
        "target_game_pbp_read": False,
        "target_game_snap_or_participation_read": False,
        "future_week_used_to_resolve_conflict": False,
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
    }
    return conflict_rows, summary, metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--git-sha", required=True)
    args = ap.parse_args()

    roster_sha = sha256(args.roster)
    schedule_sha = sha256(args.schedule)
    schedule = ol.normalize_schedule(pd.read_csv(args.schedule, low_memory=False))
    roster = pd.read_csv(args.roster, low_memory=False)
    rows, groups, meta = build_conflict_audit(roster, schedule)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "def_front_same_week_conflict_rows_v1.csv", index=False)
    groups.to_csv(args.out_dir / "def_front_same_week_conflict_groups_v1.csv", index=False)

    manifest = {
        "audit_version": "DEFENSIVE_FRONT_SAME_WEEK_IDENTITY_CONFLICT_AUDIT_V1",
        "git_sha": args.git_sha,
        "weekly_roster_sha256": roster_sha,
        "expected_weekly_roster_sha256": EXPECTED_WEEKLY_ROSTER_SHA256,
        "weekly_roster_hash_matches_original_qualification": roster_sha == EXPECTED_WEEKLY_ROSTER_SHA256,
        "schedule_sha256": schedule_sha,
        **meta,
    }
    (args.out_dir / "def_front_same_week_conflict_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("CONFLICT GROUPS")
    print(groups.to_string(index=False))
    print("\nCONFLICT ROWS")
    print(rows.to_string(index=False))
    print("\nMANIFEST")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if not manifest["weekly_roster_hash_matches_original_qualification"]:
        raise RuntimeError(
            "rehydrated weekly roster hash differs from original qualification; "
            "preserve REJECTED_INTEGRITY and do not infer repair semantics from changed bytes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
