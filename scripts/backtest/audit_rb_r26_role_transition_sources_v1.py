#!/usr/bin/env python3
"""R26 source audit for leakage-safe RB role-transition state.

This script deliberately does NOT load player outcomes, target-game participation,
model residuals, sportsbook information, or any scoring label.  It inventories
historical weekly roster/depth semantics and asks the existing historical
pregame-universe builder what role information it can safely expose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.backtest.historical_inputs import (
    _load_nflreadpy_weekly_sources,
    build_pregame_universe_for_week,
    build_schedule_history,
)

RB_POSITIONS = {"RB", "HB", "FB"}
DATE_TOKENS = ("date", "time", "timestamp", "updated", "modified", "dt")

EXPOSURE = {
    2014: "PRIOR_INPUT_ONLY_IN_LATER_STUDIES",
    2015: "R9_SCORED",
    2016: "R9_SCORED",
    2017: "R8_R9_SCORED",
    2018: "R8_SCORED",
    2019: "R8_SCORED",
    2020: "R25_SCORED",
    2021: "R10_R12_R25_SCORED",
    2022: "R10_R12_R25_SCORED",
    2023: "R10_R12_R23_R24_SCORED",
    2024: "R10_R12_R23_R24_SCORED",
    2025: "R10_R12_R23_R24_SCORED",
}


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _first(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _schema_fields(df: pd.DataFrame, kind: str) -> dict[str, str]:
    cols = list(df.columns)
    week_cols = [c for c in cols if c == "week" or c.endswith("_week") or "week" in c]
    date_cols = [c for c in cols if any(tok in c for tok in DATE_TOKENS)]
    status_cols = [c for c in cols if "status" in c]
    depth_cols = [c for c in cols if "depth" in c]
    position_cols = [c for c in cols if c == "position" or c == "pos" or "position" in c]
    team_cols = [c for c in cols if c in {"team", "club_code", "team_abbr", "team_abbreviation", "club"}]
    name_cols = [c for c in cols if c in {"full_name", "football_name", "player_name", "player", "name", "display_name"}]
    id_cols = [c for c in cols if any(k in c for k in ("gsis", "player_id", "nfl_id", "espn_id", "pfr_id"))]
    return {
        "kind": kind,
        "week_cols": ";".join(week_cols),
        "date_cols": ";".join(date_cols),
        "status_cols": ";".join(status_cols),
        "depth_cols": ";".join(depth_cols),
        "position_cols": ";".join(position_cols),
        "team_cols": ";".join(team_cols),
        "name_cols": ";".join(name_cols),
        "id_cols": ";".join(id_cols),
        "all_columns": json.dumps(cols),
    }


def _position_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    pos = _first(df, ["position", "pos", "depth_position", "position_name", "depth_chart_position"])
    if pos is None:
        # If several position-like columns exist, use the first one that actually
        # contains an RB/HB/FB token.  This is schema discovery only.
        candidates = [c for c in df.columns if "position" in c]
        for c in candidates:
            s = df[c].astype(str).str.upper().str.strip()
            if s.isin(RB_POSITIONS).any():
                pos = c
                break
    if pos is None:
        return 0
    return int(df[pos].astype(str).str.upper().str.strip().isin(RB_POSITIONS).sum())


def _depth_timing_class(depth: pd.DataFrame) -> str:
    if depth.empty:
        return "UNAVAILABLE"
    cols = set(depth.columns)
    if {"season", "week"}.issubset(cols):
        return "WEEK_TAGGED"
    date_cols = [c for c in depth.columns if any(tok in c for tok in DATE_TOKENS)]
    if date_cols:
        return "DATE_BEARING"
    return "NO_BOUNDARY"


def _safe_mean(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def audit_season(season: int) -> tuple[dict, list[dict], list[dict]]:
    schedule = build_schedule_history([int(season)])
    season_schedule = schedule.loc[pd.to_numeric(schedule["season"], errors="coerce").eq(int(season))].copy()
    weeks = sorted(pd.to_numeric(season_schedule["week"], errors="coerce").dropna().astype(int).unique().tolist())

    roster_error = ""
    depth_error = ""
    try:
        rosters, depth = _load_nflreadpy_weekly_sources(int(season))
        rosters = _lower(rosters)
        depth = _lower(depth)
    except Exception as exc:
        # Retry rosters directly so a depth-provider failure does not erase
        # roster availability from the audit.
        import nflreadpy as nfl

        try:
            raw = nfl.load_rosters_weekly(int(season))
            rosters = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
            rosters = _lower(rosters)
        except Exception as rex:
            rosters = pd.DataFrame()
            roster_error = repr(rex)
        depth = pd.DataFrame()
        depth_error = repr(exc)

    schema_rows: list[dict] = []
    for kind, frame, err in (("weekly_rosters", rosters, roster_error), ("depth_charts", depth, depth_error)):
        row = {"season": int(season), "rows": int(len(frame)), "error": err}
        row.update(_schema_fields(frame, kind))
        schema_rows.append(row)

    timing = _depth_timing_class(depth)
    week_rows: list[dict] = []
    universe_parts = []
    week_errors = []
    for week in weeks:
        try:
            u = build_pregame_universe_for_week(
                season=int(season),
                week=int(week),
                schedule_history=schedule,
                rosters_weekly=rosters,
                depth_charts=depth,
            )
            u = _lower(u)
            rb = u.loc[u["position"].astype(str).str.upper().str.strip().isin(RB_POSITIONS)].copy()
            role = rb.get("role", pd.Series("", index=rb.index, dtype=object)).fillna("").astype(str).str.strip()
            source = rb.get("pregame_source", pd.Series("", index=rb.index, dtype=object)).fillna("").astype(str)
            week_rows.append({
                "season": int(season),
                "week": int(week),
                "rb_universe_rows": int(len(rb)),
                "rb_role_nonblank": int(role.ne("").sum()),
                "rb_role_coverage": _safe_mean(role.ne("")),
                "depth_merge_rows": int(source.str.contains("depth_chart", case=False, na=False).sum()),
                "pregame_source_values": ";".join(sorted(source[source.ne("")].unique().tolist())),
                "error": "",
            })
            universe_parts.append(rb.assign(_role_nonblank=role.ne("")))
        except Exception as exc:
            week_errors.append(f"W{int(week)}:{type(exc).__name__}:{exc}")
            week_rows.append({
                "season": int(season),
                "week": int(week),
                "rb_universe_rows": 0,
                "rb_role_nonblank": 0,
                "rb_role_coverage": np.nan,
                "depth_merge_rows": 0,
                "pregame_source_values": "",
                "error": repr(exc),
            })

    if universe_parts:
        all_rb = pd.concat(universe_parts, ignore_index=True)
        role_coverage = _safe_mean(all_rb["_role_nonblank"])
        rb_universe_rows = int(len(all_rb))
        role_nonblank = int(all_rb["_role_nonblank"].sum())
    else:
        all_rb = pd.DataFrame()
        role_coverage = float("nan")
        rb_universe_rows = 0
        role_nonblank = 0

    w1 = pd.DataFrame(week_rows)
    w1 = w1.loc[w1["week"].eq(1)] if not w1.empty else w1

    roster_status_cols = [c for c in rosters.columns if "status" in c]
    depth_role_cols = [c for c in depth.columns if "depth" in c or c in {"position", "pos"}]

    summary = {
        "season": int(season),
        "historical_scoring_exposure": EXPOSURE.get(int(season), "UNMAPPED"),
        "regular_season_weeks": int(len(weeks)),
        "weekly_roster_rows": int(len(rosters)),
        "weekly_roster_rb_rows": _position_count(rosters),
        "roster_status_fields": ";".join(roster_status_cols),
        "depth_rows": int(len(depth)),
        "depth_rb_rows": _position_count(depth),
        "depth_timing_class": timing,
        "depth_role_fields": ";".join(depth_role_cols),
        "rb_pregame_universe_rows": rb_universe_rows,
        "rb_role_nonblank": role_nonblank,
        "rb_role_coverage": role_coverage,
        "week1_rb_rows": int(w1["rb_universe_rows"].sum()) if not w1.empty else 0,
        "week1_role_nonblank": int(w1["rb_role_nonblank"].sum()) if not w1.empty else 0,
        "week1_role_coverage": (
            float(w1["rb_role_nonblank"].sum() / w1["rb_universe_rows"].sum())
            if not w1.empty and int(w1["rb_universe_rows"].sum()) > 0
            else float("nan")
        ),
        "weeks_with_universe_error": int(sum(bool(r["error"]) for r in week_rows)),
        "universe_errors": " | ".join(week_errors),
        "source_contract_note": (
            "week-tagged depth eligible for later same-week timing semantics review"
            if timing == "WEEK_TAGGED"
            else "date-based depth deliberately not merged without as-of-before-kickoff proof"
            if timing == "DATE_BEARING"
            else "no timing-safe depth source established"
        ),
    }
    return summary, schema_rows, week_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2014-2025")
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r26_role_transition_source_audit"))
    args = ap.parse_args()

    token = str(args.seasons).strip()
    if "-" in token and "," not in token:
        a, b = token.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = sorted({int(x.strip()) for x in token.split(",") if x.strip()})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    schemas = []
    weeks = []
    for season in seasons:
        summary, schema_rows, week_rows = audit_season(int(season))
        summaries.append(summary)
        schemas.extend(schema_rows)
        weeks.extend(week_rows)
        print(
            f"[r26-source] {season}: timing={summary['depth_timing_class']} "
            f"rb_rows={summary['rb_pregame_universe_rows']} "
            f"role_cov={summary['rb_role_coverage']} errors={summary['weeks_with_universe_error']}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("season")
    schema_df = pd.DataFrame(schemas).sort_values(["season", "kind"])
    week_df = pd.DataFrame(weeks).sort_values(["season", "week"])
    exposure_df = pd.DataFrame(
        [{"season": s, "historical_scoring_exposure": EXPOSURE.get(s, "UNMAPPED")} for s in seasons]
    )

    summary_df.to_csv(args.out_dir / "r26_role_source_summary.csv", index=False)
    schema_df.to_csv(args.out_dir / "r26_role_source_schema.csv", index=False)
    week_df.to_csv(args.out_dir / "r26_role_source_weekly_coverage.csv", index=False)
    exposure_df.to_csv(args.out_dir / "r26_temporal_exposure_ledger.csv", index=False)

    disposition = {
        "candidate": "RB_R26_ROLE_TRANSITION_ENTITLEMENT_V1",
        "stage": "SOURCE_AUDIT_ONLY",
        "scientific_candidate_scored": False,
        "player_outcomes_loaded": False,
        "target_game_participation_used": False,
        "sportsbook_inputs_used": False,
        "production_parameters_changed": False,
        "seasons": seasons,
        "week_tagged_depth_seasons": summary_df.loc[
            summary_df["depth_timing_class"].eq("WEEK_TAGGED"), "season"
        ].astype(int).tolist(),
        "date_bearing_depth_seasons": summary_df.loc[
            summary_df["depth_timing_class"].eq("DATE_BEARING"), "season"
        ].astype(int).tolist(),
        "role_coverage_by_season": {
            str(int(r.season)): (None if pd.isna(r.rb_role_coverage) else float(r.rb_role_coverage))
            for r in summary_df.itertuples()
        },
        "week1_role_coverage_by_season": {
            str(int(r.season)): (None if pd.isna(r.week1_role_coverage) else float(r.week1_role_coverage))
            for r in summary_df.itertuples()
        },
        "temporal_freshness_statement": (
            "No untouched historical RB-receiving scoring block remains across the audited modern lineage; "
            "any subsequent historical R26 test must be labeled predeclared retrospective mechanism confirmation."
        ),
    }
    (args.out_dir / "r26_source_audit_disposition.json").write_text(
        json.dumps(disposition, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(disposition, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
