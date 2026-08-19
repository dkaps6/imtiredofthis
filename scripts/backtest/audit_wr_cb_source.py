#!/usr/bin/env python3
"""Audit historical sources for receiver-to-defender matchup responsibility.

Migration 15 is diagnostic-only. It determines whether nflverse participation/PBP
contains enough explicit information to reconstruct historical WR-CB assignments
without pretending that every defensive back on the field covered every receiver.

The audit distinguishes three levels of evidence:
1. participation/on-field evidence (receiver and defenders were on the field),
2. target-play evidence (PBP identifies the targeted receiver),
3. assignment evidence (an explicit defender/responsibility field identifies who
   covered that receiver on the play).

Only level 3 is sufficient for a true historical WR-CB matchup reconstruction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_CANDIDATES = [
    "coverage_defender_id",
    "coverage_defender_name",
    "defender_id",
    "defender_name",
    "primary_defender_id",
    "primary_defender_name",
    "target_defender_id",
    "target_defender_name",
    "nearest_defender_id",
    "nearest_defender_name",
    "coverage_responsibility_id",
    "coverage_responsibility_name",
]
RECEIVER_CANDIDATES = [
    "receiver_player_id",
    "receiver_player_name",
    "receiver_id",
    "receiver_name",
    "target_player_id",
    "target_player_name",
]
PARTICIPATION_PLAYER_COLUMNS = [
    "offense_players",
    "offense_names",
    "defense_players",
    "defense_names",
    "offense_positions",
    "defense_positions",
]


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _lower(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _present(series: pd.Series) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        x = series.astype("string").str.strip()
        return x.notna() & x.ne("") & x.str.lower().ne("nan")
    return pd.to_numeric(series, errors="coerce").notna()


def _join_keys(participation: pd.DataFrame, pbp: pd.DataFrame) -> tuple[list[str], str]:
    candidates = [
        (["old_game_id", "play_id"], "old_game_id+play_id"),
        (["nflverse_game_id", "play_id"], "nflverse_game_id+play_id"),
        (["game_id", "play_id"], "game_id+play_id"),
    ]
    for keys, label in candidates:
        if all(k in participation.columns and k in pbp.columns for k in keys):
            return keys, label
    return [], "no_common_game_play_key"


def audit_frames(participation: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = _lower(participation)
    b = _lower(pbp)
    rows: list[dict] = []

    rows.append({"metric": "participation_rows", "value": len(p), "status": "ok" if len(p) else "empty"})
    rows.append({"metric": "pbp_rows", "value": len(b), "status": "ok" if len(b) else "empty"})

    for field in PARTICIPATION_PLAYER_COLUMNS:
        if field in p.columns:
            available = int(_present(p[field]).sum())
            rows.append({"metric": f"participation_{field}", "value": available / len(p) if len(p) else np.nan, "status": f"available_rows={available}"})
        else:
            rows.append({"metric": f"participation_{field}", "value": 0.0, "status": "missing_column"})

    receiver_fields = []
    for field in RECEIVER_CANDIDATES:
        if field in b.columns:
            available = int(_present(b[field]).sum())
            receiver_fields.append(field)
            rows.append({"metric": f"pbp_{field}", "value": available / len(b) if len(b) else np.nan, "status": f"available_rows={available}"})

    assignment_fields = []
    for field in ASSIGNMENT_CANDIDATES:
        source_name = None
        source = None
        if field in p.columns:
            source_name, source = "participation", p
        elif field in b.columns:
            source_name, source = "pbp", b
        if source is None:
            continue
        available = int(_present(source[field]).sum())
        assignment_fields.append((field, source_name, available))
        rows.append({"metric": f"assignment_{field}", "value": available / len(source) if len(source) else np.nan, "status": f"source={source_name};available_rows={available}"})

    keys, label = _join_keys(p, b)
    if keys:
        left = p[keys].dropna().drop_duplicates()
        right = b[keys].dropna().drop_duplicates()
        matched = left.merge(right, on=keys, how="inner")
        rate = len(matched) / len(left) if len(left) else np.nan
        rows.append({"metric": "participation_pbp_join_key", "value": label, "status": "ok"})
        rows.append({"metric": "participation_pbp_join_rate", "value": rate, "status": f"matched={len(matched)}/{len(left)}"})
    else:
        rows.append({"metric": "participation_pbp_join_key", "value": label, "status": "unsupported"})
        rows.append({"metric": "participation_pbp_join_rate", "value": np.nan, "status": "not_tested"})

    has_on_field = any(c in p.columns and _present(p[c]).any() for c in ["defense_players", "defense_names"])
    has_target_receiver = any(_present(b[c]).any() for c in receiver_fields)
    has_assignment = any(n > 0 for _, _, n in assignment_fields)

    if has_assignment and has_target_receiver:
        recommendation = "GO_TRUE_ASSIGNMENT"
        reason = "explicit_receiver_and_defender_assignment_fields_available"
    elif has_on_field and has_target_receiver:
        recommendation = "NO_GO_TRUE_ASSIGNMENT"
        reason = "on_field_defenders_and_target_receiver_available_but_no_explicit_coverage_responsibility"
    else:
        recommendation = "NO_GO"
        reason = "insufficient_receiver_or_defender_evidence"
    rows.append({"metric": "wr_cb_source_recommendation", "value": recommendation, "status": reason})

    keywords = ("receiver", "target", "defender", "coverage", "corner", "cb", "route", "align")
    inventory = []
    for source_name, frame in [("participation", p), ("pbp", b)]:
        for col in frame.columns:
            low = str(col).lower()
            if any(k in low for k in keywords):
                available = int(_present(frame[col]).sum())
                inventory.append({"source": source_name, "column": col, "available_rows": available, "total_rows": len(frame), "coverage": available / len(frame) if len(frame) else np.nan})

    by_week = []
    if {"season", "week"}.issubset(b.columns):
        for (season, week), g in b.groupby(["season", "week"], dropna=True):
            rec = {"season": int(season), "week": int(week), "pbp_rows": len(g)}
            rec["target_receiver_coverage"] = max((_present(g[c]).mean() for c in receiver_fields if c in g.columns), default=0.0)
            direct = [field for field, source_name, _ in assignment_fields if source_name == "pbp" and field in g.columns]
            rec["explicit_assignment_coverage"] = max((_present(g[c]).mean() for c in direct), default=0.0)
            by_week.append(rec)

    return pd.DataFrame(rows), pd.DataFrame(inventory), pd.DataFrame(by_week)


def load_sources(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl
    return _to_pandas(nfl.load_participation(seasons=seasons)), _to_pandas(nfl.load_pbp(seasons=seasons))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2024,2025")
    parser.add_argument("--summary", type=Path, default=Path("data/backtests/wr_cb_source_audit.csv"))
    parser.add_argument("--inventory", type=Path, default=Path("data/backtests/wr_cb_candidate_columns.csv"))
    parser.add_argument("--by-week", type=Path, default=Path("data/backtests/wr_cb_source_coverage_by_week.csv"))
    args = parser.parse_args()
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    participation, pbp = load_sources(seasons)
    summary, inventory, by_week = audit_frames(participation, pbp)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    inventory.to_csv(args.inventory, index=False)
    by_week.to_csv(args.by_week, index=False)
    print("[wr_cb_source_audit] summary")
    print(summary.to_string(index=False))
    print("\n[wr_cb_source_audit] candidate columns")
    print(inventory.to_string(index=False) if not inventory.empty else "none")
    if not by_week.empty:
        print("\n[wr_cb_source_audit] by week")
        print(by_week.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
