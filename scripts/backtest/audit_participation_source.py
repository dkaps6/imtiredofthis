#!/usr/bin/env python3
"""Audit nflverse participation as a historical defensive-context source.

Diagnostic only. This script does not change any model inputs or projections.
It measures whether nflverse participation can support leakage-safe historical
box/coverage reconstruction and whether play rows can be joined back to PBP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["defenders_in_box", "defense_man_zone_type", "defense_coverage_type"]


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
        s = series.astype("string").str.strip()
        return s.notna() & s.ne("") & s.str.lower().ne("nan")
    return pd.to_numeric(series, errors="coerce").notna()


def _derive_season_week(participation: pd.DataFrame) -> pd.DataFrame:
    x = _lower(participation)
    if "season" not in x.columns or "week" not in x.columns:
        gid_col = next((c for c in ["nflverse_game_id", "game_id"] if c in x.columns), None)
        if gid_col is not None:
            parts = x[gid_col].astype("string").str.split("_", expand=True)
            if "season" not in x.columns and parts.shape[1] >= 1:
                x["season"] = pd.to_numeric(parts[0], errors="coerce")
            if "week" not in x.columns and parts.shape[1] >= 2:
                x["week"] = pd.to_numeric(parts[1], errors="coerce")
    return x


def _join_keys(participation: pd.DataFrame, pbp: pd.DataFrame) -> tuple[list[str], str]:
    p, b = _lower(participation), _lower(pbp)
    candidates = [
        (["nflverse_game_id", "play_id"], "nflverse_game_id+play_id"),
        (["old_game_id", "play_id"], "old_game_id+play_id"),
        (["game_id", "play_id"], "game_id+play_id"),
    ]
    for keys, label in candidates:
        if all(k in p.columns and k in b.columns for k in keys):
            return keys, label
    return [], "no_common_game_play_key"


def audit_frames(participation: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = _derive_season_week(participation)
    b = _lower(pbp)
    total = int(len(p))
    seasons = sorted(pd.to_numeric(p.get("season"), errors="coerce").dropna().astype(int).unique().tolist()) if "season" in p.columns else []
    weeks = sorted(pd.to_numeric(p.get("week"), errors="coerce").dropna().astype(int).unique().tolist()) if "week" in p.columns else []

    rows = [
        {"metric": "participation_rows", "value": total, "status": "ok" if total else "empty"},
        {"metric": "seasons", "value": ",".join(map(str, seasons)), "status": "ok" if seasons else "missing"},
        {"metric": "weeks", "value": ",".join(map(str, weeks)), "status": "ok" if weeks else "missing"},
    ]
    for field in FIELDS:
        if field not in p.columns:
            rows.append({"metric": f"field_{field}", "value": 0.0, "status": "missing_column"})
        else:
            available = int(_present(p[field]).sum())
            rows.append({"metric": f"field_{field}", "value": float(available / total) if total else np.nan, "status": f"available_rows={available}"})

    keys, label = _join_keys(p, b)
    if keys:
        left = p[keys].dropna().drop_duplicates()
        right = b[keys].dropna().drop_duplicates()
        joined = left.merge(right, on=keys, how="inner")
        match_rate = float(len(joined) / len(left)) if len(left) else np.nan
        dup_rate = float(p.duplicated(keys, keep=False).mean()) if len(p) else np.nan
        rows += [
            {"metric": "pbp_join_key", "value": label, "status": "ok"},
            {"metric": "pbp_join_match_rate", "value": match_rate, "status": f"matched_keys={len(joined)}/{len(left)}"},
            {"metric": "participation_duplicate_key_rate", "value": dup_rate, "status": "diagnostic"},
        ]
    else:
        rows += [
            {"metric": "pbp_join_key", "value": label, "status": "unsupported"},
            {"metric": "pbp_join_match_rate", "value": np.nan, "status": "not_tested"},
        ]

    go = all(field in p.columns and _present(p[field]).any() for field in FIELDS)
    rows.append({"metric": "source_recommendation", "value": "GO" if go else "NO_GO", "status": "all_three_fields_populated" if go else "one_or_more_fields_unavailable"})

    by_week_rows = []
    if {"season", "week"}.issubset(p.columns):
        for (season, week), g in p.groupby(["season", "week"], dropna=True):
            rec = {"season": int(season), "week": int(week), "rows": int(len(g))}
            for field in FIELDS:
                rec[f"{field}_coverage"] = float(_present(g[field]).mean()) if field in g.columns and len(g) else 0.0
            by_week_rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(by_week_rows)


def load_sources(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl

    participation = _to_pandas(nfl.load_participation(seasons=seasons))
    pbp = _to_pandas(nfl.load_pbp(seasons=seasons))
    return participation, pbp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2024,2025")
    parser.add_argument("--summary", type=Path, default=Path("data/backtests/participation_source_audit.csv"))
    parser.add_argument("--by-week", type=Path, default=Path("data/backtests/participation_field_coverage_by_week.csv"))
    args = parser.parse_args()
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    participation, pbp = load_sources(seasons)
    summary, by_week = audit_frames(participation, pbp)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    by_week.to_csv(args.by_week, index=False)
    print("[participation_audit] summary")
    print(summary.to_string(index=False))
    if not by_week.empty:
        print("\n[participation_audit] field coverage by week")
        print(by_week.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
