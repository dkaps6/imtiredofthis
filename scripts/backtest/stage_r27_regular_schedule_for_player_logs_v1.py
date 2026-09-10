#!/usr/bin/env python3
"""Mechanical R27 repair: stage canonical REG-week schedule scope for player logs only.

This helper does not alter any retained cell value.  It removes only requested-season
rows whose week lies outside the known NFL regular-season week range used by the
frozen R27 study: 1..17 through 2020, 1..18 from 2021 onward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _canonical_json_hash(df: pd.DataFrame) -> str:
    payload = df.to_json(orient="split", index=False, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _max_reg_week(season: int) -> int:
    return 17 if int(season) <= 2020 else 18


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--seasons", required=True, help="Comma-separated requested seasons")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()

    if not args.schedule.exists() or args.schedule.stat().st_size == 0:
        raise RuntimeError(f"missing/empty schedule: {args.schedule}")

    requested = sorted({int(x.strip()) for x in str(args.seasons).split(",") if x.strip()})
    if not requested:
        raise RuntimeError("no requested seasons")

    # Strings preserve source cell semantics through the mechanical staging seam.
    src = pd.read_csv(args.schedule, dtype=str, keep_default_na=False)
    if src.empty:
        raise RuntimeError("schedule is empty")
    for col in ("season", "week"):
        if col not in src.columns:
            raise RuntimeError(f"schedule missing required column {col}")

    season_num = pd.to_numeric(src["season"], errors="coerce")
    week_num = pd.to_numeric(src["week"], errors="coerce")
    requested_mask = season_num.isin(requested)
    if requested_mask.sum() == 0:
        raise RuntimeError(f"schedule contains no requested seasons: {requested}")
    if season_num.loc[requested_mask].isna().any() or week_num.loc[requested_mask].isna().any():
        raise RuntimeError("requested-season schedule contains nonnumeric season/week")

    in_scope = pd.Series(True, index=src.index)
    for season in requested:
        mask = season_num.eq(season)
        max_week = _max_reg_week(season)
        in_scope.loc[mask] = week_num.loc[mask].between(1, max_week, inclusive="both")

    remove_mask = requested_mask & ~in_scope
    staged = src.loc[~remove_mask].copy()

    # Exact retained-row semantic identity check: same index subset, same columns, same strings.
    retained_source = src.loc[~remove_mask].copy()
    if list(retained_source.columns) != list(staged.columns):
        raise RuntimeError("staging changed schedule columns")
    if not retained_source.reset_index(drop=True).equals(staged.reset_index(drop=True)):
        raise RuntimeError("staging changed one or more retained schedule cell values")

    staged_season = pd.to_numeric(staged["season"], errors="coerce")
    staged_week = pd.to_numeric(staged["week"], errors="coerce")
    for season in requested:
        max_week = _max_reg_week(season)
        bad = staged.loc[staged_season.eq(season) & ~staged_week.between(1, max_week, inclusive="both")]
        if not bad.empty:
            raise RuntimeError(f"staged schedule still contains out-of-scope {season} rows")

        src_good = src.loc[season_num.eq(season) & week_num.between(1, max_week, inclusive="both")].copy()
        staged_good = staged.loc[staged_season.eq(season)].copy()
        if len(src_good) != len(staged_good) or not src_good.reset_index(drop=True).equals(staged_good.reset_index(drop=True)):
            raise RuntimeError(f"in-scope schedule rows changed/disappeared for season {season}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(args.out, index=False)

    # Re-read to prove CSV round-trip retains semantic values for all retained rows.
    reread = pd.read_csv(args.out, dtype=str, keep_default_na=False)
    if not staged.reset_index(drop=True).equals(reread.reset_index(drop=True)):
        raise RuntimeError("staged schedule CSV round-trip changed retained cell values")

    removed = src.loc[remove_mask].copy()
    identity_cols = [c for c in ("season", "week", "team", "opponent", "game_type", "event_id", "game_id") if c in src.columns]
    removed_records = removed[identity_cols].to_dict("records") if identity_cols else []

    by_season = []
    for season in requested:
        ssrc = int(season_num.eq(season).sum())
        srem = int((season_num.eq(season) & remove_mask).sum())
        sstage = int(pd.to_numeric(staged["season"], errors="coerce").eq(season).sum())
        by_season.append({
            "season": season,
            "max_regular_week": _max_reg_week(season),
            "source_rows": ssrc,
            "staged_rows": sstage,
            "removed_out_of_scope_rows": srem,
        })

    audit = {
        "repair": "R27_RUN1_HISTORICAL_SCHEDULE_SCOPE_MECHANICAL_REPAIR_V1",
        "requested_seasons": requested,
        "source_rows": int(len(src)),
        "staged_rows": int(len(staged)),
        "removed_rows": int(remove_mask.sum()),
        "retained_rows_exact": True,
        "retained_columns_exact": True,
        "roundtrip_values_exact": True,
        "football_values_transformed": False,
        "fuzzy_identity_used": False,
        "source_semantic_hash": _canonical_json_hash(src),
        "retained_source_semantic_hash": _canonical_json_hash(retained_source.reset_index(drop=True)),
        "staged_semantic_hash": _canonical_json_hash(staged.reset_index(drop=True)),
        "by_season": by_season,
        "removed_identities": removed_records,
    }
    if audit["retained_source_semantic_hash"] != audit["staged_semantic_hash"]:
        raise RuntimeError("retained/staged semantic hash mismatch")

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
