#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
FIELDS = [
    "avg_separation",
    "avg_cushion",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
    "catch_percentage",
]
PLAYER_FIELDS = ["player_display_name", "player_name", "player_short_name"]
TEAM_FIELDS = ["team_abbr", "team", "club"]


def one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def first_present(columns: list[str], choices: list[str]) -> str | None:
    return next((c for c in choices if c in columns), None)


def regular_week_mask(df: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(df["season"], errors="coerce")
    week = pd.to_numeric(df["week"], errors="coerce")
    max_week = season.map(lambda s: 17 if s == 2020 else 18)
    return season.isin(SEASONS) & week.ge(1) & week.le(max_week)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    try:
        import nflreadpy as nfl
    except Exception as exc:
        raise RuntimeError(f"nflreadpy unavailable: {exc}") from exc

    ref = pd.read_csv(one(args.wr_r1_root, "wr_r1_paired_wr_casebook.csv"), low_memory=False)
    ref.columns = [str(c).strip().lower() for c in ref.columns]
    needed_ref = {"season", "week", "team", "player_clean_key"}
    if needed_ref - set(ref.columns):
        raise RuntimeError(f"WR-R1 reference missing {sorted(needed_ref-set(ref.columns))}")
    ref["season"] = pd.to_numeric(ref["season"], errors="coerce")
    ref["week"] = pd.to_numeric(ref["week"], errors="coerce")
    ref["team"] = ref["team"].map(canon_team)
    ref["player_key"] = ref["player_clean_key"].map(key)
    ref = ref.loc[regular_week_mask(ref), ["season", "week", "team", "player_key"]].drop_duplicates().reset_index(drop=True)

    ngs_raw = nfl.load_nextgen_stats(seasons=SEASONS, stat_type="receiving")
    if hasattr(ngs_raw, "to_pandas"):
        ngs = ngs_raw.to_pandas()
    else:
        ngs = pd.DataFrame(ngs_raw)
    ngs.columns = [str(c).strip().lower() for c in ngs.columns]
    schema = list(ngs.columns)

    player_col = first_present(schema, PLAYER_FIELDS)
    team_col = first_present(schema, TEAM_FIELDS)
    week_exists = "week" in schema
    season_exists = "season" in schema

    integrity = {
        "season_field_exists": bool(season_exists),
        "week_field_exists": bool(week_exists),
        "player_field": player_col,
        "team_field": team_col,
    }

    if not season_exists or not week_exists or player_col is None or team_col is None:
        result = {
            "migration": "WR_R9_NGS_SOURCE_AUDIT",
            "source": "nflreadpy.load_nextgen_stats(receiving)",
            "schema": schema,
            "integrity": integrity,
            "sportsbook_inputs_used": False,
            "model_fitting_used": False,
            "production_changed": False,
            "disposition": "NGS_RECEIVING_SOURCE_INELIGIBLE",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "wr_r9_ngs_source_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        pd.DataFrame({"column": schema}).to_csv(args.out_dir / "wr_r9_ngs_schema.csv", index=False)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    ngs["season"] = pd.to_numeric(ngs["season"], errors="coerce")
    ngs["week"] = pd.to_numeric(ngs["week"], errors="coerce")
    if "season_type" in ngs.columns:
        ngs = ngs.loc[ngs["season_type"].fillna("").astype(str).str.upper().eq("REG")].copy()
    ngs = ngs.loc[regular_week_mask(ngs)].copy()
    ngs["player_key"] = ngs[player_col].map(key)
    ngs["team"] = ngs[team_col].map(canon_team)
    ngs = ngs.loc[ngs["player_key"].ne("") & ngs["team"].notna()].copy()

    keys = ["season", "week", "team", "player_key"]
    dup_mask = ngs.duplicated(keys, keep=False)
    duplicate_rows = int(dup_mask.sum())
    duplicate_rate = float(duplicate_rows / len(ngs)) if len(ngs) else 1.0
    ngs_unique = ngs.sort_values(keys, kind="stable").drop_duplicates(keys, keep="first").copy()

    season_rows = (
        ngs_unique.groupby("season", dropna=False).size().rename("ngs_rows").reset_index()
    )
    week_rows = (
        ngs_unique.groupby(["season", "week"], dropna=False).size().rename("ngs_rows").reset_index()
    )
    seasons_present = sorted(int(v) for v in ngs_unique["season"].dropna().unique() if int(v) in SEASONS)
    all_seasons_present = seasons_present == SEASONS
    weekly_variation = bool(all(ngs_unique.loc[ngs_unique["season"].eq(y), "week"].nunique() >= 8 for y in SEASONS)) if all_seasons_present else False

    keep = keys + [f for f in FIELDS if f in ngs_unique.columns]
    joined = ref.merge(ngs_unique[keep], on=keys, how="left", indicator=True, validate="one_to_one")
    joined["ngs_matched"] = joined["_merge"].eq("both")
    joined = joined.drop(columns=["_merge"])
    pooled_join_coverage = float(joined["ngs_matched"].mean()) if len(joined) else 0.0

    coverage_rows = []
    for y in SEASONS:
        g = joined.loc[joined["season"].eq(y)]
        coverage_rows.append({
            "season": y,
            "wr_r1_player_games": int(len(g)),
            "ngs_matched_player_games": int(g["ngs_matched"].sum()),
            "join_coverage": float(g["ngs_matched"].mean()) if len(g) else 0.0,
        })
    coverage = pd.DataFrame(coverage_rows)
    every_season_join_ge_50 = bool((coverage["join_coverage"] >= .50).all())

    field_rows = []
    matched = joined.loc[joined["ngs_matched"]].copy()
    eligible_fields_80 = 0
    for field in FIELDS:
        exists = field in ngs_unique.columns
        pooled_nonnull = float(pd.to_numeric(matched[field], errors="coerce").notna().mean()) if exists and len(matched) else 0.0
        if exists and pooled_nonnull >= .80:
            eligible_fields_80 += 1
        field_rows.append({
            "season": "POOLED",
            "field": field,
            "exists": bool(exists),
            "nonnull_coverage_matched": pooled_nonnull,
        })
        for y in SEASONS:
            gy = matched.loc[matched["season"].eq(y)]
            nonnull = float(pd.to_numeric(gy[field], errors="coerce").notna().mean()) if exists and len(gy) else 0.0
            field_rows.append({
                "season": y,
                "field": field,
                "exists": bool(exists),
                "nonnull_coverage_matched": nonnull,
            })
    field_cov = pd.DataFrame(field_rows)

    ngs_unique["ordinal"] = ngs_unique["season"] * 100 + ngs_unique["week"]
    first_obs = ngs_unique.groupby("player_key")["ordinal"].min()
    joined["ordinal"] = joined["season"] * 100 + joined["week"]
    joined["first_ngs_ordinal"] = joined["player_key"].map(first_obs)
    joined["has_earlier_ngs_observation"] = joined["first_ngs_ordinal"].notna() & joined["first_ngs_ordinal"].lt(joined["ordinal"])
    prior_eligible_rate = float(joined["has_earlier_ngs_observation"].mean()) if len(joined) else 0.0

    unmatched_ref = joined.loc[~joined["ngs_matched"], keys].head(100).copy()
    ngs_join_keys = ref[keys].copy()
    unmatched_ngs = ngs_unique.merge(ngs_join_keys, on=keys, how="left", indicator=True)
    unmatched_ngs = unmatched_ngs.loc[unmatched_ngs["_merge"].eq("left_only"), keys + [player_col]].head(100).copy()

    gates = {
        "all_six_seasons_present": bool(all_seasons_present),
        "week_field_and_variation": bool(week_exists and weekly_variation),
        "duplicate_rate_le_0_01": bool(duplicate_rate <= .01),
        "pooled_join_coverage_ge_0_60": bool(pooled_join_coverage >= .60),
        "every_season_join_coverage_ge_0_50": bool(every_season_join_ge_50),
        "four_fields_80pct_nonnull": bool(eligible_fields_80 >= 4),
        "prior_observation_eligibility_ge_0_50": bool(prior_eligible_rate >= .50),
    }
    integrity_pass = gates["all_six_seasons_present"] and gates["week_field_and_variation"] and gates["duplicate_rate_le_0_01"]
    if all(gates.values()):
        disposition = "NGS_RECEIVING_SOURCE_MULTISEASON_ELIGIBLE"
    elif integrity_pass:
        disposition = "NGS_RECEIVING_SOURCE_PARTIAL_ONLY"
    else:
        disposition = "NGS_RECEIVING_SOURCE_INELIGIBLE"

    result = {
        "migration": "WR_R9_NGS_SOURCE_AUDIT",
        "source": "nflreadpy.load_nextgen_stats(receiving)",
        "target_seasons": SEASONS,
        "reference_player_games": int(len(ref)),
        "ngs_regular_season_rows": int(len(ngs)),
        "ngs_unique_player_games": int(len(ngs_unique)),
        "seasons_present": seasons_present,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rate,
        "pooled_join_coverage": pooled_join_coverage,
        "min_single_season_join_coverage": float(coverage["join_coverage"].min()) if len(coverage) else 0.0,
        "fields_existing_with_80pct_matched_nonnull": int(eligible_fields_80),
        "prior_observation_eligible_player_games": int(joined["has_earlier_ngs_observation"].sum()),
        "prior_observation_eligibility_rate": prior_eligible_rate,
        "integrity": integrity,
        "gates": gates,
        "schema": schema,
        "sportsbook_inputs_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": disposition,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"column": schema}).to_csv(args.out_dir / "wr_r9_ngs_schema.csv", index=False)
    season_rows.to_csv(args.out_dir / "wr_r9_ngs_rows_by_season.csv", index=False)
    week_rows.to_csv(args.out_dir / "wr_r9_ngs_rows_by_week.csv", index=False)
    coverage.to_csv(args.out_dir / "wr_r9_ngs_join_coverage.csv", index=False)
    field_cov.to_csv(args.out_dir / "wr_r9_ngs_field_coverage.csv", index=False)
    joined.to_csv(args.out_dir / "wr_r9_reference_join_casebook.csv", index=False)
    unmatched_ref.to_csv(args.out_dir / "wr_r9_unmatched_wr_reference_sample.csv", index=False)
    unmatched_ngs.to_csv(args.out_dir / "wr_r9_unmatched_ngs_sample.csv", index=False)
    (args.out_dir / "wr_r9_ngs_source_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nJOIN COVERAGE")
    print(coverage.to_string(index=False))
    print("\nFIELD COVERAGE")
    print(field_cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
