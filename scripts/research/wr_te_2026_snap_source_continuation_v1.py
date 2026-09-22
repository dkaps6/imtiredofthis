#!/usr/bin/env python3
"""WR/TE 2026 snap-source continuation audit.

Research-only. Does not modify production adapters or model parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling import te_r5p_entitlement_adapter_v1 as te


BASE_SEASONS = list(te.SOURCE_SEASONS)
TARGET_SEASON = 2026
CANDIDATE_SEASONS = list(range(min(BASE_SEASONS), TARGET_SEASON + 1))
FEATURE_COLS = [
    "prior_count_anyteam",
    "prior_count_same_team",
    "prior1_anyteam",
    "prior3_anyteam",
    "prior1_same_team",
    "prior3_same_team",
    "prior1_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps",
    "prior3_anyteam_offense_pct",
    "prior3_anyteam_offense_snaps",
    "prior1_same_team_offense_pct",
    "prior1_same_team_offense_snaps",
]


def _pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    if hasattr(obj, "to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    return pd.DataFrame(obj)


def _first(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(pd.NA, index=df.index)


def load_normalized(seasons: list[int]) -> tuple[pd.DataFrame, float, list[int]]:
    import nflreadpy as nfl

    q = _pandas(nfl.load_snap_counts(seasons=[int(v) for v in seasons]))
    q.columns = [str(c).strip().lower() for c in q.columns]
    if "season" not in q.columns or "week" not in q.columns:
        raise RuntimeError("snap source missing season/week")
    q["season"] = pd.to_numeric(q["season"], errors="coerce")
    q["week"] = pd.to_numeric(q["week"], errors="coerce")
    q = q.loc[q["season"].isin(seasons) & q["week"].between(1, 18)].copy()
    q["team"] = _first(q, ["team", "team_abbr", "club"]).map(te._team)
    q["player_key"] = _first(q, ["player", "player_name", "full_name"]).map(te._key)
    q["offense_pct"] = pd.to_numeric(
        _first(q, ["offense_pct", "offense_percentage"]), errors="coerce"
    )
    q["offense_snaps"] = pd.to_numeric(_first(q, ["offense_snaps"]), errors="coerce")
    pos = _first(q, ["position", "pos", "position_group"])
    q["position_audit"] = pos.astype("string").fillna("").str.upper().str.strip()
    q["ordinal"] = q["season"] * 100 + q["week"]
    q = q.loc[q["team"].ne("") & q["player_key"].ne("")].copy()
    keys = ["season", "week", "team", "player_key"]
    dup_rate = float(q.duplicated(keys, keep=False).mean()) if len(q) else 1.0
    q = q.sort_values(keys, kind="stable").drop_duplicates(keys, keep="last").reset_index(drop=True)
    got_seasons = sorted(int(v) for v in q["season"].dropna().unique())
    return q, dup_rate, got_seasons


def _parity_frame(q: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "season", "week", "team", "player_key",
        "offense_pct", "offense_snaps", "ordinal",
    ]
    return q[cols].sort_values(["season", "week", "team", "player_key"], kind="stable").reset_index(drop=True)


def assert_historical_parity(base: pd.DataFrame, candidate_through_2025: pd.DataFrame) -> dict:
    a = _parity_frame(base)
    b = _parity_frame(candidate_through_2025)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    payload = a.to_csv(index=False).encode("utf-8")
    return {
        "rows": int(len(a)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "max_abs_offense_pct_gap": 0.0,
        "max_abs_offense_snaps_gap": 0.0,
    }


def _target_frame(source: pd.DataFrame, *, target_week: int, source_weeks: list[int]) -> pd.DataFrame:
    q = source.loc[
        source["season"].eq(TARGET_SEASON)
        & source["week"].isin(source_weeks)
        & source["position_audit"].isin(["WR", "TE"])
    ].copy()
    if q.empty:
        raise RuntimeError(
            f"no WR/TE completed snap identities for target Week {target_week} source_weeks={source_weeks}"
        )
    q = q.sort_values(["season", "week"]).drop_duplicates(["team", "player_key"], keep="last")
    return pd.DataFrame({
        "player_clean_key": q["player_key"].astype(str),
        "team": q["team"].astype(str),
        "season": TARGET_SEASON,
        "week": int(target_week),
        "position": q["position_audit"].astype(str),
    }).reset_index(drop=True)


def _week1_synthetic_targets(base: pd.DataFrame) -> pd.DataFrame:
    q = base.loc[
        base["season"].eq(2025)
        & base["position_audit"].isin(["WR", "TE"])
    ].copy()
    if q.empty:
        raise RuntimeError("2025 baseline source contains no WR/TE position rows")
    latest_week = int(q["week"].max())
    q = q.loc[q["week"].eq(latest_week)].copy()
    q = q.drop_duplicates(["team", "player_key"], keep="last")
    return pd.DataFrame({
        "player_clean_key": q["player_key"].astype(str),
        "team": q["team"].astype(str),
        "season": TARGET_SEASON,
        "week": 1,
        "position": q["position_audit"].astype(str),
    }).reset_index(drop=True)


def _features(target: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    f = te._strict_prior_features(target, snaps)
    out = f[["player_clean_key", "team", "season", "week", "position", *FEATURE_COLS]].copy()
    return out.sort_values(["team", "player_clean_key"]).reset_index(drop=True)


def assert_week1_invariance(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    target_source: pd.DataFrame,
) -> dict:
    target = _week1_synthetic_targets(target_source)
    a = _features(target, base)
    b = _features(target, candidate)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    return {
        "target_rows": int(len(target)),
        "wr_rows": int(target["position"].eq("WR").sum()),
        "te_rows": int(target["position"].eq("TE").sum()),
        "feature_parity": True,
    }


def _prior_lineage(target: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in target.iterrows():
        pk = te._key(r["player_clean_key"])
        tm = te._team(r["team"])
        ordinal = int(r["season"]) * 100 + int(r["week"])
        hist = candidate.loc[
            candidate["player_key"].eq(pk)
            & candidate["ordinal"].lt(ordinal)
        ].copy()
        same = hist.loc[hist["team"].eq(tm)]
        hist26 = hist.loc[hist["season"].eq(TARGET_SEASON)]
        same26 = same.loc[same["season"].eq(TARGET_SEASON)]
        rows.append({
            "player_clean_key": r["player_clean_key"],
            "team": tm,
            "position": r["position"],
            "target_week": int(r["week"]),
            "max_anyteam_2026_week_used": (
                int(hist26["week"].max()) if not hist26.empty else 0
            ),
            "max_sameteam_2026_week_used": (
                int(same26["week"].max()) if not same26.empty else 0
            ),
        })
    return pd.DataFrame(rows)


def compare_target_week(
    *,
    target: pd.DataFrame,
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    target_week: int,
) -> tuple[dict, pd.DataFrame]:
    a = _features(target, base)
    b = _features(target, candidate)
    keys = ["player_clean_key", "team", "season", "week", "position"]
    joined = a.merge(b, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    lineage = _prior_lineage(target, candidate)
    joined = joined.merge(
        lineage,
        on=["player_clean_key", "team", "position"],
        how="left",
        validate="one_to_one",
    )
    max_allowed = int(target_week) - 1
    if (
        joined["max_anyteam_2026_week_used"].gt(max_allowed).any()
        or joined["max_sameteam_2026_week_used"].gt(max_allowed).any()
    ):
        raise RuntimeError(f"candidate snap features used target/future 2026 rows for Week {target_week}")

    gains_same = (
        ~joined["prior1_same_team_base"].astype(bool)
        & joined["prior1_same_team_candidate"].astype(bool)
    )
    gains_any = (
        ~joined["prior1_anyteam_base"].astype(bool)
        & joined["prior1_anyteam_candidate"].astype(bool)
    )
    changed = pd.Series(False, index=joined.index)
    for c in FEATURE_COLS:
        left = joined[f"{c}_base"]
        right = joined[f"{c}_candidate"]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right):
            changed |= left.fillna(False).astype(bool).ne(right.fillna(False).astype(bool))
        else:
            l = pd.to_numeric(left, errors="coerce")
            r = pd.to_numeric(right, errors="coerce")
            changed |= ~(np.isclose(l, r, equal_nan=True))
    return {
        "target_week": int(target_week),
        "target_rows": int(len(joined)),
        "wr_rows": int(joined["position"].eq("WR").sum()),
        "te_rows": int(joined["position"].eq("TE").sum()),
        "rows_with_any_feature_change": int(changed.sum()),
        "rows_gaining_prior1_same_team": int(gains_same.sum()),
        "rows_gaining_prior1_any_team": int(gains_any.sum()),
        "max_anyteam_2026_week_used": int(joined["max_anyteam_2026_week_used"].max()),
        "max_sameteam_2026_week_used": int(joined["max_sameteam_2026_week_used"].max()),
        "strict_prior_pass": True,
    }, joined


def model_hashes() -> dict:
    paths = {
        "te_r5p": Path("data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json"),
        "wr_r15": Path("data/models/wr_r15_production_model_v1/wr_r15_production_model_v1.json"),
    }
    out = {}
    for name, path in paths.items():
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing frozen model JSON: {path}")
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main() -> int:
    out_dir = Path("data/research/wr_te_2026_snap_source_continuation_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    base, base_dup, base_seasons = te._load_snaps()
    candidate_2025, cand25_dup, cand25_seasons = load_normalized(BASE_SEASONS)
    candidate, candidate_dup, candidate_seasons = load_normalized(CANDIDATE_SEASONS)

    if base_seasons != BASE_SEASONS:
        raise RuntimeError(f"frozen production source seasons drifted: {base_seasons}")
    if candidate_seasons != CANDIDATE_SEASONS:
        raise RuntimeError(f"candidate source seasons incomplete: {candidate_seasons}")

    historical = assert_historical_parity(base, candidate_2025)
    week1 = assert_week1_invariance(base, candidate, candidate_2025)

    source26 = candidate.loc[candidate["season"].eq(TARGET_SEASON)].copy()
    source_quality = {
        "rows": int(len(source26)),
        "weeks": sorted(int(v) for v in source26["week"].dropna().unique()),
        "teams": int(source26["team"].nunique()),
        "wr_rows": int(source26["position_audit"].eq("WR").sum()),
        "te_rows": int(source26["position_audit"].eq("TE").sum()),
        "offense_pct_nonnull": int(source26["offense_pct"].notna().sum()),
        "offense_snaps_nonnull": int(source26["offense_snaps"].notna().sum()),
        "candidate_duplicate_rate": float(candidate_dup),
        "baseline_duplicate_rate": float(base_dup),
        "candidate_2025_duplicate_rate": float(cand25_dup),
    }
    if source_quality["weeks"] != [1, 2] or source_quality["teams"] != 32:
        raise RuntimeError(f"2026 snap source not ready for Week 3: {source_quality}")

    week2_target = _target_frame(candidate, target_week=2, source_weeks=[1])
    week2, week2_trace = compare_target_week(
        target=week2_target, base=base, candidate=candidate, target_week=2
    )
    week3_target = _target_frame(candidate, target_week=3, source_weeks=[1, 2])
    week3, week3_trace = compare_target_week(
        target=week3_target, base=base, candidate=candidate, target_week=3
    )

    result = {
        "version": "WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1",
        "disposition": "WR_TE_2026_SNAP_SOURCE_CONTINUATION_READY",
        "base_source_seasons": BASE_SEASONS,
        "candidate_source_seasons": CANDIDATE_SEASONS,
        "historical_parity_2020_2025": historical,
        "week1_invariance": week1,
        "week2_strict_prior": week2,
        "week3_readiness": week3,
        "source_quality_2026": source_quality,
        "frozen_model_sha256": model_hashes(),
        "model_refit_performed": 0,
        "sportsbook_inputs_used": 0,
        "2026_outcomes_used": 0,
        "production_changed": 0,
    }
    week2_trace.to_csv(out_dir / "week2_source_continuation_trace.csv", index=False)
    week3_trace.to_csv(out_dir / "week3_source_continuation_trace.csv", index=False)
    (out_dir / "wr_te_2026_snap_source_continuation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
