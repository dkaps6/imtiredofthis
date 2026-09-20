#!/usr/bin/env python3
"""Outcome-free qualification for BDB2026 targeted-receiver release history.

Frozen plan:
docs/research/BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1.md

This script:
- uses only the certified strict-prior BDB history summaries;
- bridges BDB nfl_id directly through nflverse player IDs to canonical GSIS identity;
- profiles broad 2023 WR/TE/RB player-game pregame coverage;
- recomputes source geometry persistence;
- audits reconstructibility from canonical PlayerForm target-opportunity state;
- never evaluates receiving yards, receptions, sportsbook lines, residuals, or bets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.build_context_signal_qualification_inventory import qualify
from scripts.research.build_role_room_redundancy_audit import (
    build_production_opportunity_state,
)

CANDIDATES = {
    "hist_receiver_release_nearest_defender_median_yards":
        "receiver_release_nearest_defender_distance_yards",
    "hist_receiver_release_second_defender_median_yards":
        "receiver_release_second_defender_distance_yards",
}
SUPPORT = "hist_receiver_release_geometry_sample_count"
PROD = [
    "prod_tgt_prior_share",
    "prod_tgt_prior_games",
    "prod_tgt_current_share",
    "prod_tgt_current_games",
    "prod_tgt_playerform_blend",
]
POSITIONS = {"WR", "TE", "RB"}
SOURCE_HASH = "228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07"
MIN_SOURCE_SUPPORT = 8
MIN_COVERAGE = 0.80
MIN_ELIGIBLE_ROWS = 500
MIN_STABLE_ID_COVERAGE = 0.99
REDUNDANCY_HIGH = 0.90
REDUNDANCY_REVIEW = 0.75
REDUNDANCY_TRAIN_MIN = 500
REDUNDANCY_HOLDOUT_MIN = 200

HISTORY_KEY = ["season", "week", "team", "player_identity_key"]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _norm_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    # Kaggle tracking IDs are often serialized as numeric floats.
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def _norm_gsis(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"),
         "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].corr(z["b"], method="spearman"))


def build_direct_identity_bridge(
    raw_bdb: pd.DataFrame,
    players: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """BDB nfl_id -> nflverse gsis_id -> canonical historical identity.

    Names are deliberately not accepted as a fallback.
    """
    raw = _lower(raw_bdb)
    p = _lower(players)
    h = _lower(history)

    if "nfl_id" not in raw.columns:
        raise RuntimeError("BDB raw materialization missing nfl_id")

    nfl_col = _first_col(
        p,
        ["nfl_id", "nfl_player_id", "nflid", "nfl_player_id_number"],
    )
    gsis_col = _first_col(
        p,
        ["gsis_id", "player_id", "player_gsis_id"],
    )
    if not nfl_col or not gsis_col:
        raise RuntimeError(
            "nflverse player crosswalk lacks direct nfl_id/gsis_id columns; "
            f"columns={sorted(p.columns.tolist())}"
        )

    source_ids = pd.Series(raw["nfl_id"].map(_norm_id).unique(), dtype="object")
    source_ids = source_ids[source_ids.ne("")]

    cross = p[[nfl_col, gsis_col]].copy()
    cross["bdb_nfl_id"] = cross[nfl_col].map(_norm_id)
    cross["gsis_id"] = cross[gsis_col].map(_norm_gsis)
    cross = cross.loc[
        cross["bdb_nfl_id"].ne("") & cross["gsis_id"].ne(""),
        ["bdb_nfl_id", "gsis_id"],
    ].drop_duplicates()

    nfl_ambiguity = (
        cross.groupby("bdb_nfl_id")["gsis_id"].nunique().gt(1)
        if len(cross) else pd.Series(dtype=bool)
    )
    ambiguous_nfl_ids = sorted(nfl_ambiguity[nfl_ambiguity].index.tolist())
    safe_cross = cross.loc[
        ~cross["bdb_nfl_id"].isin(ambiguous_nfl_ids)
    ].drop_duplicates("bdb_nfl_id")

    required_history = {"player_id", "player_identity_key"}
    missing = required_history - set(h.columns)
    if missing:
        raise RuntimeError(f"history missing canonical identity columns: {sorted(missing)}")
    hist_map = h[["player_id", "player_identity_key"]].copy()
    hist_map["gsis_id"] = hist_map["player_id"].map(_norm_gsis)
    hist_map = hist_map.loc[
        hist_map["gsis_id"].ne("") & hist_map["player_identity_key"].notna(),
        ["gsis_id", "player_identity_key"],
    ].drop_duplicates()

    gsis_ambiguity = (
        hist_map.groupby("gsis_id")["player_identity_key"].nunique().gt(1)
        if len(hist_map) else pd.Series(dtype=bool)
    )
    ambiguous_gsis_ids = sorted(gsis_ambiguity[gsis_ambiguity].index.tolist())
    safe_hist = hist_map.loc[
        ~hist_map["gsis_id"].isin(ambiguous_gsis_ids)
    ].drop_duplicates("gsis_id")

    bridge = safe_cross.merge(
        safe_hist, on="gsis_id", how="inner", validate="one_to_one"
    )
    mapped_source_ids = set(bridge["bdb_nfl_id"].astype(str))
    total = int(len(source_ids))
    mapped = int(sum(str(v) in mapped_source_ids for v in source_ids))
    coverage = float(mapped / total) if total else 0.0

    report = {
        "player_crosswalk_nfl_id_column": nfl_col,
        "player_crosswalk_gsis_id_column": gsis_col,
        "bdb_unique_targeted_receiver_ids": total,
        "bdb_ids_mapped_to_canonical_gsis_identity": mapped,
        "direct_stable_id_bridge_coverage": coverage,
        "ambiguous_nfl_id_count": int(len(ambiguous_nfl_ids)),
        "ambiguous_canonical_gsis_count": int(len(ambiguous_gsis_ids)),
        "name_fallback_used": False,
    }
    return bridge, report


def map_history_snapshots(
    snapshots: pd.DataFrame,
    bridge: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    s = _lower(snapshots)
    required = {
        "target_week", "nfl_id", "history_max_source_week", SUPPORT, *CANDIDATES.keys()
    }
    missing = required - set(s.columns)
    if missing:
        raise RuntimeError(f"BDB receiver history missing columns: {sorted(missing)}")
    s["bdb_nfl_id"] = s["nfl_id"].map(_norm_id)
    s["target_week"] = pd.to_numeric(s["target_week"], errors="coerce").astype("Int64")
    s["history_max_source_week"] = pd.to_numeric(
        s["history_max_source_week"], errors="coerce"
    ).astype("Int64")
    s[SUPPORT] = pd.to_numeric(s[SUPPORT], errors="coerce")
    out = s.merge(
        bridge[["bdb_nfl_id", "player_identity_key"]],
        on="bdb_nfl_id",
        how="left",
        validate="many_to_one",
    )
    comparable = out["target_week"].notna() & out["history_max_source_week"].notna()
    chronology_violations = int(
        (out.loc[comparable, "history_max_source_week"]
         >= out.loc[comparable, "target_week"]).sum()
    )
    support_violations = int(
        (
            out[list(CANDIDATES.keys())].notna().any(axis=1)
            & out[SUPPORT].lt(MIN_SOURCE_SUPPORT)
        ).sum()
    )
    mapped = out.loc[out["player_identity_key"].notna()].copy()
    duplicate_rows = int(
        mapped.duplicated(["target_week", "player_identity_key"], keep=False).sum()
    )
    return mapped, {
        "chronology_violations": chronology_violations,
        "materializer_support_threshold_violations": support_violations,
        "mapped_snapshot_duplicate_rows": duplicate_rows,
    }


def build_broad_universe(
    history: pd.DataFrame,
    mapped_snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    h = _lower(history)
    required = {
        *HISTORY_KEY, "player_id", "position", "targets", "team_targets"
    }
    missing = required - set(h.columns)
    if missing:
        raise RuntimeError(f"history missing qualification columns: {sorted(missing)}")
    if h.duplicated(HISTORY_KEY).any():
        raise RuntimeError("canonical history contains duplicate player-game keys")

    h["season"] = pd.to_numeric(h["season"], errors="coerce")
    h["week"] = pd.to_numeric(h["week"], errors="coerce")
    h["position"] = h["position"].astype(str).str.upper().str.strip()
    universe = h.loc[
        h["season"].eq(2023) & h["position"].isin(POSITIONS)
    ].copy()
    universe["stable_identity_flag"] = (
        universe["player_id"].map(_norm_gsis).ne("")
        & universe["player_identity_key"].astype(str).str.startswith("gsis:")
    ).astype(int)

    snap = mapped_snapshots.copy()
    snap = snap[[
        "target_week", "player_identity_key", SUPPORT, "history_max_source_week",
        *CANDIDATES.keys()
    ]]
    fanout_rows = int(
        snap.duplicated(["target_week", "player_identity_key"], keep=False).sum()
    )
    if fanout_rows:
        # Scientific outputs must not be computed from an ambiguous join.
        # Preserve the count in the manifest and fail closed.
        raise RuntimeError(
            f"mapped BDB snapshots would fan out canonical player-weeks: {fanout_rows}"
        )

    before = len(universe)
    joined = universe.merge(
        snap,
        left_on=["week", "player_identity_key"],
        right_on=["target_week", "player_identity_key"],
        how="left",
        validate="many_to_one",
    )
    if len(joined) != before:
        raise RuntimeError("BDB history join changed broad-universe row count")
    joined["bdb_history_unknown_flag"] = (
        joined[list(CANDIDATES.keys())].isna().all(axis=1)
    ).astype(int)

    report = {
        "eligible_universe_rows": int(len(universe)),
        "eligible_universe_players": int(universe["player_identity_key"].nunique()),
        "stable_identity_coverage": (
            float(universe["stable_identity_flag"].mean()) if len(universe) else 0.0
        ),
        "canonical_duplicate_key_rows": int(
            universe.duplicated(HISTORY_KEY, keep=False).sum()
        ),
        "canonical_join_fanout_rows": 0,
        "week1_rows_in_denominator": int(universe["week"].eq(1).sum()),
    }
    return joined, report


def source_persistence(raw_bdb: pd.DataFrame) -> pd.DataFrame:
    raw = _lower(raw_bdb)
    required = {"week", "nfl_id", *CANDIDATES.values()}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"BDB raw geometry missing persistence columns: {sorted(missing)}")
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
    rows: list[dict[str, object]] = []
    for candidate, raw_metric in CANDIDATES.items():
        q = raw[["nfl_id", "week", raw_metric]].copy()
        q[raw_metric] = pd.to_numeric(q[raw_metric], errors="coerce")
        early = (
            q.loc[q["week"].between(1, 9)]
            .dropna(subset=[raw_metric])
            .groupby("nfl_id")[raw_metric]
            .agg(["median", "count"])
            .reset_index()
            .rename(columns={"median": "early_value", "count": "early_n"})
        )
        late = (
            q.loc[q["week"].between(10, 18)]
            .dropna(subset=[raw_metric])
            .groupby("nfl_id")[raw_metric]
            .agg(["median", "count"])
            .reset_index()
            .rename(columns={"median": "late_value", "count": "late_n"})
        )
        z = early.merge(late, on="nfl_id", how="inner")
        z = z.loc[
            z["early_n"].ge(MIN_SOURCE_SUPPORT)
            & z["late_n"].ge(MIN_SOURCE_SUPPORT)
        ].copy()
        rows.append({
            "feature_name": candidate,
            "raw_geometry_metric": raw_metric,
            "stability_stat": "player_early_late_median_spearman_weeks1_9_vs10_18",
            "stability_value": _safe_spearman(z["early_value"], z["late_value"]),
            "stability_pairs": int(len(z)),
            "minimum_observations_each_half": MIN_SOURCE_SUPPORT,
            "box_score_outcomes_read": False,
        })
    return pd.DataFrame(rows)


def coverage_by_week_position(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (week, position), g in joined.groupby(["week", "position"], sort=True):
        rec: dict[str, object] = {
            "season": 2023,
            "week": int(week),
            "position": str(position),
            "eligible_rows": int(len(g)),
        }
        for candidate in CANDIDATES:
            n = int(g[candidate].notna().sum())
            rec[f"{candidate}__observed_rows"] = n
            rec[f"{candidate}__coverage"] = float(n / len(g)) if len(g) else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def _holdout_r2(
    frame: pd.DataFrame,
    candidate: str,
) -> tuple[float, int, int]:
    z = frame[["week", candidate, *PROD]].copy()
    z[candidate] = pd.to_numeric(z[candidate], errors="coerce")
    for col in PROD:
        z[col] = pd.to_numeric(z[col], errors="coerce").fillna(0.0)
    z = z.dropna(subset=[candidate])
    train = z.loc[z["week"].between(1, 9)].copy()
    holdout = z.loc[z["week"].between(10, 18)].copy()
    if len(train) < REDUNDANCY_TRAIN_MIN or len(holdout) < REDUNDANCY_HOLDOUT_MIN:
        return np.nan, int(len(train)), int(len(holdout))
    xtr = np.column_stack([np.ones(len(train)), train[PROD].to_numpy(float)])
    xte = np.column_stack([np.ones(len(holdout)), holdout[PROD].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(xtr, train[candidate].to_numpy(float), rcond=None)
    y = holdout[candidate].to_numpy(float)
    pred = xte @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum((y - pred) ** 2)) / sst
        if sst > 0 else np.nan
    )
    return r2, int(len(train)), int(len(holdout))


def redundancy_audit(
    history: pd.DataFrame,
    joined: pd.DataFrame,
) -> pd.DataFrame:
    prod = build_production_opportunity_state(history)
    frame = joined.drop(
        columns=[c for c in joined.columns if c.startswith("prod_")],
        errors="ignore",
    ).merge(
        prod,
        on=HISTORY_KEY,
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        r2, train_n, holdout_n = _holdout_r2(frame, candidate)
        observed = frame.loc[frame[candidate].notna()].copy()
        rho = _safe_spearman(
            observed[candidate], observed["prod_tgt_playerform_blend"]
        )
        if not np.isfinite(r2):
            disposition = "REDUNDANCY_UNRESOLVED_SOURCE_THIN"
        elif r2 >= REDUNDANCY_HIGH:
            disposition = "HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
        elif r2 >= REDUNDANCY_REVIEW:
            disposition = "REDUNDANCY_REVIEW"
        else:
            disposition = "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
        rows.append({
            "feature_name": candidate,
            "production_domain": "tgt",
            "production_blend_spearman": rho,
            "holdout_reconstructibility_r2": r2,
            "reconstruction_train_rows": train_n,
            "reconstruction_holdout_rows": holdout_n,
            "train_weeks": "2023 W1-W9",
            "holdout_weeks": "2023 W10-W18",
            "redundancy_disposition": disposition,
            "box_score_outcomes_read": False,
            "sportsbook_read": False,
        })
    return pd.DataFrame(rows)


def build_profiles(
    joined: pd.DataFrame,
    stability: pd.DataFrame,
    redundancy: pd.DataFrame,
    identity: dict[str, object],
    integrity: dict[str, object],
) -> pd.DataFrame:
    st = stability.set_index("feature_name")
    rd = redundancy.set_index("feature_name")
    eligible_rows = int(len(joined))
    stable_coverage = (
        float(joined["stable_identity_flag"].mean()) if eligible_rows else 0.0
    )
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        observed = joined[candidate].notna()
        support = pd.to_numeric(
            joined.loc[observed, SUPPORT], errors="coerce"
        ).dropna()
        redundancy_disp = str(rd.loc[candidate, "redundancy_disposition"])
        row = pd.Series({
            "feature_name": candidate,
            "family": "BDB2026_RECEIVER_RELEASE_GEOMETRY_V1",
            "grain": "player_game",
            "seasons_available": "2023",
            "eligible_rows": eligible_rows,
            "pregame_coverage": (
                float(observed.sum() / eligible_rows) if eligible_rows else 0.0
            ),
            "stable_id_coverage": stable_coverage,
            "unknown_rate": (
                float((~observed).sum() / eligible_rows) if eligible_rows else 1.0
            ),
            "duplicate_key_count": int(
                joined.duplicated(HISTORY_KEY, keep=False).sum()
            ),
            "fanout_count": int(integrity.get("canonical_join_fanout_rows", 0)),
            "prior_support_median": (
                float(support.median()) if len(support) else 0.0
            ),
            "stability_stat": st.loc[candidate, "stability_stat"],
            "stability_value": st.loc[candidate, "stability_value"],
            "intended_component":
                "receiver spatial environment, target-quality/efficiency context, and uncertainty",
            "redundancy_notes": redundancy_disp,
            "source_class": "CERTIFIED_BDB2026_RESEARCH_ACCESS",
            "mechanism_note":
                "strict-prior targeted-receiver release spacing may describe a persistent "
                "spatial environment not captured by target-share opportunity state",
        })
        disposition, reason = qualify(row, MIN_COVERAGE, MIN_ELIGIBLE_ROWS)

        hard_integrity_ok = (
            float(identity["direct_stable_id_bridge_coverage"])
            >= MIN_STABLE_ID_COVERAGE
            and int(identity["ambiguous_nfl_id_count"]) == 0
            and int(identity["ambiguous_canonical_gsis_count"]) == 0
            and int(integrity["chronology_violations"]) == 0
            and int(integrity["materializer_support_threshold_violations"]) == 0
            and int(integrity["mapped_snapshot_duplicate_rows"]) == 0
            and int(integrity["canonical_duplicate_key_rows"]) == 0
            and int(integrity["canonical_join_fanout_rows"]) == 0
        )
        if not hard_integrity_ok:
            disposition = "REJECTED_INTEGRITY"
            reason = "direct identity, chronology, support-threshold, duplicate, or fanout integrity failure"
        elif redundancy_disp == "HIGHLY_RECONSTRUCTIBLE_REDUNDANT":
            disposition = "DESCRIPTIVE_ONLY"
            reason = "candidate is highly reconstructible from canonical PlayerForm target-opportunity state"
        elif redundancy_disp == "REDUNDANCY_UNRESOLVED_SOURCE_THIN":
            if disposition == "READY_FOR_FROZEN_EXPERIMENT":
                disposition = "ENGINEERING_READY_SOURCE_THIN"
                reason = "insufficient outcome-free rows to resolve redundancy on the frozen temporal split"

        rows.append({
            **row.to_dict(),
            "stability_pairs": int(st.loc[candidate, "stability_pairs"]),
            "production_blend_spearman": rd.loc[
                candidate, "production_blend_spearman"
            ],
            "holdout_reconstructibility_r2": rd.loc[
                candidate, "holdout_reconstructibility_r2"
            ],
            "reconstruction_train_rows": int(
                rd.loc[candidate, "reconstruction_train_rows"]
            ),
            "reconstruction_holdout_rows": int(
                rd.loc[candidate, "reconstruction_holdout_rows"]
            ),
            "qualification_disposition": disposition,
            "qualification_reason": reason,
        })
    return pd.DataFrame(rows)


def run(
    history_path: Path,
    players_path: Path,
    materialized_dir: Path,
    out_dir: Path,
    *,
    active_sha: str,
    materializer_sha: str,
) -> dict[str, object]:
    private = materialized_dir / "private"
    sanitized = materialized_dir / "sanitized"
    raw_path = private / "bdb2026_targeted_receiver_throw_window_v1.csv"
    hist_path = private / "bdb2026_receiver_history_snapshots_v1.csv"
    qa_path = sanitized / "bdb2026_throw_window_advanced_feature_qa_v1.json"
    if not qa_path.exists():
        # Certified helper names from SOURCE_KEY lowercasing.
        qa_path = sanitized / "bdb2026_throw_window_advanced_feature_qa_v1.json"
    for path in [history_path, players_path, raw_path, hist_path, qa_path]:
        if not path.exists():
            raise RuntimeError(f"required qualification input missing: {path}")

    qa = json.loads(qa_path.read_text())
    observed_hash = str(qa.get("source_hash_observed", ""))
    if observed_hash != SOURCE_HASH or not qa.get("source_hash_verified", False):
        raise RuntimeError(
            f"certified BDB source hash mismatch: {observed_hash} != {SOURCE_HASH}"
        )
    if int(qa.get("temporal_audit", {}).get("receiver_history", {}).get("violations", -1)) != 0:
        raise RuntimeError("certified materializer reports receiver-history temporal violations")
    if int(qa.get("temporal_audit", {}).get("target_game_rows_used", -1)) != 0:
        raise RuntimeError("certified materializer reports target-game rows in pregame history")
    if int(
        qa.get("temporal_audit", {}).get(
            "landing_and_postrelease_fields_pregame_used", -1
        )
    ) != 0:
        raise RuntimeError("certified materializer reports post-release pregame use")

    history = pd.read_csv(history_path, low_memory=False)
    players = pd.read_csv(players_path, low_memory=False)
    raw = pd.read_csv(raw_path, low_memory=False)
    snapshots = pd.read_csv(hist_path, low_memory=False)

    bridge, identity = build_direct_identity_bridge(raw, players, history)
    mapped, temporal = map_history_snapshots(snapshots, bridge)
    joined, universe = build_broad_universe(history, mapped)
    integrity = {**temporal, **universe}

    stability = source_persistence(raw)
    redundancy = redundancy_audit(history, joined)
    profiles = build_profiles(joined, stability, redundancy, identity, integrity)
    coverage = coverage_by_week_position(joined)

    out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out_dir / "bdb2026_receiver_release_geometry_coverage_v1.csv", index=False)
    stability.to_csv(out_dir / "bdb2026_receiver_release_geometry_stability_v1.csv", index=False)
    redundancy.to_csv(out_dir / "bdb2026_receiver_release_geometry_redundancy_v1.csv", index=False)
    profiles.to_csv(out_dir / "bdb2026_receiver_release_geometry_qualification_inventory_v1.csv", index=False)

    dispositions = dict(
        zip(profiles["feature_name"], profiles["qualification_disposition"])
    )
    manifest = {
        "qualification_version":
            "BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1",
        "frozen_pre_result_disposition":
            "BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1_FROZEN_PRE_RESULT",
        "active_context_sha": active_sha,
        "certified_materializer_sha": materializer_sha,
        "source_hash_expected": SOURCE_HASH,
        "source_hash_observed": observed_hash,
        "source_hash_verified": True,
        "history_sha256": sha256(history_path),
        "players_crosswalk_sha256": sha256(players_path),
        "raw_geometry_ephemeral_sha256": sha256(raw_path),
        "history_snapshots_ephemeral_sha256": sha256(hist_path),
        "identity": identity,
        "integrity": integrity,
        "materializer_temporal_audit": qa.get("temporal_audit", {}),
        "broad_eligible_positions": sorted(POSITIONS),
        "broad_target_season": 2023,
        "broad_week1_retained": True,
        "materializer_min_prior_observations": MIN_SOURCE_SUPPORT,
        "qualification_min_pregame_coverage": MIN_COVERAGE,
        "qualification_min_eligible_rows": MIN_ELIGIBLE_ROWS,
        "qualification_min_stable_id_coverage": MIN_STABLE_ID_COVERAGE,
        "redundancy_train": "2023 W1-W9",
        "redundancy_holdout": "2023 W10-W18",
        "redundancy_high_r2": REDUNDANCY_HIGH,
        "redundancy_review_r2": REDUNDANCY_REVIEW,
        "box_score_outcomes_scored": False,
        "receiving_yards_read_for_candidate_selection": False,
        "receptions_read_for_candidate_selection": False,
        "sportsbook_read": False,
        "production_changed": False,
        "issue_535_touched": False,
        "nearest_defender_claimed_as_coverage_responsibility": False,
        "raw_competition_files_uploaded": False,
        "derived_per_row_bdb_files_uploaded": False,
        "candidate_dispositions": dispositions,
        "all_candidates_ready": bool(
            len(profiles)
            and profiles["qualification_disposition"]
            .eq("READY_FOR_FROZEN_EXPERIMENT").all()
        ),
        "any_candidate_ready": bool(
            profiles["qualification_disposition"]
            .eq("READY_FOR_FROZEN_EXPERIMENT").any()
        ),
    }
    (out_dir / "bdb2026_receiver_release_geometry_qualification_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print(profiles.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--players", type=Path, required=True)
    ap.add_argument("--materialized-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--active-sha", required=True)
    ap.add_argument("--materializer-sha", required=True)
    args = ap.parse_args()
    run(
        args.history,
        args.players,
        args.materialized_dir,
        args.out_dir,
        active_sha=args.active_sha,
        materializer_sha=args.materializer_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
