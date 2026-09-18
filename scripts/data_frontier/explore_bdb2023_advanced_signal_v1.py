"""Advanced-data signal reconnaissance for BDB 2023 protection geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_frontier.advanced_data_signal_exploration_common_v1 import (
    confidence_tier,
    dataframe_records,
    numeric_correlation_matrix,
    pair_metrics,
    split_persistence,
    write_sanitized,
)

GEOMETRY_METRICS = [
    "blocker_target_snap_distance_yards",
    "blocker_target_min_distance_yards",
    "blocker_target_terminal_distance_yards",
    "blocker_target_time_to_min_distance_seconds",
    "protection_window_length_seconds",
]


def _future_validation(raw: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    actual = (
        raw.groupby(["week", "blocker_nfl_id"], dropna=False)
        .agg(
            actual_snap=("blocker_target_snap_distance_yards", "median"),
            actual_min=("blocker_target_min_distance_yards", "median"),
            actual_time_to_min=("blocker_target_time_to_min_distance_seconds", "median"),
        )
        .reset_index()
    )
    z = hist.merge(
        actual,
        left_on=["target_week", "blocker_nfl_id"],
        right_on=["week", "blocker_nfl_id"],
        how="inner",
    )
    specs = [
        ("hist_blocker_snap_distance_median_yards", "actual_snap", "snap_sample_count"),
        ("hist_blocker_min_distance_median_yards", "actual_min", "min_distance_sample_count"),
        ("hist_blocker_time_to_min_distance_median_seconds", "actual_time_to_min", "time_to_min_sample_count"),
    ]
    rows = []
    for pred, actual_col, count_col in specs:
        q = z.copy()
        q["confidence_tier"] = confidence_tier(q[count_col], 10, 20, 50)
        q = q.loc[q["confidence_tier"].ne("ABSTAIN")].copy()
        for tier in ["ALL", "LOW", "MEDIUM", "HIGH"]:
            g = q if tier == "ALL" else q.loc[q["confidence_tier"].eq(tier)]
            rows.append(
                {
                    "predictor": pred,
                    "actual_geometry": actual_col,
                    "confidence_tier": tier,
                    **pair_metrics(g, pred, actual_col),
                }
            )
    return pd.DataFrame(rows), int(len(z))


def _outcome_geometry(joined: pd.DataFrame) -> pd.DataFrame:
    outcome_cols = [
        "pff_beatenByDefender",
        "pff_hitAllowed",
        "pff_hurryAllowed",
        "pff_sackAllowed",
    ]
    rows = []
    pb = joined.loc[joined["block_interaction_role"].eq("Pass Block")].copy()
    for outcome in outcome_cols:
        if outcome not in pb.columns:
            continue
        y = pd.to_numeric(pb[outcome], errors="coerce")
        for metric in GEOMETRY_METRICS:
            x = pd.to_numeric(pb[metric], errors="coerce")
            q = pd.DataFrame({"y": y, "x": x}).dropna()
            q = q.loc[q["y"].isin([0, 1])]
            g0 = q.loc[q["y"].eq(0), "x"]
            g1 = q.loc[q["y"].eq(1), "x"]
            corr = q["x"].corr(q["y"], method="pearson") if len(q) >= 3 and q["y"].nunique() > 1 else np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "geometry_metric": metric,
                    "n": int(len(q)),
                    "positive_n": int(len(g1)),
                    "negative_n": int(len(g0)),
                    "positive_rate": float(q["y"].mean()) if len(q) else None,
                    "median_geometry_outcome_1": float(g1.median()) if len(g1) else None,
                    "median_geometry_outcome_0": float(g0.median()) if len(g0) else None,
                    "median_difference_1_minus_0": float(g1.median() - g0.median()) if len(g1) and len(g0) else None,
                    "point_biserial_corr": float(corr) if pd.notna(corr) else None,
                }
            )
    return pd.DataFrame(rows)


def run(materialized_dir: Path, corpus_dir: Path, out_dir: Path) -> dict:
    private = materialized_dir / "private"
    raw = pd.read_csv(private / "bdb2023_protection_interactions_v1.csv")
    hist = pd.read_csv(private / "bdb2023_blocker_history_snapshots_v1.csv")

    persistence, _ = split_persistence(
        raw,
        keys=["blocker_nfl_id"],
        metrics=[
            "blocker_target_snap_distance_yards",
            "blocker_target_min_distance_yards",
            "blocker_target_time_to_min_distance_seconds",
        ],
        early_mask=raw["week"].between(1, 4),
        late_mask=raw["week"].between(5, 8),
        min_obs_each=10,
    )

    future, matched_future = _future_validation(raw, hist)

    pff = pd.read_csv(next(corpus_dir.rglob("pffScoutingData.csv")))
    pff["blocked_nfl_id"] = pd.to_numeric(pff["pff_nflIdBlockedPlayer"], errors="coerce")
    pff["blocker_nfl_id"] = pd.to_numeric(pff["nflId"], errors="coerce")
    labels = pff[
        [
            "gameId",
            "playId",
            "blocker_nfl_id",
            "blocked_nfl_id",
            "pff_beatenByDefender",
            "pff_hitAllowed",
            "pff_hurryAllowed",
            "pff_sackAllowed",
        ]
    ].rename(columns={"gameId":"game_id","playId":"play_id"})
    labels = labels.drop_duplicates(["game_id","play_id","blocker_nfl_id","blocked_nfl_id"])
    joined = raw.merge(
        labels,
        on=["game_id","play_id","blocker_nfl_id","blocked_nfl_id"],
        how="left",
        validate="one_to_one",
    )
    outcome_geometry = _outcome_geometry(joined)

    block_type_summary = (
        joined.groupby(["block_interaction_role", "pff_blockType"], dropna=False)
        .agg(
            interactions=("play_id", "size"),
            median_snap_distance=("blocker_target_snap_distance_yards", "median"),
            median_min_distance=("blocker_target_min_distance_yards", "median"),
            median_terminal_distance=("blocker_target_terminal_distance_yards", "median"),
            median_time_to_min=("blocker_target_time_to_min_distance_seconds", "median"),
        )
        .reset_index()
    )

    coverage = (
        hist.assign(qualified=hist["min_distance_sample_count"] >= 10)
        .groupby("target_week")
        .agg(
            blocker_history_rows=("blocker_nfl_id","size"),
            qualified_rows=("qualified","sum"),
            median_prior_samples=("min_distance_sample_count","median"),
        )
        .reset_index()
    )
    coverage["qualified_rate"] = coverage["qualified_rows"] / coverage["blocker_history_rows"]

    corr = numeric_correlation_matrix(raw, GEOMETRY_METRICS + ["blocker_target_shared_protection_frames"])

    payload = {
        "exploration_version":"ADVANCED_DATA_SIGNAL_EXPLORATION_V1",
        "source":"BDB2023_PROTECTION_GEOMETRY",
        "predictive_model_fit":False,
        "production_changed":False,
        "sportsbook_inputs_used":False,
        "universal_blocker_rusher_assignment_claimed":False,
        "raw_interactions":int(len(raw)),
        "unique_blockers":int(raw["blocker_nfl_id"].nunique()),
        "e0_history_coverage_by_week":dataframe_records(coverage),
        "e1_blocker_temporal_persistence":dataframe_records(persistence),
        "e2_strict_prior_future_geometry":dataframe_records(future),
        "e3_geometry_vs_pff_outcomes_retrospective_only":dataframe_records(outcome_geometry),
        "block_type_geometry_summary":dataframe_records(block_type_summary),
        "spearman_correlation_matrix":corr.where(pd.notna(corr),None).to_dict(),
        "matched_future_blocker_rows":matched_future,
        "disposition":"BDB2023_ADVANCED_SIGNAL_RECONNAISSANCE_COMPLETE",
    }

    out_dir.mkdir(parents=True,exist_ok=True)
    persistence.to_csv(out_dir/"bdb2023_blocker_persistence_v1.csv",index=False)
    future.to_csv(out_dir/"bdb2023_future_geometry_validation_v1.csv",index=False)
    outcome_geometry.to_csv(out_dir/"bdb2023_geometry_vs_pff_outcomes_v1.csv",index=False)
    block_type_summary.to_csv(out_dir/"bdb2023_block_type_geometry_v1.csv",index=False)
    corr.to_csv(out_dir/"bdb2023_geometry_spearman_v1.csv")
    write_sanitized(out_dir,"bdb2023_advanced_signal_exploration_v1.json",payload)
    print(persistence.to_string(index=False))
    print(future.to_string(index=False))
    print(outcome_geometry.to_string(index=False))
    return payload


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--materialized-dir",type=Path,required=True)
    ap.add_argument("--corpus-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    run(a.materialized_dir,a.corpus_dir,a.out_dir)


if __name__=="__main__":
    main()
