"""Advanced-data signal reconnaissance for BDB 2021 route geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.data_frontier.advanced_data_signal_exploration_common_v1 import (
    dataframe_records,
    future_geometry_validation,
    numeric_correlation_matrix,
    pair_metrics,
    split_persistence,
    write_sanitized,
)

THROW_METRICS = [
    "throw_nearest_defender_distance_yards",
    "throw_second_nearest_defender_distance_yards",
    "throw_spacing_gap_yards",
]


def run(materialized_dir: Path, out_dir: Path) -> dict:
    private = materialized_dir / "private"
    raw = pd.read_csv(private / "bdb2021_route_playerplays_v1.csv")
    hist = pd.read_csv(private / "bdb2021_route_history_snapshots_v1.csv")
    raw["throw_spacing_gap_yards"] = (
        pd.to_numeric(raw["throw_second_nearest_defender_distance_yards"], errors="coerce")
        - pd.to_numeric(raw["throw_nearest_defender_distance_yards"], errors="coerce")
    )

    persistence, _ = split_persistence(
        raw,
        keys=["nfl_id", "route_label"],
        metrics=THROW_METRICS,
        early_mask=raw["week"].between(1, 8),
        late_mask=raw["week"].between(9, 17),
        min_obs_each=5,
    )

    future, matched = future_geometry_validation(
        raw=raw,
        history=hist,
        target_keys=["nfl_id", "route_label"],
        target_week_col="week",
        history_target_week_col="target_week",
        history_pred_actual_pairs=[
            ("hist_player_route_throw_nearest_defender_median_yards", "throw_nearest_defender_distance_yards"),
            ("hist_player_route_throw_second_defender_median_yards", "throw_second_nearest_defender_distance_yards"),
            ("hist_player_route_throw_spacing_gap_median_yards", "throw_spacing_gap_yards"),
        ],
        history_count_col="hist_player_route_throw_geometry_sample_count",
        tier_breaks=(5, 10, 25),
    )

    retrospective = pd.DataFrame([
        {
            "relationship":"throw_nearest_vs_arrival_nearest",
            **pair_metrics(raw,"throw_nearest_defender_distance_yards","arrival_nearest_defender_distance_yards"),
        },
        {
            "relationship":"throw_second_vs_arrival_second",
            **pair_metrics(raw,"throw_second_nearest_defender_distance_yards","arrival_second_nearest_defender_distance_yards"),
        },
        {
            "relationship":"snap_nearest_vs_throw_nearest",
            **pair_metrics(raw,"snap_nearest_defender_distance_yards","throw_nearest_defender_distance_yards"),
        },
    ])

    route_summary = (
        raw.groupby("route_label", dropna=False)
        .agg(
            player_plays=("play_id", "size"),
            unique_players=("nfl_id", "nunique"),
            median_snap_nearest=("snap_nearest_defender_distance_yards", "median"),
            median_throw_nearest=("throw_nearest_defender_distance_yards", "median"),
            median_throw_second=("throw_second_nearest_defender_distance_yards", "median"),
            median_throw_spacing_gap=("throw_spacing_gap_yards", "median"),
            median_arrival_nearest=("arrival_nearest_defender_distance_yards", "median"),
        )
        .reset_index()
    )

    coverage = (
        hist.assign(qualified=hist["hist_player_route_throw_geometry_sample_count"] >= 5)
        .groupby("target_week")
        .agg(
            history_rows=("nfl_id", "size"),
            qualified_rows=("qualified", "sum"),
            median_prior_samples=("hist_player_route_throw_geometry_sample_count", "median"),
        )
        .reset_index()
    )
    coverage["qualified_rate"] = coverage["qualified_rows"] / coverage["history_rows"]

    corr = numeric_correlation_matrix(
        raw,
        [
            "snap_nearest_defender_distance_yards",
            "throw_nearest_defender_distance_yards",
            "throw_second_nearest_defender_distance_yards",
            "throw_spacing_gap_yards",
            "arrival_nearest_defender_distance_yards",
            "arrival_second_nearest_defender_distance_yards",
            "snap_to_throw_nearest_defender_delta_yards",
        ],
    )

    payload = {
        "exploration_version":"ADVANCED_DATA_SIGNAL_EXPLORATION_V1",
        "source":"BDB2021_ROUTE_GEOMETRY",
        "predictive_model_fit":False,
        "production_changed":False,
        "sportsbook_inputs_used":False,
        "exact_coverage_responsibility_claimed":False,
        "semantic_target_identity_available":False,
        "raw_route_playerplays":int(len(raw)),
        "unique_route_runners":int(raw["nfl_id"].nunique()),
        "e0_history_coverage_by_week":dataframe_records(coverage),
        "e1_player_route_temporal_persistence":dataframe_records(persistence),
        "e2_strict_prior_future_geometry":dataframe_records(future),
        "e3_event_geometry_validation":dataframe_records(retrospective),
        "route_geometry_summary":dataframe_records(route_summary),
        "spearman_correlation_matrix":corr.where(pd.notna(corr), None).to_dict(),
        "matched_future_geometry_rows":int(len(matched)),
        "disposition":"BDB2021_ADVANCED_SIGNAL_RECONNAISSANCE_COMPLETE",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    persistence.to_csv(out_dir/"bdb2021_player_route_persistence_v1.csv",index=False)
    future.to_csv(out_dir/"bdb2021_future_geometry_validation_v1.csv",index=False)
    retrospective.to_csv(out_dir/"bdb2021_event_geometry_validation_v1.csv",index=False)
    route_summary.to_csv(out_dir/"bdb2021_route_geometry_summary_v1.csv",index=False)
    corr.to_csv(out_dir/"bdb2021_geometry_spearman_v1.csv")
    write_sanitized(out_dir,"bdb2021_advanced_signal_exploration_v1.json",payload)
    print(persistence.to_string(index=False))
    print(future.to_string(index=False))
    return payload


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--materialized-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    run(a.materialized_dir,a.out_dir)


if __name__=="__main__":
    main()
