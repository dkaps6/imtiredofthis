"""Advanced-data signal reconnaissance for BDB 2026 throw-window features."""
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

RELEASE_METRICS = [
    "receiver_release_nearest_defender_distance_yards",
    "receiver_release_second_defender_distance_yards",
    "receiver_release_defenders_within_2yd_count",
    "receiver_release_defenders_within_3yd_count",
]


def run(materialized_dir: Path, out_dir: Path) -> dict:
    private = materialized_dir / "private"
    raw = pd.read_csv(private / "bdb2026_targeted_receiver_throw_window_v1.csv")
    hist = pd.read_csv(private / "bdb2026_receiver_history_snapshots_v1.csv")
    route_hist = pd.read_csv(private / "bdb2026_receiver_route_history_snapshots_v1.csv")

    persistence_player, _ = split_persistence(
        raw,
        keys=["nfl_id"],
        metrics=RELEASE_METRICS,
        early_mask=raw["week"].between(1, 9),
        late_mask=raw["week"].between(10, 18),
        min_obs_each=8,
    )
    persistence_route, _ = split_persistence(
        raw.loc[raw["source_target_route_label"].notna()].copy(),
        keys=["nfl_id", "source_target_route_label"],
        metrics=[
            "receiver_release_nearest_defender_distance_yards",
            "receiver_release_second_defender_distance_yards",
        ],
        early_mask=raw.loc[raw["source_target_route_label"].notna(), "week"].between(1, 9),
        late_mask=raw.loc[raw["source_target_route_label"].notna(), "week"].between(10, 18),
        min_obs_each=5,
    )

    future_player, matched_player = future_geometry_validation(
        raw=raw,
        history=hist,
        target_keys=["nfl_id"],
        target_week_col="week",
        history_target_week_col="target_week",
        history_pred_actual_pairs=[
            ("hist_receiver_release_nearest_defender_median_yards", "receiver_release_nearest_defender_distance_yards"),
            ("hist_receiver_release_second_defender_median_yards", "receiver_release_second_defender_distance_yards"),
            ("hist_receiver_release_crowding_2yd_rate", "receiver_release_defenders_within_2yd_count"),
            ("hist_receiver_release_crowding_3yd_rate", "receiver_release_defenders_within_3yd_count"),
        ],
        history_count_col="hist_receiver_release_geometry_sample_count",
        tier_breaks=(8, 16, 40),
    )

    # Convert target-week crowding counts to incidence rates so they match historical rate semantics.
    # Rebuild those two evaluations at player-week grain.
    crowd = raw.copy()
    crowd["crowd2_inc"] = (pd.to_numeric(crowd["receiver_release_defenders_within_2yd_count"], errors="coerce") >= 1).astype(float)
    crowd["crowd3_inc"] = (pd.to_numeric(crowd["receiver_release_defenders_within_3yd_count"], errors="coerce") >= 1).astype(float)
    actual_crowd = (
        crowd.groupby(["week", "nfl_id"], dropna=False)
        .agg(actual_crowd2_rate=("crowd2_inc", "mean"), actual_crowd3_rate=("crowd3_inc", "mean"))
        .reset_index()
    )
    crowd_join = hist.merge(actual_crowd, left_on=["target_week", "nfl_id"], right_on=["week", "nfl_id"], how="inner")
    crowd_join = crowd_join.loc[crowd_join["hist_receiver_release_geometry_sample_count"] >= 8].copy()
    crowd_metrics = pd.DataFrame([
        {"predictor":"hist_receiver_release_crowding_2yd_rate","actual_geometry":"actual_crowd2_rate","confidence_tier":"ALL",**pair_metrics(crowd_join,"hist_receiver_release_crowding_2yd_rate","actual_crowd2_rate")},
        {"predictor":"hist_receiver_release_crowding_3yd_rate","actual_geometry":"actual_crowd3_rate","confidence_tier":"ALL",**pair_metrics(crowd_join,"hist_receiver_release_crowding_3yd_rate","actual_crowd3_rate")},
    ])
    future_player = future_player.loc[~future_player["predictor"].str.contains("crowding")].copy()
    future_player = pd.concat([future_player, crowd_metrics], ignore_index=True)

    # Route-conditioned future geometry is retrospective validation because target-game route is not known pregame.
    actual_route = (
        raw.loc[raw["source_target_route_label"].notna()]
        .groupby(["week", "nfl_id", "source_target_route_label"], dropna=False)
        .agg(actual_route_release_nearest=("receiver_release_nearest_defender_distance_yards", "median"))
        .reset_index()
    )
    route_join = route_hist.merge(
        actual_route,
        left_on=["target_week", "nfl_id", "source_target_route_label"],
        right_on=["week", "nfl_id", "source_target_route_label"],
        how="inner",
    )
    route_join = route_join.loc[route_join["route_history_sample_count"] >= 5].copy()
    route_future = pd.DataFrame([
        {
            "predictor":"hist_receiver_route_release_nearest_defender_median_yards",
            "actual_geometry":"actual_route_release_nearest",
            "confidence_tier":"ALL",
            **pair_metrics(route_join,"hist_receiver_route_release_nearest_defender_median_yards","actual_route_release_nearest"),
        }
    ])

    retrospective = pd.DataFrame([
        {
            "relationship":"release_nearest_vs_terminal_nearest_predicted",
            **pair_metrics(raw,"receiver_release_nearest_defender_distance_yards","terminal_nearest_predicted_defender_to_target_yards"),
        },
        {
            "relationship":"release_nearest_vs_postrelease_min_nearest_predicted",
            **pair_metrics(raw,"receiver_release_nearest_defender_distance_yards","postrelease_min_nearest_predicted_defender_distance_yards"),
        },
        {
            "relationship":"release_target_to_land_vs_terminal_target_to_land",
            **pair_metrics(raw,"receiver_release_target_to_land_distance_yards","terminal_receiver_to_land_distance_yards"),
        },
    ])

    route_coverage = (
        raw.groupby(["source_target_route_label", "source_team_coverage_man_zone"], dropna=False)
        .agg(
            plays=("play_id", "size"),
            median_release_nearest=("receiver_release_nearest_defender_distance_yards", "median"),
            median_release_second=("receiver_release_second_defender_distance_yards", "median"),
            mean_crowd2=("receiver_release_defenders_within_2yd_count", "mean"),
            mean_crowd3=("receiver_release_defenders_within_3yd_count", "mean"),
        )
        .reset_index()
    )

    corr = numeric_correlation_matrix(
        raw,
        [
            "receiver_release_nearest_defender_distance_yards",
            "receiver_release_second_defender_distance_yards",
            "receiver_release_defenders_within_2yd_count",
            "receiver_release_defenders_within_3yd_count",
            "receiver_release_target_to_land_distance_yards",
            "terminal_receiver_to_land_distance_yards",
            "postrelease_closing_delta_yards",
        ],
    )

    by_week_coverage = (
        hist.assign(qualified=hist["hist_receiver_release_geometry_sample_count"] >= 8)
        .groupby("target_week")
        .agg(
            receiver_history_rows=("nfl_id", "size"),
            qualified_rows=("qualified", "sum"),
            median_prior_samples=("hist_receiver_release_geometry_sample_count", "median"),
        )
        .reset_index()
    )
    by_week_coverage["qualified_rate"] = by_week_coverage["qualified_rows"] / by_week_coverage["receiver_history_rows"]

    payload = {
        "exploration_version": "ADVANCED_DATA_SIGNAL_EXPLORATION_V1",
        "source": "BDB2026_THROW_WINDOW",
        "predictive_model_fit": False,
        "production_changed": False,
        "sportsbook_inputs_used": False,
        "exact_coverage_responsibility_claimed": False,
        "raw_rows": int(len(raw)),
        "unique_targeted_receivers": int(raw["nfl_id"].nunique()),
        "e0_history_coverage_by_week": dataframe_records(by_week_coverage),
        "e1_player_temporal_persistence": dataframe_records(persistence_player),
        "e1_receiver_route_temporal_persistence": dataframe_records(persistence_route),
        "e2_strict_prior_future_geometry": dataframe_records(future_player),
        "e2_route_conditioned_future_geometry_retrospective_only": dataframe_records(route_future),
        "e3_release_to_postrelease_validation": dataframe_records(retrospective),
        "route_x_man_zone_geometry": dataframe_records(route_coverage),
        "spearman_correlation_matrix": corr.where(pd.notna(corr), None).to_dict(),
        "matched_future_player_rows": int(len(matched_player)),
        "matched_route_history_rows": int(len(route_join)),
        "disposition": "BDB2026_ADVANCED_SIGNAL_RECONNAISSANCE_COMPLETE",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    persistence_player.to_csv(out_dir / "bdb2026_player_persistence_v1.csv", index=False)
    persistence_route.to_csv(out_dir / "bdb2026_route_persistence_v1.csv", index=False)
    future_player.to_csv(out_dir / "bdb2026_future_geometry_validation_v1.csv", index=False)
    route_future.to_csv(out_dir / "bdb2026_route_future_geometry_validation_v1.csv", index=False)
    retrospective.to_csv(out_dir / "bdb2026_retrospective_geometry_validation_v1.csv", index=False)
    route_coverage.to_csv(out_dir / "bdb2026_route_x_man_zone_geometry_v1.csv", index=False)
    corr.to_csv(out_dir / "bdb2026_geometry_spearman_v1.csv")
    write_sanitized(out_dir, "bdb2026_advanced_signal_exploration_v1.json", payload)
    print(pd.DataFrame(payload["e1_player_temporal_persistence"]).to_string(index=False))
    print(pd.DataFrame(payload["e2_strict_prior_future_geometry"]).to_string(index=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialized-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    run(a.materialized_dir, a.out_dir)


if __name__ == "__main__":
    main()
