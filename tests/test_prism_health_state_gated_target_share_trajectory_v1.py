import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_health_state_gated_target_share_trajectory_v1 import (
    apply_candidate,
    metrics,
)


def _rows():
    return pd.DataFrame([
        {
            "season": 2024, "week": 5, "event_id": "AAA|BBB", "team": "AAA",
            "player_clean_key": "anchor", "player": "Anchor", "is_wr1": True,
            "position_group": "WR", "trajectory_delta": 0.20, "feature_max_week": 4,
            "baseline_targets": 5.0, "baseline_ypt": 8.0,
            "shadow_targets": 5.0, "actual_targets": 5.0, "actual_rec_yards": 40.0,
            "shadow_rec_yards": 40.0, "shadow_room_share": 0.5, "actual_room_share": 0.5,
            "recent2_observation_contaminated": False,
            "recent2_participation_certified": True,
        },
        {
            "season": 2024, "week": 5, "event_id": "AAA|BBB", "team": "AAA",
            "player_clean_key": "secondary_a", "player": "Secondary A", "is_wr1": False,
            "position_group": "WR", "trajectory_delta": -0.30, "feature_max_week": 4,
            "baseline_targets": 3.0, "baseline_ypt": 7.0,
            "shadow_targets": 2.4, "actual_targets": 3.0, "actual_rec_yards": 21.0,
            "shadow_rec_yards": 16.8, "shadow_room_share": 0.24, "actual_room_share": 0.3,
            "recent2_observation_contaminated": True,
            "recent2_participation_certified": False,
        },
        {
            "season": 2024, "week": 5, "event_id": "AAA|BBB", "team": "AAA",
            "player_clean_key": "secondary_b", "player": "Secondary B", "is_wr1": False,
            "position_group": "WR", "trajectory_delta": 0.30, "feature_max_week": 4,
            "baseline_targets": 2.0, "baseline_ypt": 6.0,
            "shadow_targets": 2.6, "actual_targets": 2.0, "actual_rec_yards": 12.0,
            "shadow_rec_yards": 15.6, "shadow_room_share": 0.26, "actual_room_share": 0.2,
            "recent2_observation_contaminated": False,
            "recent2_participation_certified": True,
        },
    ])


def test_candidate_neutralizes_only_contaminated_delta_and_preserves_wr_anchor_pool():
    rows = _rows()
    out, audit = apply_candidate(rows)

    anchor = out.loc[out["player_clean_key"].eq("anchor")].iloc[0]
    a = out.loc[out["player_clean_key"].eq("secondary_a")].iloc[0]
    b = out.loc[out["player_clean_key"].eq("secondary_b")].iloc[0]

    assert anchor["candidate_targets"] == 5.0
    assert a["health_state_trajectory_delta"] == 0.0
    assert b["health_state_trajectory_delta"] == 0.30

    secondaries = out.loc[~out["is_wr1"]]
    assert np.isclose(secondaries["candidate_targets"].sum(), 5.0, atol=1e-12)
    assert float(audit["wr1_max_abs_gap"].max()) <= 1e-12
    assert float(audit["protected_pool_max_abs_gap"].max()) <= 1e-12
    assert float(audit["room_pool_max_abs_gap"].max()) <= 1e-12


def test_all_clean_rows_reproduce_original_trajectory_allocation_when_shadow_is_exact():
    rows = _rows()
    rows["recent2_observation_contaminated"] = False
    rows["recent2_participation_certified"] = True

    # Rebuild an exact frozen shadow from the same baseline/deltas first.
    expected, _ = apply_candidate(rows)
    rows["shadow_targets"] = expected["candidate_targets"]
    rows["shadow_rec_yards"] = rows["shadow_targets"] * rows["baseline_ypt"]
    pool = rows.groupby(["season","week","event_id","team","position_group"])["baseline_targets"].transform("sum")
    rows["shadow_room_share"] = rows["shadow_targets"] / pool

    out, _ = apply_candidate(rows)
    assert np.allclose(out["candidate_targets"], rows["shadow_targets"], atol=1e-12)
    assert np.allclose(out["candidate_rec_yards"], rows["shadow_rec_yards"], atol=1e-12)
    assert np.allclose(out["candidate_room_share"], rows["shadow_room_share"], atol=1e-12)


def test_metrics_compare_candidate_to_existing_trajectory_not_raw_parent():
    rows = _rows()
    out, _ = apply_candidate(rows)
    m = metrics(out)
    assert "original_target_mae" in m
    assert "candidate_target_mae" in m
    assert m["rows"] == 3


def test_candidate_room_share_uses_combined_wr_te_parent_denominator():
    rows = _rows()
    te = pd.DataFrame([{
        "season": 2024, "week": 5, "event_id": "AAA|BBB", "team": "AAA",
        "player_clean_key": "tightend", "player": "Tight End", "is_wr1": False,
        "position_group": "TE", "trajectory_delta": 0.0, "feature_max_week": 4,
        "baseline_targets": 4.0, "baseline_ypt": 5.0,
        "shadow_targets": 4.0, "actual_targets": 4.0, "actual_rec_yards": 20.0,
        "shadow_rec_yards": 20.0, "shadow_room_share": 4.0/14.0,
        "actual_room_share": 4.0/14.0,
        "recent2_observation_contaminated": False,
        "recent2_participation_certified": True,
    }])
    rows = pd.concat([rows, te], ignore_index=True)
    out, _ = apply_candidate(rows)
    assert np.allclose(
        out["candidate_room_share"].to_numpy(float),
        out["candidate_targets"].to_numpy(float) / 14.0,
        atol=1e-12,
    )
