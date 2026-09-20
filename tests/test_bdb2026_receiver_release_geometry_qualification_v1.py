import numpy as np
import pandas as pd

from scripts.football_context.qualify_bdb2026_receiver_release_geometry_v1 import (
    MIN_SOURCE_SUPPORT,
    _holdout_r2,
    _norm_id,
    build_broad_universe,
    build_direct_identity_bridge,
    map_history_snapshots,
    source_persistence,
)


def test_numeric_bdb_ids_normalize_without_name_fallback():
    assert _norm_id(12345.0) == "12345"
    assert _norm_id("12345.0") == "12345"
    assert _norm_id("12345") == "12345"


def _history():
    rows = []
    for week in range(1, 19):
        rows.append({
            "season": 2023,
            "week": week,
            "team": "A",
            "player_id": "00-001",
            "player_identity_key": "gsis:00-001",
            "player": "Name Does Not Matter",
            "position": "WR",
            "targets": 1,
            "team_targets": 20,
            "rushes": 0,
            "team_rushes": 20,
        })
        rows.append({
            "season": 2023,
            "week": week,
            "team": "B",
            "player_id": "00-002",
            "player_identity_key": "gsis:00-002",
            "player": "Other",
            "position": "TE",
            "targets": 1,
            "team_targets": 20,
            "rushes": 0,
            "team_rushes": 20,
        })
    return pd.DataFrame(rows)


def test_direct_crosswalk_uses_nfl_id_to_gsis_only():
    raw = pd.DataFrame({"nfl_id": [111.0, 222.0]})
    players = pd.DataFrame({
        "nfl_id": [111, 222],
        "gsis_id": ["00-001", "00-002"],
        "display_name": ["Totally Wrong Name", "Also Wrong"],
    })
    bridge, report = build_direct_identity_bridge(raw, players, _history())
    assert report["name_fallback_used"] is False
    assert report["direct_stable_id_bridge_coverage"] == 1.0
    assert dict(zip(bridge.bdb_nfl_id, bridge.player_identity_key)) == {
        "111": "gsis:00-001",
        "222": "gsis:00-002",
    }


def test_ambiguous_direct_nfl_id_is_not_silently_resolved():
    raw = pd.DataFrame({"nfl_id": [111.0]})
    players = pd.DataFrame({
        "nfl_id": [111, 111],
        "gsis_id": ["00-001", "00-002"],
    })
    bridge, report = build_direct_identity_bridge(raw, players, _history())
    assert report["ambiguous_nfl_id_count"] == 1
    assert report["direct_stable_id_bridge_coverage"] == 0.0
    assert bridge.empty


def test_strict_prior_and_support_threshold_are_recomputed():
    snapshots = pd.DataFrame([
        {
            "target_week": 4,
            "nfl_id": 111,
            "history_max_source_week": 3,
            "hist_receiver_release_geometry_sample_count": MIN_SOURCE_SUPPORT,
            "hist_receiver_release_nearest_defender_median_yards": 2.0,
            "hist_receiver_release_second_defender_median_yards": 5.0,
        },
        {
            "target_week": 5,
            "nfl_id": 111,
            "history_max_source_week": 5,
            "hist_receiver_release_geometry_sample_count": MIN_SOURCE_SUPPORT - 1,
            "hist_receiver_release_nearest_defender_median_yards": 2.1,
            "hist_receiver_release_second_defender_median_yards": 5.1,
        },
    ])
    bridge = pd.DataFrame({
        "bdb_nfl_id": ["111"],
        "player_identity_key": ["gsis:00-001"],
    })
    _, report = map_history_snapshots(snapshots, bridge)
    assert report["chronology_violations"] == 1
    assert report["materializer_support_threshold_violations"] == 1


def test_broad_universe_keeps_week1_and_unobserved_players_in_coverage_denominator():
    hist = _history()
    snapshots = pd.DataFrame([
        {
            "target_week": 10,
            "player_identity_key": "gsis:00-001",
            "hist_receiver_release_geometry_sample_count": 9,
            "history_max_source_week": 9,
            "hist_receiver_release_nearest_defender_median_yards": 2.0,
            "hist_receiver_release_second_defender_median_yards": 5.0,
        }
    ])
    joined, report = build_broad_universe(hist, snapshots)
    assert len(joined) == 36
    assert report["week1_rows_in_denominator"] == 2
    assert joined[
        "hist_receiver_release_nearest_defender_median_yards"
    ].notna().sum() == 1


def test_source_persistence_uses_frozen_8_per_half():
    rows = []
    for nfl_id, offset in [(1, 0.0), (2, 1.0), (3, 2.0)]:
        for week in range(1, 19):
            rows.append({
                "nfl_id": nfl_id,
                "week": week,
                "receiver_release_nearest_defender_distance_yards":
                    float(offset + (0.1 if week >= 10 else 0.0)),
                "receiver_release_second_defender_distance_yards":
                    float(2 * offset + (0.1 if week >= 10 else 0.0)),
            })
    out = source_persistence(pd.DataFrame(rows))
    assert set(out["minimum_observations_each_half"]) == {8}
    assert (out["stability_pairs"] == 3).all()
    assert np.allclose(out["stability_value"], 1.0)


def test_redundancy_r2_requires_frozen_row_floors():
    rows = []
    for week in range(1, 19):
        n = 20
        for i in range(n):
            rows.append({
                "week": week,
                "candidate": float(i),
                "prod_tgt_prior_share": float(i),
                "prod_tgt_prior_games": 1.0,
                "prod_tgt_current_share": 0.0,
                "prod_tgt_current_games": 1.0,
                "prod_tgt_playerform_blend": float(i),
            })
    r2, train_n, holdout_n = _holdout_r2(pd.DataFrame(rows), "candidate")
    assert train_n == 180
    assert holdout_n == 180
    assert np.isnan(r2)
