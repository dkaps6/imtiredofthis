import numpy as np
import pandas as pd

from scripts.football_context.qualify_bdb2023_blocker_protection_geometry_v1 import (
    CANDIDATES,
    MIN_SOURCE_SUPPORT,
    _design_train_holdout,
    _height_inches,
    _norm_numeric_id,
    build_broad_ol_universe,
    build_direct_identity_bridge,
    map_blocker_history,
    source_stability,
)


def test_id_and_height_normalization_are_deterministic():
    assert _norm_numeric_id(12345.0) == "12345"
    assert _norm_numeric_id("12345.0") == "12345"
    assert _height_inches("6-5") == 77.0
    assert _height_inches("6' 4") == 76.0


def _roster():
    rows = []
    for week in range(1, 9):
        rows.extend([
            {
                "season": 2021, "week": week, "team": "A",
                "gsis_id": "00-001", "full_name": "Blocker One",
                "position": "T", "depth_chart_position": "LT",
                "height": "6-5", "weight": 315, "years_exp": 4,
            },
            {
                "season": 2021, "week": week, "team": "A",
                "gsis_id": "00-002", "full_name": "Backup Two",
                "position": "G", "depth_chart_position": "RG",
                "height": "6-3", "weight": 305, "years_exp": 2,
            },
            {
                "season": 2021, "week": week, "team": "A",
                "gsis_id": "00-003", "full_name": "Receiver Three",
                "position": "WR", "depth_chart_position": "WR",
                "height": "6-0", "weight": 195, "years_exp": 3,
            },
        ])
    return pd.DataFrame(rows)


def test_direct_bridge_uses_numeric_nfl_id_to_gsis_without_name_fallback():
    raw = pd.DataFrame({"blocker_nfl_id": [111, 222]})
    players = pd.DataFrame({
        "nfl_id": [111, 222],
        "gsis_id": ["00-001", "00-002"],
        "display_name": ["Wrong Name", "Also Wrong"],
    })
    bridge, report = build_direct_identity_bridge(raw, players, _roster())
    assert report["name_fallback_used"] is False
    assert report["direct_stable_id_bridge_coverage"] == 1.0
    assert dict(zip(bridge.bdb_nfl_id, bridge.gsis_id)) == {
        "111": "00-001", "222": "00-002"
    }


def test_ambiguous_bdb_id_fails_coverage_instead_of_name_rescue():
    raw = pd.DataFrame({"blocker_nfl_id": [111]})
    players = pd.DataFrame({
        "nfl_id": [111, 111],
        "gsis_id": ["00-001", "00-002"],
    })
    bridge, report = build_direct_identity_bridge(raw, players, _roster())
    assert report["ambiguous_bdb_nfl_id_count"] == 1
    assert report["direct_stable_id_bridge_coverage"] == 0.0
    assert bridge.empty

def test_name_variation_under_same_gsis_is_diagnostic_not_identity_ambiguity():
    raw = pd.DataFrame({"blocker_nfl_id": [111]})
    players = pd.DataFrame({"nfl_id": [111], "gsis_id": ["00-001"]})
    roster = _roster()
    roster.loc[
        roster["gsis_id"].eq("00-001") & roster["week"].eq(8), "full_name"
    ] = "Blocker One Jr."
    bridge, report = build_direct_identity_bridge(raw, players, roster)
    assert report["direct_stable_id_bridge_coverage"] == 1.0
    assert report["ambiguous_roster_gsis_count"] == 0
    assert report["roster_name_variation_gsis_count"] == 1
    assert len(bridge) == 1


def test_same_week_conflicting_team_for_gsis_is_real_roster_ambiguity():
    raw = pd.DataFrame({"blocker_nfl_id": [111]})
    players = pd.DataFrame({"nfl_id": [111], "gsis_id": ["00-001"]})
    roster = _roster()
    conflict = roster[
        roster["gsis_id"].eq("00-001") & roster["week"].eq(4)
    ].iloc[[0]].copy()
    conflict["team"] = "B"
    roster = pd.concat([roster, conflict], ignore_index=True)
    bridge, report = build_direct_identity_bridge(raw, players, roster)
    assert report["ambiguous_roster_gsis_count"] == 1
    assert report["direct_stable_id_bridge_coverage"] == 0.0
    assert bridge.empty



def test_broad_ol_universe_keeps_week1_and_backups_and_excludes_wr():
    universe, report = build_broad_ol_universe(_roster())
    assert len(universe) == 16
    assert report["week1_rows_in_denominator"] == 2
    assert set(universe["gsis_id"]) == {"00-001", "00-002"}
    assert report["target_game_snap_or_participation_used_for_eligibility"] is False


def test_strict_prior_and_support_threshold_are_recomputed():
    feature_names = list(CANDIDATES)
    snapshots = pd.DataFrame([
        {
            "target_week": 3, "blocker_nfl_id": 111,
            "history_max_source_week": 2,
            "snap_sample_count": 10,
            "min_distance_sample_count": 10,
            "time_to_min_sample_count": 10,
            feature_names[0]: 2.0,
            feature_names[1]: 0.7,
            feature_names[2]: 1.8,
        },
        {
            "target_week": 4, "blocker_nfl_id": 111,
            "history_max_source_week": 4,
            "snap_sample_count": 9,
            "min_distance_sample_count": 9,
            "time_to_min_sample_count": 9,
            feature_names[0]: 2.1,
            feature_names[1]: 0.8,
            feature_names[2]: 1.9,
        },
    ])
    bridge = pd.DataFrame({"bdb_nfl_id": ["111"], "gsis_id": ["00-001"]})
    _, report = map_blocker_history(snapshots, bridge)
    assert report["chronology_violations"] == 1
    assert report["materializer_support_threshold_violations"] == 3


def test_source_stability_uses_frozen_10_interactions_each_half():
    rows = []
    for blocker, offset in [(1, 0.0), (2, 1.0), (3, 2.0)]:
        for week in range(1, 9):
            for j in range(3):
                rows.append({
                    "blocker_nfl_id": blocker,
                    "week": week,
                    "blocker_target_snap_distance_yards": offset + j / 100,
                    "blocker_target_min_distance_yards": offset + 0.5 + j / 100,
                    "blocker_target_time_to_min_distance_seconds": offset + 1.0 + j / 100,
                })
    out = source_stability(pd.DataFrame(rows))
    assert set(out["minimum_observations_each_half"]) == {10}
    assert (out["stability_pairs"] == 3).all()
    assert np.allclose(out["stability_value"], 1.0)


def test_redundancy_row_floors_are_locked():
    feature = list(CANDIDATES)[0]
    rows = []
    # 499 qualifying early rows, 200 late rows => fail-closed unresolved.
    for i in range(699):
        early = i < 499
        rows.append({
            "week": 1 + (i % 4) if early else 5 + (i % 4),
            feature: float(i % 11),
            "position_norm": "T",
            "depth_position_norm": "LT",
            "height_inches": 77,
            "weight_num": 315,
            "years_exp_num": 4,
        })
    Xtr, ytr, Xte, yte, ntr, nte = _design_train_holdout(
        pd.DataFrame(rows), feature
    )
    assert ntr == 499
    assert nte == 200
    assert Xtr.size == 0
    assert Xte.size == 0


def test_redundancy_encoding_uses_train_schema_and_train_numeric_fill():
    feature = list(CANDIDATES)[0]
    rows = []
    for i in range(700):
        early = i < 500
        rows.append({
            "week": 1 + (i % 4) if early else 5 + (i % 4),
            feature: float(i % 13),
            "position_norm": "T" if early else "NEW_HOLDOUT_POSITION",
            "depth_position_norm": "LT" if early else "NEW_DEPTH",
            "height_inches": 77 if i % 5 else np.nan,
            "weight_num": 315,
            "years_exp_num": 4,
        })
    Xtr, ytr, Xte, yte, ntr, nte = _design_train_holdout(
        pd.DataFrame(rows), feature
    )
    assert ntr == 500 and nte == 200
    assert Xtr.shape[1] == Xte.shape[1]
    assert np.isfinite(Xtr).all()
    assert np.isfinite(Xte).all()
