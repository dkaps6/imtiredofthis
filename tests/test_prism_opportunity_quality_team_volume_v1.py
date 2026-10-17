import numpy as np
import pandas as pd

from scripts.research.audit_prism_opportunity_quality_team_volume_v1 import (
    attach_strict_prior_state,
    aggregate_teamgames,
)


def test_team_high_value_pool_preserves_named_and_complement_mass():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 5, "game_id": "g", "team": "AAA", "opponent": "BBB"},
        {"season": 2025, "week": 5, "game_id": "g", "team": "BBB", "opponent": "AAA"},
    ])
    rush = pd.DataFrame([
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "r1", "inside20": True, "inside10": True, "inside5": False,
            "family": "RB",
        },
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "r2", "inside20": True, "inside10": True, "inside5": True,
            "family": "QB",
        },
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "BBB",
            "play_id": "r3", "inside20": False, "inside10": False, "inside5": False,
            "family": "RB",
        },
    ])
    target = pd.DataFrame([
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "t1", "red_zone": True, "end_zone_target": False,
            "family": "WR",
        },
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "t2", "red_zone": True, "end_zone_target": True,
            "family": "",
        },
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "BBB",
            "play_id": "t3", "red_zone": False, "end_zone_target": False,
            "family": "TE",
        },
    ])
    plays = pd.DataFrame([
        {"season": 2025, "week": 5, "game_id": "g", "team": "AAA", "opponent": "BBB", "yardline_100": 8, "drive_id": "1"},
        {"season": 2025, "week": 5, "game_id": "g", "team": "AAA", "opponent": "BBB", "yardline_100": 5, "drive_id": "1"},
        {"season": 2025, "week": 5, "game_id": "g", "team": "BBB", "opponent": "AAA", "yardline_100": 50, "drive_id": "2"},
    ])

    out = aggregate_teamgames(schedule, rush, target, plays, 2025)
    a = out.loc[out["team"].eq("AAA")].iloc[0]

    assert a["inside10_rush"] == 2
    assert a["rb_inside10_rush"] == 1
    assert a["inside10_rush_complement"] == 1
    assert a["red_zone_targets"] == 2
    assert a["modeled_rz_targets"] == 1
    assert a["rz_target_complement"] == 1
    assert a["rb_inside10_rush"] + a["inside10_rush_complement"] == a["inside10_rush"]
    assert a["modeled_rz_targets"] + a["rz_target_complement"] == a["red_zone_targets"]


def test_strict_prior_state_never_consumes_target_or_future_game():
    rows = []
    for week in range(1, 7):
        rows.append({
            "season": 2025,
            "week": week,
            "game_id": f"g{week}",
            "team": "AAA",
            "opponent": "BBB",
            "inside10_rush": 100 if week == 5 else float(week),
            "red_zone_targets": 200 if week == 5 else float(week + 10),
            "inside10_rush_allowed": float(week + 20),
            "red_zone_targets_allowed": float(week + 30),
        })
        rows.append({
            "season": 2025,
            "week": week,
            "game_id": f"g{week}",
            "team": "BBB",
            "opponent": "AAA",
            "inside10_rush": float(week + 40),
            "red_zone_targets": float(week + 50),
            "inside10_rush_allowed": 300 if week == 5 else float(week + 60),
            "red_zone_targets_allowed": 400 if week == 5 else float(week + 70),
        })

    out = attach_strict_prior_state(pd.DataFrame(rows))
    a5 = out.loc[out["team"].eq("AAA") & out["week"].eq(5)].iloc[0]

    assert a5["prior_off_games"] == 4
    assert a5["prior_opp_def_games"] == 4
    assert np.isclose(a5["prior_inside10_rush_mean"], np.mean([1, 2, 3, 4]))
    assert np.isclose(a5["prior_rz_target_mean"], np.mean([11, 12, 13, 14]))
    assert np.isclose(
        a5["prior_opp_inside10_rush_allowed_mean"], np.mean([61, 62, 63, 64])
    )
    assert np.isclose(
        a5["prior_opp_rz_targets_allowed_mean"], np.mean([71, 72, 73, 74])
    )
    # The target-week 100/200/300/400 values must not enter any prior state.
    assert a5["prior_inside10_rush_mean"] < 10
    assert a5["prior_opp_inside10_rush_allowed_mean"] < 100


def test_missing_rush_source_is_not_relabelled_zero():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 5, "game_id": "g", "team": "AAA", "opponent": "BBB"},
        {"season": 2025, "week": 5, "game_id": "g", "team": "BBB", "opponent": "AAA"},
    ])
    # Only AAA has a rush event source row. Both teams have target/play source.
    rush = pd.DataFrame([
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "r1", "inside20": False, "inside10": False,
            "inside5": False, "family": "RB",
        },
    ])
    target = pd.DataFrame([
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "AAA",
            "play_id": "t1", "red_zone": False, "end_zone_target": False,
            "family": "TE",
        },
        {
            "season": 2025, "week": 5, "game_id": "g", "team": "BBB",
            "play_id": "t2", "red_zone": False, "end_zone_target": False,
            "family": "TE",
        },
    ])
    plays = pd.DataFrame([
        {"season": 2025, "week": 5, "game_id": "g", "team": "AAA", "opponent": "BBB", "yardline_100": 50, "drive_id": "1"},
        {"season": 2025, "week": 5, "game_id": "g", "team": "BBB", "opponent": "AAA", "yardline_100": 50, "drive_id": "2"},
    ])

    out = aggregate_teamgames(schedule, rush, target, plays, 2025)
    a = out.loc[out["team"].eq("AAA")].iloc[0]
    b = out.loc[out["team"].eq("BBB")].iloc[0]

    assert bool(a["rush_source_present"])
    assert float(a["inside10_rush"]) == 0.0
    assert not bool(b["rush_source_present"])
    assert np.isnan(float(b["inside10_rush"]))
    assert bool(b["target_source_present"])
    assert float(b["red_zone_targets"]) == 0.0
