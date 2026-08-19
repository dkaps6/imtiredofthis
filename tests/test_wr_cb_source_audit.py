import pandas as pd

from scripts.backtest.audit_wr_cb_source import audit_frames


def test_audit_rejects_on_field_only_as_true_assignment():
    participation = pd.DataFrame([
        {"old_game_id": "g1", "play_id": 1, "offense_names": "Receiver One", "defense_names": "Corner One; Safety One"},
    ])
    pbp = pd.DataFrame([
        {"season": 2025, "week": 1, "old_game_id": "g1", "play_id": 1, "receiver_player_name": "Receiver One"},
    ])
    summary, inventory, by_week = audit_frames(participation, pbp)
    rec = summary.loc[summary["metric"].eq("wr_cb_source_recommendation")].iloc[0]
    assert rec["value"] == "NO_GO_TRUE_ASSIGNMENT"
    assert "no_explicit_coverage_responsibility" in rec["status"]
    assert not inventory.empty
    assert by_week.iloc[0]["target_receiver_coverage"] == 1.0
    assert by_week.iloc[0]["explicit_assignment_coverage"] == 0.0


def test_audit_accepts_explicit_receiver_defender_assignment():
    participation = pd.DataFrame([
        {"old_game_id": "g1", "play_id": 1, "defense_names": "Corner One"},
    ])
    pbp = pd.DataFrame([
        {
            "season": 2025,
            "week": 1,
            "old_game_id": "g1",
            "play_id": 1,
            "receiver_player_name": "Receiver One",
            "coverage_defender_name": "Corner One",
        },
    ])
    summary, _, by_week = audit_frames(participation, pbp)
    rec = summary.loc[summary["metric"].eq("wr_cb_source_recommendation")].iloc[0]
    assert rec["value"] == "GO_TRUE_ASSIGNMENT"
    assert by_week.iloc[0]["explicit_assignment_coverage"] == 1.0


def test_join_rate_is_reported():
    participation = pd.DataFrame([
        {"old_game_id": "g1", "play_id": 1},
        {"old_game_id": "g1", "play_id": 2},
    ])
    pbp = pd.DataFrame([
        {"old_game_id": "g1", "play_id": 1, "season": 2025, "week": 1},
    ])
    summary, _, _ = audit_frames(participation, pbp)
    rate = summary.loc[summary["metric"].eq("participation_pbp_join_rate"), "value"].iloc[0]
    assert float(rate) == 0.5
