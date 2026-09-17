import pandas as pd

from scripts.data_frontier.bdb_2024_artifact_qa import event_window_diagnostics, persist_normalized


def _tracking():
    return pd.DataFrame([
        {"gameId": 1, "playId": 10, "frameId": 1, "nflId": 11, "club": "A", "event": "ball_snap"},
        {"gameId": 1, "playId": 10, "frameId": 1, "nflId": 21, "club": "B", "event": "ball_snap"},
        {"gameId": 1, "playId": 10, "frameId": 2, "nflId": 11, "club": "A", "event": "handoff"},
        {"gameId": 1, "playId": 10, "frameId": 2, "nflId": 21, "club": "B", "event": "handoff"},
        {"gameId": 1, "playId": 11, "frameId": 8, "nflId": 12, "club": "A", "event": "tackle"},
    ])


def test_event_window_diagnostics_deduplicates_same_frame_event_across_players():
    qa = event_window_diagnostics(_tracking())
    assert qa["tracking_plays"] == 2
    assert qa["event_counts"]["ball_snap"] == 1
    assert qa["event_counts"]["handoff"] == 1
    assert qa["event_counts"]["tackle"] == 1
    assert qa["play_event_presence"]["handoff"]["plays"] == 1
    assert qa["play_event_presence"]["handoff"]["rate"] == 0.5
    assert qa["frames_per_play"] == {"min": 1, "median": 1.5, "max": 2}


def test_event_window_diagnostics_handles_missing_event_column():
    qa = event_window_diagnostics(_tracking().drop(columns=["event"]))
    assert qa["event_column_present"] is False
    assert qa["tracking_plays"] == 2


def test_persist_normalized_writes_all_three_tables(tmp_path):
    tracking = _tracking()
    plays = pd.DataFrame([{"gameId": 1, "playId": 10, "ballCarrierId": 11}])
    tackles = pd.DataFrame([{"gameId": 1, "playId": 10, "nflId": 21, "tackle": 1}])
    manifest = persist_normalized(tmp_path, tracking, plays, tackles)
    assert set(manifest) == {"tracking", "plays", "tackles"}
    assert manifest["tracking"]["rows"] == 5
    for name in manifest:
        assert (tmp_path / "normalized" / f"{name}.csv").exists()
