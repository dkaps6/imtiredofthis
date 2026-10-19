import pandas as pd

from scripts.research.evaluate_prism_return_state_next_role_persistence_v1 import (
    RETURN_RESTORED,
    START_COLLAPSE,
    START_NO_SNAPS,
    TARGET_CLEAN,
    build_scored_rows,
)


def _episodes(rows):
    return pd.DataFrame(rows)


def _health(rows):
    return pd.DataFrame(rows)


def test_restored_collapse_scores_pre_role_against_decline_anchor():
    episodes = _episodes([{
        "season": 2024,
        "team": "AAA",
        "stable_id": "p1",
        "family": "WR",
        "episode_number": 1,
        "episode_start_week": 4,
        "episode_start_type": START_COLLAPSE,
        "pre_episode_snap_pct": 0.80,
        "return_week": 6,
        "return_offense_pct": 0.76,
        "return_state": RETURN_RESTORED,
    }])
    health = _health([
        {"season":2024,"week":4,"team":"AAA","stable_id":"p1","family":"WR",
         "offense_pct":0.30,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2024,"week":6,"team":"AAA","stable_id":"p1","family":"WR",
         "offense_pct":0.76,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2024,"week":7,"team":"AAA","stable_id":"p1","family":"WR",
         "offense_pct":0.78,"positive_participation":True,
         "observation_state":TARGET_CLEAN},
    ])
    rows, audit = build_scored_rows(episodes, health)
    assert audit["scored_episodes"] == 1
    r = rows.iloc[0]
    assert r["decline_anchor"] == 0.30
    assert r["pre_role_anchor"] == 0.80
    assert r["actual_next_offense_pct"] == 0.78
    assert r["pre_role_closeness_gain"] > 0


def test_immediate_reported_target_fails_closed_and_does_not_skip_forward():
    episodes = _episodes([{
        "season": 2025,
        "team": "BBB",
        "stable_id": "p2",
        "family": "RB",
        "episode_number": 1,
        "episode_start_week": 3,
        "episode_start_type": START_COLLAPSE,
        "pre_episode_snap_pct": 0.65,
        "return_week": 5,
        "return_offense_pct": 0.61,
        "return_state": RETURN_RESTORED,
    }])
    health = _health([
        {"season":2025,"week":3,"team":"BBB","stable_id":"p2","family":"RB",
         "offense_pct":0.20,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2025,"week":5,"team":"BBB","stable_id":"p2","family":"RB",
         "offense_pct":0.61,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2025,"week":6,"team":"BBB","stable_id":"p2","family":"RB",
         "offense_pct":0.45,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2025,"week":7,"team":"BBB","stable_id":"p2","family":"RB",
         "offense_pct":0.66,"positive_participation":True,
         "observation_state":TARGET_CLEAN},
    ])
    rows, audit = build_scored_rows(episodes, health)
    assert rows.empty
    assert audit["exclusions"]["immediate_next_state__REPORTED_INJURY_PARTICIPATED"] == 1


def test_no_snap_interruption_uses_zero_only_as_participation_anchor():
    episodes = _episodes([{
        "season": 2025,
        "team": "CCC",
        "stable_id": "p3",
        "family": "TE",
        "episode_number": 1,
        "episode_start_week": 8,
        "episode_start_type": START_NO_SNAPS,
        "pre_episode_snap_pct": 0.72,
        "return_week": 10,
        "return_offense_pct": 0.70,
        "return_state": RETURN_RESTORED,
    }])
    health = _health([
        {"season":2025,"week":8,"team":"CCC","stable_id":"p3","family":"TE",
         "offense_pct":float("nan"),"positive_participation":False,
         "observation_state":"REPORTED_INJURY_NO_OFFENSIVE_SNAPS"},
        {"season":2025,"week":10,"team":"CCC","stable_id":"p3","family":"TE",
         "offense_pct":0.70,"positive_participation":True,
         "observation_state":"REPORTED_INJURY_PARTICIPATED"},
        {"season":2025,"week":11,"team":"CCC","stable_id":"p3","family":"TE",
         "offense_pct":0.74,"positive_participation":True,
         "observation_state":TARGET_CLEAN},
    ])
    rows, audit = build_scored_rows(episodes, health)
    assert audit["scored_episodes"] == 1
    r = rows.iloc[0]
    assert r["decline_anchor"] == 0.0
    assert r["pre_role_closeness_gain"] > 0
