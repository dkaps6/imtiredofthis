import pandas as pd

from scripts.research.evaluate_prism_restored_return_injury_report_state_persistence_v1 import (
    NEXT_CLEAN,
    RETURN_RESTORED,
    STATE_NO_REPORT,
    STATE_REPORT,
    bootstrap_difference,
    build_rows,
)


def _episode(pid, *, return_reported, season=2024, team="AAA", start=4, ret=6):
    return {
        "season":season,
        "team":team,
        "stable_id":pid,
        "family":"WR",
        "episode_number":1,
        "episode_start_week":start,
        "return_week":ret,
        "return_state":RETURN_RESTORED,
        "return_injury_reported":return_reported,
    }


def _health(pid, week, *, injury, state, season=2024, team="AAA", positive=True):
    return {
        "season":season,
        "week":week,
        "team":team,
        "stable_id":pid,
        "family":"WR",
        "positive_participation":positive,
        "injury_reported":injury,
        "observation_state":state,
    }


def test_return_report_state_and_immediate_next_are_exact():
    episodes=pd.DataFrame([
        _episode("p1",return_reported=False),
        _episode("p2",return_reported=True),
    ])
    health=pd.DataFrame([
        _health("p1",6,injury=False,state=NEXT_CLEAN),
        _health("p1",7,injury=False,state=NEXT_CLEAN),
        _health("p2",6,injury=True,state="REPORTED_INJURY_PARTICIPATED"),
        _health("p2",7,injury=True,state="REPORTED_INJURY_PARTICIPATED"),
        _health("p2",8,injury=False,state=NEXT_CLEAN),
    ])
    rows,audit=build_rows(episodes,health)
    assert audit["return_report_mismatches"] == 0
    assert audit["skipped_immediate_next_violations"] == 0
    assert len(rows) == 2

    p1=rows.loc[rows["stable_id"].eq("p1")].iloc[0]
    assert p1["current_return_report_state"] == STATE_NO_REPORT
    assert int(p1["next_week"]) == 7
    assert int(p1["next_clean_no_report_participation"]) == 1

    p2=rows.loc[rows["stable_id"].eq("p2")].iloc[0]
    assert p2["current_return_report_state"] == STATE_REPORT
    assert int(p2["next_week"]) == 7
    assert int(p2["next_clean_no_report_participation"]) == 0


def test_return_report_mismatch_fails_closed():
    episodes=pd.DataFrame([_episode("p1",return_reported=True)])
    health=pd.DataFrame([
        _health("p1",6,injury=False,state=NEXT_CLEAN),
        _health("p1",7,injury=False,state=NEXT_CLEAN),
    ])
    rows,audit=build_rows(episodes,health)
    assert rows.empty
    assert audit["return_report_mismatches"] == 1


def test_no_next_observation_is_not_fabricated():
    episodes=pd.DataFrame([_episode("p1",return_reported=False)])
    health=pd.DataFrame([
        _health("p1",6,injury=False,state=NEXT_CLEAN),
    ])
    rows,audit=build_rows(episodes,health)
    assert rows.empty
    assert audit["no_next_observation"] == 1


def test_player_cluster_bootstrap_preserves_state_means():
    rows=pd.DataFrame([
        {"stable_id":"p1","current_return_report_state":STATE_NO_REPORT,
         "next_clean_no_report_participation":1},
        {"stable_id":"p1","current_return_report_state":STATE_REPORT,
         "next_clean_no_report_participation":0},
        {"stable_id":"p2","current_return_report_state":STATE_NO_REPORT,
         "next_clean_no_report_participation":1},
        {"stable_id":"p3","current_return_report_state":STATE_REPORT,
         "next_clean_no_report_participation":0},
    ])
    b=bootstrap_difference(rows)
    assert b["valid_reps"] > 0
    assert b["ci_low"] > 0
