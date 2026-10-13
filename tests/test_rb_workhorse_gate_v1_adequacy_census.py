import pandas as pd

from scripts.backtest.rb_workhorse_gate_v1_adequacy_census import (
    WORKHORSE_CARRY_FLOOR,
    compute_workhorse_events,
    season_census,
)


def _roster_row(season, week, team, player_clean_key, name_key="p"):
    return {
        "season": season, "week": week, "team": team, "player_key": player_clean_key,
        "position": "RB", "status": "ACT", "name_key": name_key, "player_clean_key": player_clean_key,
    }


def _log_row(season, week, team, player_clean_key, rushes):
    return {"season": season, "week": week, "team": team, "player_clean_key": player_clean_key, "rushes": rushes}


def test_compute_workhorse_events_flags_event_with_qualifying_active_player():
    scored_events = pd.DataFrame([{"season": 2019, "week": 3, "team": "TB", "prior_season": 2019, "prior_week": 2}])
    roster_state = pd.DataFrame(
        [_roster_row(2019, 3, "TB", "p1"), _roster_row(2019, 3, "TB", "p2")]
    )
    injury_state = pd.DataFrame(columns=["season", "week", "team", "player", "status"])
    player_logs = pd.DataFrame(
        [_log_row(2019, 3, "TB", "p1", 22.0), _log_row(2019, 3, "TB", "p2", 4.0)]
    )
    out = compute_workhorse_events(scored_events, roster_state, injury_state, player_logs)
    assert len(out) == 1
    assert bool(out.iloc[0]["workhorse_event"])
    assert out.iloc[0]["max_active_realized_carries"] == 22.0


def test_compute_workhorse_events_not_flagged_below_floor():
    scored_events = pd.DataFrame([{"season": 2019, "week": 3, "team": "TB", "prior_season": 2019, "prior_week": 2}])
    roster_state = pd.DataFrame([_roster_row(2019, 3, "TB", "p1")])
    injury_state = pd.DataFrame(columns=["season", "week", "team", "player", "status"])
    player_logs = pd.DataFrame([_log_row(2019, 3, "TB", "p1", WORKHORSE_CARRY_FLOOR - 1)])
    out = compute_workhorse_events(scored_events, roster_state, injury_state, player_logs)
    assert not bool(out.iloc[0]["workhorse_event"])


def test_compute_workhorse_events_ignores_unavailable_active_room_player():
    """A player with a >=20-carry realized log who is flagged unavailable
    (e.g. left mid-game) must not count -- the target is defined over the
    pregame-resolvable active room, not whoever happened to touch the ball.
    """
    scored_events = pd.DataFrame([{"season": 2019, "week": 3, "team": "TB", "prior_season": 2019, "prior_week": 2}])
    roster_state = pd.DataFrame([_roster_row(2019, 3, "TB", "p1")])
    injury_state = pd.DataFrame([{"season": 2019, "week": 3, "team": "TB", "player": "p", "status": "Out"}])
    player_logs = pd.DataFrame([_log_row(2019, 3, "TB", "p1", 25.0)])
    out = compute_workhorse_events(scored_events, roster_state, injury_state, player_logs)
    assert not bool(out.iloc[0]["workhorse_event"])


def test_season_census_aggregates_counts_and_prevalence():
    rows = pd.DataFrame(
        [
            {"season": 2019, "workhorse_event": True},
            {"season": 2019, "workhorse_event": False},
            {"season": 2019, "workhorse_event": False},
            {"season": 2020, "workhorse_event": True},
        ]
    )
    out = season_census(rows, [2019, 2020, 2021])
    assert out["2019"] == {"n_scored_transition_events": 3, "n_positive_workhorse_events": 1, "positive_prevalence": 1 / 3}
    assert out["2020"] == {"n_scored_transition_events": 1, "n_positive_workhorse_events": 1, "positive_prevalence": 1.0}
    assert out["2021"] == {"n_scored_transition_events": 0, "n_positive_workhorse_events": 0, "positive_prevalence": None}
