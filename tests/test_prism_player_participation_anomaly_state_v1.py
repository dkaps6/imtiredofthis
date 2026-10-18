import pandas as pd

from scripts.research.audit_prism_player_participation_anomaly_state_v1 import (
    STATE_COLLAPSE_CORROBORATED,
    STATE_COLLAPSE_UNRESOLVED,
    STATE_NOT_COLLAPSED,
    construct_states,
    load_parent,
)


def _base_rows():
    rows = []
    # Player A establishes an 80% same-team participation role.
    for week, pct in [(1, 0.80), (2, 0.82), (3, 0.78), (4, 0.81)]:
        rows.append({
            "season": 2024,
            "week": week,
            "team": "AAA",
            "history_key": "ID:A",
            "stable_id": "A",
            "family": "WR",
            "positive_participation": True,
            "offense_snaps": 50,
            "offense_pct": pct,
            "injury_reported": False,
            "observation_state": "NO_REPORT_PARTICIPATED",
        })

    # Week 5 collapses to 30%; no same-week report.
    rows.append({
        "season": 2024,
        "week": 5,
        "team": "AAA",
        "history_key": "ID:A",
        "stable_id": "A",
        "family": "WR",
        "positive_participation": True,
        "offense_snaps": 20,
        "offense_pct": 0.30,
        "injury_reported": False,
        "observation_state": "NO_REPORT_PARTICIPATED",
    })

    # Week 6 injury report corroborates Week 5 retrospectively.
    rows.append({
        "season": 2024,
        "week": 6,
        "team": "AAA",
        "history_key": "ID:A",
        "stable_id": "A",
        "family": "WR",
        "positive_participation": False,
        "offense_snaps": 0,
        "offense_pct": 0.0,
        "injury_reported": True,
        "observation_state": "REPORTED_INJURY_NO_OFFENSIVE_SNAPS",
    })

    # Player B has a similar collapse but no injury corroboration.
    for week, pct in [(1, 0.70), (2, 0.72), (3, 0.69), (4, 0.71)]:
        rows.append({
            "season": 2024,
            "week": week,
            "team": "BBB",
            "history_key": "ID:B",
            "stable_id": "B",
            "family": "TE",
            "positive_participation": True,
            "offense_snaps": 45,
            "offense_pct": pct,
            "injury_reported": False,
            "observation_state": "NO_REPORT_PARTICIPATED",
        })
    rows.append({
        "season": 2024,
        "week": 5,
        "team": "BBB",
        "history_key": "ID:B",
        "stable_id": "B",
        "family": "TE",
        "positive_participation": True,
        "offense_snaps": 18,
        "offense_pct": 0.25,
        "injury_reported": False,
        "observation_state": "NO_REPORT_PARTICIPATED",
    })
    rows.append({
        "season": 2024,
        "week": 6,
        "team": "BBB",
        "history_key": "ID:B",
        "stable_id": "B",
        "family": "TE",
        "positive_participation": True,
        "offense_snaps": 44,
        "offense_pct": 0.70,
        "injury_reported": False,
        "observation_state": "NO_REPORT_PARTICIPATED",
    })
    return pd.DataFrame(rows)


def test_player_specific_collapse_and_future_corroboration():
    out, audit = construct_states(_base_rows())
    a = out.loc[(out["history_key"] == "ID:A") & (out["week"] == 5)].iloc[0]

    assert a["participation_anomaly_state"] == STATE_COLLAPSE_CORROBORATED
    assert a["corroboration_source"] == "NEXT_SCHEDULED_GAME_INJURY"
    assert a["corroboration_week"] == 6
    assert a["player_prior_snap_pct"] > 0.79
    assert a["snap_retention_ratio"] < 0.40
    assert audit["chronology_violations"] == 0
    assert audit["next_corroboration_order_violations"] == 0


def test_collapse_without_injury_remains_unresolved():
    out, _ = construct_states(_base_rows())
    b = out.loc[(out["history_key"] == "ID:B") & (out["week"] == 5)].iloc[0]
    assert b["participation_anomaly_state"] == STATE_COLLAPSE_UNRESOLVED
    assert b["corroboration_source"] == "NONE"


def test_normal_participation_is_not_collapsed():
    out, _ = construct_states(_base_rows())
    b6 = out.loc[(out["history_key"] == "ID:B") & (out["week"] == 6)].iloc[0]
    assert b6["participation_anomaly_state"] == STATE_NOT_COLLAPSED


def test_baseline_is_same_player_same_team_only():
    rows = _base_rows().copy()
    # Add huge-snap unrelated WR rows; they must not alter A's baseline.
    extra = []
    for week in range(1, 5):
        extra.append({
            "season": 2024,
            "week": week,
            "team": "AAA",
            "history_key": "ID:OTHER",
            "stable_id": "OTHER",
            "family": "WR",
            "positive_participation": True,
            "offense_snaps": 70,
            "offense_pct": 1.0,
            "injury_reported": False,
            "observation_state": "NO_REPORT_PARTICIPATED",
        })
    out, _ = construct_states(pd.concat([rows, pd.DataFrame(extra)], ignore_index=True))
    a = out.loc[(out["history_key"] == "ID:A") & (out["week"] == 5)].iloc[0]
    assert 0.79 < a["player_prior_snap_pct"] < 0.82
