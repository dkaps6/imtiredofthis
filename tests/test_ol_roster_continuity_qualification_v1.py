import numpy as np
import pandas as pd

from scripts.football_context.qualify_ol_roster_continuity_v1 import (
    CANDIDATE,
    _redundancy_design,
    build_roster_sets,
    materialize_continuity,
    normalize_schedule,
)


def _schedule():
    # Team A has a bye-like gap between W2 and W4. Previous GAME must be W2.
    return pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A"},
        {"season": 2024, "week": 2, "team": "A"},
        {"season": 2024, "week": 4, "team": "A"},
        {"season": 2024, "week": 1, "team": "B"},
        {"season": 2024, "week": 2, "team": "B"},
    ])


def _roster():
    rows = []
    by_week = {
        ("A", 1): ["a1", "a2", "a3"],
        ("A", 2): ["a1", "a2", "a4"],
        ("A", 4): ["a1", "a4", "a5"],
        ("B", 1): ["b1", "b2"],
        ("B", 2): ["b1", "b2"],
    }
    for (team, week), ids in by_week.items():
        for i, pid in enumerate(ids):
            rows.append({
                "season": 2024, "week": week, "team": team,
                "gsis_id": pid, "position": "T" if i == 0 else "G",
                "depth_chart_position": "LT" if i == 0 else "RG",
                # Deliberate target-game-like columns must be ignored.
                "snap_count": 0 if i else 999,
                "games_played": 0,
            })
        # Backup is intentionally retained.
        rows.append({
            "season": 2024, "week": week, "team": team,
            "gsis_id": f"{team.lower()}backup{week}", "position": "OL",
            "depth_chart_position": "OL", "snap_count": 0, "games_played": 0,
        })
        # Non-OL excluded.
        rows.append({
            "season": 2024, "week": week, "team": team,
            "gsis_id": f"{team.lower()}wr{week}", "position": "WR",
            "depth_chart_position": "WR", "snap_count": 999, "games_played": 1,
        })
    return pd.DataFrame(rows)


def test_week1_is_explicit_unknown_and_backup_is_retained():
    schedule = normalize_schedule(_schedule())
    sets, report = build_roster_sets(_roster(), schedule)
    out, integrity = materialize_continuity(schedule, sets)
    a1 = out[(out.team == "A") & (out.week == 1)].iloc[0]
    assert a1.continuity_state == "UNKNOWN_NO_PRIOR_GAME"
    assert np.isnan(a1[CANDIDATE])
    assert a1.current_ol_roster_count == 4
    assert report["target_game_snap_or_participation_used"] is False
    assert integrity["target_game_pbp_used_in_continuity"] is False


def test_previous_scheduled_game_not_previous_calendar_week():
    schedule = normalize_schedule(_schedule())
    sets, _ = build_roster_sets(_roster(), schedule)
    out, _ = materialize_continuity(schedule, sets)
    a4 = out[(out.team == "A") & (out.week == 4)].iloc[0]
    assert a4.prior_game_week == 2
    # W4: current a1,a4,a5,backup4; W2: a1,a2,a4,backup2 -> 2/4 return.
    assert abs(a4[CANDIDATE] - 0.5) < 1e-12
    assert a4.returning_ol_count == 2
    assert a4.added_ol_count == 2
    assert a4.departed_ol_count == 2


def test_same_week_gsis_on_two_teams_is_integrity_conflict():
    schedule = normalize_schedule(_schedule())
    roster = _roster()
    conflict = roster[(roster.team == "A") & (roster.week == 2)].iloc[[0]].copy()
    conflict["team"] = "B"
    roster = pd.concat([roster, conflict], ignore_index=True)
    _, report = build_roster_sets(roster, schedule)
    assert report["ambiguous_same_week_gsis_team_conflicts"] == 1


def test_redundancy_row_floors_are_fail_closed():
    rows = []
    # 999 train, 500 holdout.
    for i in range(1499):
        train = i < 999
        season = 2020 if train else 2024
        rows.append({
            "season": season,
            "week": 2 + (i % 16),
            "team": f"T{i % 10}",
            CANDIDATE: 0.8,
            "prior_pressure_rate_allowed": 0.2,
            "prior_success_rate_off": 0.5,
            "prior_dropback_rate": 0.6,
            "prior_plays_est": 65,
            "prior_proe": 0.03,
        })
    Xtr, ytr, Xte, yte, ntr, nte = _redundancy_design(pd.DataFrame(rows))
    assert ntr == 999 and nte == 500
    assert Xtr.size == 0 and Xte.size == 0


def test_redundancy_encoding_uses_train_team_schema_and_train_imputation():
    rows = []
    for i in range(1500):
        train = i < 1000
        season = 2020 if train else 2024
        rows.append({
            "season": season,
            "week": 2 + (i % 16),
            "team": f"T{i % 8}" if train else "NEW_HOLDOUT_TEAM",
            CANDIDATE: 0.7 + (i % 3) / 10,
            "prior_pressure_rate_allowed": np.nan if i % 11 == 0 else 0.2,
            "prior_success_rate_off": 0.5,
            "prior_dropback_rate": 0.6,
            "prior_plays_est": 65,
            "prior_proe": 0.03,
        })
    Xtr, ytr, Xte, yte, ntr, nte = _redundancy_design(pd.DataFrame(rows))
    assert ntr == 1000 and nte == 500
    assert Xtr.shape[1] == Xte.shape[1]
    assert np.isfinite(Xtr).all()
    assert np.isfinite(Xte).all()
