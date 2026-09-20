import numpy as np
import pandas as pd

from scripts.football_context import qualify_ol_roster_continuity_v1 as cont
from scripts.football_context.qualify_ol_roster_pairwise_cohesion_v1 import (
    CANDIDATE,
    LOOKBACK_GAMES,
    _redundancy_design,
    materialize_pairwise_cohesion,
)


def test_cross_season_prior_history_is_used_strictly_prior():
    schedule = pd.DataFrame([
        {"season": 2023, "week": 17, "team": "A"},
        {"season": 2023, "week": 18, "team": "A"},
        {"season": 2024, "week": 1, "team": "A"},
    ])
    schedule = cont.normalize_schedule(schedule)
    sets = {
        (2023, 17, "A"): {"a", "b", "x"},
        (2023, 18, "A"): {"a", "b", "y"},
        (2024, 1, "A"): {"a", "b", "z"},
    }
    out, integrity = materialize_pairwise_cohesion(schedule, sets)
    row = out[(out.season == 2024) & (out.week == 1)].iloc[0]

    # Current pairs: ab, az, bz. Across two prior games ab=2, az=0, bz=0.
    # Mean pair share = (1 + 0 + 0) / 3.
    assert abs(row[CANDIDATE] - (1 / 3)) < 1e-12
    assert row.prior_roster_games_available == 2
    assert row.cohesion_state == "KNOWN"
    assert integrity["chronology_violations"] == 0
    assert integrity["target_game_pbp_used_in_cohesion"] is False


def test_first_available_team_game_is_unknown_not_zero():
    schedule = cont.normalize_schedule(pd.DataFrame([
        {"season": 2019, "week": 1, "team": "A"},
    ]))
    out, _ = materialize_pairwise_cohesion(
        schedule, {(2019, 1, "A"): {"a", "b", "c"}}
    )
    row = out.iloc[0]
    assert row.cohesion_state == "UNKNOWN_NO_PRIOR_HISTORY"
    assert np.isnan(row[CANDIDATE])


def test_current_roster_with_fewer_than_two_ids_is_unknown():
    schedule = cont.normalize_schedule(pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A"},
        {"season": 2024, "week": 2, "team": "A"},
    ]))
    sets = {
        (2024, 1, "A"): {"a", "b"},
        (2024, 2, "A"): {"a"},
    }
    out, _ = materialize_pairwise_cohesion(schedule, sets)
    row = out[out.week == 2].iloc[0]
    assert row.cohesion_state == "UNKNOWN_CURRENT_ROSTER_LT2"
    assert np.isnan(row[CANDIDATE])


def test_lookback_is_capped_at_twenty_scheduled_games():
    schedule = cont.normalize_schedule(pd.DataFrame([
        {"season": 2024, "week": w, "team": "A"}
        for w in range(1, 23)
    ]))
    sets = {
        (2024, w, "A"): {"a", "b"} if w >= 2 else {"x", "y"}
        for w in range(1, 23)
    }
    out, _ = materialize_pairwise_cohesion(schedule, sets)
    row = out[out.week == 22].iloc[0]
    assert LOOKBACK_GAMES == 20
    assert row.prior_scheduled_games_in_window == 20
    assert row.prior_roster_games_available == 20
    assert abs(row[CANDIDATE] - 1.0) < 1e-12


def test_redundancy_row_floors_fail_closed():
    rows = []
    for i in range(1499):
        train = i < 999
        rows.append({
            "season": 2020 if train else 2024,
            "week": 2 + (i % 16),
            "team": f"T{i % 10}",
            CANDIDATE: 0.6,
            cont.CANDIDATE: 0.8,
            "prior_pressure_rate_allowed": 0.2,
            "prior_success_rate_off": 0.5,
            "prior_dropback_rate": 0.6,
            "prior_plays_est": 65,
            "prior_proe": 0.03,
        })
    Xtr, ytr, Xte, yte, ntr, nte = _redundancy_design(pd.DataFrame(rows))
    assert ntr == 999 and nte == 500
    assert Xtr.size == 0 and Xte.size == 0


def test_redundancy_uses_train_schema_and_train_imputation():
    rows = []
    for i in range(1500):
        train = i < 1000
        rows.append({
            "season": 2020 if train else 2024,
            "week": 2 + (i % 16),
            "team": f"T{i % 8}" if train else "NEW_HOLDOUT_TEAM",
            CANDIDATE: 0.4 + (i % 4) / 10,
            cont.CANDIDATE: np.nan if i % 9 == 0 else 0.8,
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
