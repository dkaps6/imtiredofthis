import pandas as pd
import pytest

from scripts.research.evaluate_prism_comparable_observation_receiving_identity_v1 import (
    blend,
    construct_rows,
    should_exclude_history_row,
)


def test_blend_matches_playerform_four_game_prior_weight():
    assert blend(0.20, 0.30, 4) == pytest.approx(0.25)
    assert blend(0.20, 0.30, 8) == pytest.approx((1/3)*0.20 + (2/3)*0.30)
    assert blend(float("nan"), 0.30, 2) == pytest.approx(0.30)
    assert blend(0.20, float("nan"), 0) == pytest.approx(0.20)


def test_history_exclusion_requires_corroboration_before_target():
    anomalies = {
        (2024, 3, "CAR", "P1"): {
            "state": "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
            "corroboration_week": 4,
        },
        (2024, 2, "CAR", "P1"): {
            "state": "PARTICIPATION_COLLAPSE_UNRESOLVED",
            "corroboration_week": None,
        },
    }
    assert not should_exclude_history_row(
        row_season=2024,row_week=3,row_team="CAR",stable_id="P1",
        target_week=4,anomalies=anomalies,
    )
    assert should_exclude_history_row(
        row_season=2024,row_week=3,row_team="CAR",stable_id="P1",
        target_week=5,anomalies=anomalies,
    )
    assert not should_exclude_history_row(
        row_season=2024,row_week=2,row_team="CAR",stable_id="P1",
        target_week=5,anomalies=anomalies,
    )


def test_construct_rows_removes_only_legally_corroborated_history():
    ledger = pd.DataFrame([
        # Prior-season prior: 20% target share.
        {"season":2023,"week":1,"team":"CAR","stable_id":"P1","family":"WR","targets":2.0,"team_targets":10.0},
        # Current ordinary week: 40%.
        {"season":2024,"week":1,"team":"CAR","stable_id":"P1","family":"WR","targets":4.0,"team_targets":10.0},
        # Current injury-corroborated partial week: 0%.
        {"season":2024,"week":2,"team":"CAR","stable_id":"P1","family":"WR","targets":0.0,"team_targets":10.0},
        # Current unresolved collapse: 10%, must remain in candidate.
        {"season":2024,"week":3,"team":"CAR","stable_id":"P1","family":"WR","targets":1.0,"team_targets":10.0},
        # Target week actual: 40%.
        {"season":2024,"week":5,"team":"CAR","stable_id":"P1","family":"WR","targets":4.0,"team_targets":10.0},
        # Teammate ensures team target ledger is populated independently.
        {"season":2024,"week":5,"team":"CAR","stable_id":"P2","family":"WR","targets":6.0,"team_targets":10.0},
    ])
    health = pd.DataFrame([
        {"season":2024,"week":5,"team":"CAR","stable_id":"P1","family":"WR","positive_participation":True},
    ])
    anomalies = {
        (2024,2,"CAR","P1"): {
            "state":"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
            "corroboration_week":3,
        },
        (2024,3,"CAR","P1"): {
            "state":"PARTICIPATION_COLLAPSE_UNRESOLVED",
            "corroboration_week":None,
        },
        (2024,5,"CAR","P1"): {
            "state":"PARTICIPATION_NOT_COLLAPSED",
            "corroboration_week":None,
        },
    }

    rows,audit=construct_rows(health=health,ledger=ledger,anomalies=anomalies)
    r=rows.iloc[0]

    assert audit["chronology_violations"] == 0
    assert r["history_rows_removed"] == 1
    assert r["baseline_current_games"] == 3
    assert r["candidate_current_games"] == 2
    assert bool(r["candidate_changed"]) is True
    # Candidate should move upward because the zero-share injury-partial row is removed.
    assert r["candidate_target_share"] > r["baseline_target_share"]
    assert r["player_game_improvement"] > 0


def test_abnormal_target_game_is_not_primary_role_evidence():
    ledger = pd.DataFrame([
        {"season":2023,"week":1,"team":"CAR","stable_id":"P1","family":"WR","targets":2.0,"team_targets":10.0},
        {"season":2024,"week":1,"team":"CAR","stable_id":"P1","family":"WR","targets":4.0,"team_targets":10.0},
        {"season":2024,"week":5,"team":"CAR","stable_id":"P1","family":"WR","targets":1.0,"team_targets":10.0},
        {"season":2024,"week":5,"team":"CAR","stable_id":"P2","family":"WR","targets":9.0,"team_targets":10.0},
    ])
    health = pd.DataFrame([
        {"season":2024,"week":5,"team":"CAR","stable_id":"P1","family":"WR","positive_participation":True},
    ])
    anomalies = {
        (2024,5,"CAR","P1"): {
            "state":"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",
            "corroboration_week":5,
        }
    }
    rows,audit=construct_rows(health=health,ledger=ledger,anomalies=anomalies)
    assert rows.empty
    assert audit["abnormal_target_games_excluded"] == 1
