import pandas as pd
import pytest

from scripts.research.evaluate_prism_injury_partial_receiving_exposure_decomposition_v1 import (
    build_scored_rows,
)


def _row(week,state,pct,targets,team_targets=30.0):
    return {
        "season":2025,
        "week":week,
        "team":"CAR",
        "stable_id":"P1",
        "family":"WR",
        "offense_pct":pct,
        "targets":targets,
        "team_targets":team_targets,
        "target_share_game":targets/team_targets,
        "participation_anomaly_state":state,
    }


def test_injury_partial_contributes_to_earning_not_expected_ordinary_exposure():
    x=pd.DataFrame([
        _row(1,"PARTICIPATION_NOT_COLLAPSED",0.80,6),
        _row(2,"PARTICIPATION_NOT_COLLAPSED",0.82,7),
        _row(3,"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",0.30,4),
        _row(4,"PARTICIPATION_NOT_COLLAPSED",0.81,8),
        _row(5,"PARTICIPATION_NOT_COLLAPSED",0.80,7),
    ])
    rows,audit=build_scored_rows(x)
    assert audit["chronology_violations"] == 0
    assert len(rows) == 1
    r=rows.iloc[0]

    # Injury-partial row is included in total targets/earning intensity.
    expected_earning=(6+7+4+8)/(30*(0.80+0.82+0.30+0.81))
    assert r["earning_intensity"] == pytest.approx(expected_earning)

    # But ordinary expected participation comes only from the three comparable rows.
    assert r["expected_offense_pct"] == pytest.approx(0.81)
    assert r["prior_injury_partial_rows"] == 1
    assert r["prior_comparable_rows"] == 3


def test_unresolved_prior_anomaly_fails_closed():
    x=pd.DataFrame([
        _row(1,"PARTICIPATION_NOT_COLLAPSED",0.80,6),
        _row(2,"PARTICIPATION_COLLAPSE_UNRESOLVED",0.35,3),
        _row(3,"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",0.30,4),
        _row(4,"PARTICIPATION_NOT_COLLAPSED",0.81,8),
        _row(5,"PARTICIPATION_NOT_COLLAPSED",0.80,7),
    ])
    rows,audit=build_scored_rows(x)
    assert rows.empty
    assert audit["prior_unresolved_excluded"] == 1


def test_target_game_must_be_comparable_not_collapse():
    x=pd.DataFrame([
        _row(1,"PARTICIPATION_NOT_COLLAPSED",0.80,6),
        _row(2,"PARTICIPATION_NOT_COLLAPSED",0.82,7),
        _row(3,"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",0.30,4),
        _row(4,"PARTICIPATION_NOT_COLLAPSED",0.81,8),
        _row(5,"PARTICIPATION_COLLAPSE_INJURY_CORROBORATED",0.25,2),
    ])
    rows,audit=build_scored_rows(x)
    assert rows.empty
    assert audit["target_major_collapse_excluded"] >= 2


def test_without_recent_injury_partial_candidate_is_not_scored():
    x=pd.DataFrame([
        _row(1,"PARTICIPATION_NOT_COLLAPSED",0.80,6),
        _row(2,"PARTICIPATION_NOT_COLLAPSED",0.82,7),
        _row(3,"PARTICIPATION_NOT_COLLAPSED",0.79,6),
        _row(4,"PARTICIPATION_NOT_COLLAPSED",0.81,8),
        _row(5,"PARTICIPATION_NOT_COLLAPSED",0.80,7),
    ])
    rows,audit=build_scored_rows(x)
    assert rows.empty
    assert audit["no_injury_partial_history_excluded"] == 1
