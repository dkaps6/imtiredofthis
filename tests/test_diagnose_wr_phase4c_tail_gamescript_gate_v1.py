import pandas as pd

from scripts.research.diagnose_wr_phase4c_tail_gamescript_gate_v1 import (
    PRIMARY,
    _evaluate_gate,
    _masks,
)


def test_frozen_cohort_directions_are_distinct():
    d = pd.DataFrame(
        {
            "actual_rec_yards": [120.0, 80.0, 20.0],
            "yard_residual": [40.0, -35.0, 31.0],
            "opportunity_yards": [30.0, -10.0, 5.0],
            "efficiency_yards": [10.0, -25.0, 20.0],
        }
    )
    m = _masks(d)
    assert m["ACTUAL_100_PLUS"].tolist() == [True, False, False]
    assert m["UNDERPROJECT_30_PLUS"].tolist() == [True, False, True]
    assert m["UNDERPROJECT_30_PLUS_OPP_DOM"].tolist() == [True, False, False]
    assert m["OVERPROJECT_30_PLUS"].tolist() == [False, True, False]


def _player_rows(season_2024_sign=1.0):
    rows = [
        dict(scope="POOLED", cohort=PRIMARY, family="continuous", concept="market_total", view="market_total", estimate=1.0, ci_low=0.2, ci_high=1.8),
        dict(scope="2023", cohort=PRIMARY, family="continuous", concept="market_total", view="market_total", estimate=0.5, ci_low=-0.1, ci_high=1.1),
        dict(scope="2024", cohort=PRIMARY, family="continuous", concept="market_total", view="market_total", estimate=season_2024_sign * 0.4, ci_low=-0.2, ci_high=1.0),
        dict(scope="POOLED", cohort="ACTUAL_100_PLUS_OPP_DOM", family="continuous", concept="market_total", view="market_total", estimate=0.3, ci_low=-0.2, ci_high=0.8),
        dict(scope="POOLED", cohort="UNDERPROJECT_30_PLUS", family="continuous", concept="market_total", view="market_total", estimate=0.2, ci_low=-0.2, ci_high=0.6),
    ]
    return pd.DataFrame(rows)


def test_gate_advances_only_when_all_frozen_conditions_align():
    player = _player_rows()
    l2corr = pd.DataFrame(
        [dict(scope="POOLED", concept="market_total", outcome="direct_team_target_residual", estimate=0.2)]
    )
    l2bucket = pd.DataFrame(columns=["scope", "family", "view", "estimate"])
    out = _evaluate_gate(player, l2corr, l2bucket)
    assert out["disposition"] == "ADVANCE_TO_SCRIPT_PREDICTOR_DESIGN"
    assert len(out["advancing_candidates"]) == 1


def test_gate_stops_on_season_sign_reversal():
    player = _player_rows(season_2024_sign=-1.0)
    l2corr = pd.DataFrame(
        [dict(scope="POOLED", concept="market_total", outcome="direct_team_target_residual", estimate=0.2)]
    )
    l2bucket = pd.DataFrame(columns=["scope", "family", "view", "estimate"])
    out = _evaluate_gate(player, l2corr, l2bucket)
    assert out["disposition"] == "NO_ACTIONABLE_TAIL_GAMESCRIPT_BRIDGE"


def test_gate_stops_without_layer2_bridge():
    player = _player_rows()
    l2corr = pd.DataFrame(
        [dict(scope="POOLED", concept="market_total", outcome="direct_team_target_residual", estimate=-0.2)]
    )
    l2bucket = pd.DataFrame(columns=["scope", "family", "view", "estimate"])
    out = _evaluate_gate(player, l2corr, l2bucket)
    assert out["disposition"] == "NO_ACTIONABLE_TAIL_GAMESCRIPT_BRIDGE"
