import numpy as np
import pandas as pd

from scripts.football_context import qualify_defensive_front_pairwise_cohesion_v1 as coh
from scripts.football_context.test_defensive_front_pairwise_cohesion_pressure_mechanism_v1 import (
    _matrix,
    attach_target_and_prior_states,
    fit_nested_models,
    primary_gate,
    replication_gate,
)


def _synthetic(n=1200):
    rng = np.random.default_rng(321)
    teams = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        c = float(rng.uniform(0, 1))
        prior_pressure = float(rng.uniform(0.1, 0.5))
        team = teams[i % 4]
        opponent = teams[(i + (i // 4) + 1) % 4]
        rows.append({
            "season": 2020 + (i % 4),
            "week": 1 + (i % 18),
            "team": team,
            "opponent": opponent,
            coh.CANDIDATE: c,
            coh.IMMEDIATE: float(rng.uniform(0, 1)),
            "prior_def_pressure_rate_generated": prior_pressure,
            "prior_def_success_rate_def": float(rng.uniform(0.3, 0.7)),
            "prior_def_def_pass_epa": float(rng.uniform(-0.3, 0.3)),
            "prior_def_explosive_play_rate_allowed": float(rng.uniform(0.05, 0.25)),
            "prior_opp_pressure_rate_allowed": float(rng.uniform(0.1, 0.5)),
            "prior_opp_success_rate_off": float(rng.uniform(0.3, 0.7)),
            "prior_opp_dropback_rate": float(rng.uniform(0.4, 0.8)),
            "prior_opp_plays_est": float(rng.uniform(50, 80)),
            "prior_opp_proe": float(rng.uniform(-0.2, 0.2)),
            "target_pressure_rate_generated": 0.20 + 0.10 * prior_pressure + 0.08 * c,
        })
    return pd.DataFrame(rows)


def test_nested_fit_recovers_positive_cohesion_direction():
    fit = fit_nested_models(_synthetic())
    assert fit["train_rows"] == 1200
    assert fit["cohesion_coefficient"] > 0
    assert abs(fit["cohesion_coefficient"] - 0.08) < 1e-8


def test_train_encoder_does_not_add_unseen_team_or_opponent_columns():
    train = _synthetic()
    fit = fit_nested_models(train)
    test = train.iloc[:5].copy()
    test["team"] = "UNSEEN_TEAM"
    test["opponent"] = "UNSEEN_OPP"
    X = _matrix(test, fit["cand_enc"])
    assert X.shape[1] == len(fit["beta_candidate"])
    assert np.isfinite(X).all()


def test_primary_gate_is_all_gates_and_requires_positive_coefficient():
    result = {
        "scored_rows": 500,
        "scoring_coverage": 0.9,
        "mae_gain": 0.01,
        "bootstrap_ci_low": 0.001,
        "rmse_gain": 0.001,
        "p90_abs_error_gain": 0.0,
    }
    gate, passed = primary_gate(result, 0.01)
    assert passed
    gate2, passed2 = primary_gate(result, -0.01)
    assert not passed2
    assert not gate2.loc[
        gate2.gate.eq("cohesion_coefficient_direction"), "passed"
    ].iloc[0]


def test_replication_gate_has_no_refit_direction_gate():
    result = {
        "scored_rows": 500,
        "scoring_coverage": 0.9,
        "mae_gain": 0.01,
        "bootstrap_ci_low": 0.001,
        "rmse_gain": 0.001,
        "p90_abs_error_gain": 0.0,
    }
    gate, passed = replication_gate(result)
    assert passed
    assert "cohesion_coefficient_direction" not in set(gate.gate)


def test_prior_opponent_state_uses_opponents_previous_scheduled_game():
    schedule = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A", "opponent": "B"},
        {"season": 2024, "week": 1, "team": "B", "opponent": "A"},
        {"season": 2024, "week": 2, "team": "A", "opponent": "B"},
        {"season": 2024, "week": 2, "team": "B", "opponent": "A"},
    ])
    weekly = pd.DataFrame([
        {
            "season": 2024, "week": 1, "team": "A",
            "pressure_rate_generated": 0.20, "success_rate_def": 0.45,
            "def_pass_epa": -0.1, "explosive_play_rate_allowed": 0.10,
            "pressure_rate_allowed": 0.30, "success_rate_off": 0.50,
            "dropback_rate": 0.60, "plays_est": 60, "proe": 0.02,
        },
        {
            "season": 2024, "week": 1, "team": "B",
            "pressure_rate_generated": 0.40, "success_rate_def": 0.55,
            "def_pass_epa": 0.1, "explosive_play_rate_allowed": 0.20,
            "pressure_rate_allowed": 0.25, "success_rate_off": 0.48,
            "dropback_rate": 0.57, "plays_est": 64, "proe": -0.01,
        },
        {
            "season": 2024, "week": 2, "team": "A",
            "pressure_rate_generated": 0.22, "success_rate_def": 0.46,
            "def_pass_epa": -0.08, "explosive_play_rate_allowed": 0.11,
            "pressure_rate_allowed": 0.31, "success_rate_off": 0.51,
            "dropback_rate": 0.61, "plays_est": 61, "proe": 0.03,
        },
        {
            "season": 2024, "week": 2, "team": "B",
            "pressure_rate_generated": 0.38, "success_rate_def": 0.54,
            "def_pass_epa": 0.08, "explosive_play_rate_allowed": 0.19,
            "pressure_rate_allowed": 0.27, "success_rate_off": 0.49,
            "dropback_rate": 0.58, "plays_est": 63, "proe": 0.00,
        },
    ])
    out, integrity = attach_target_and_prior_states(schedule, weekly)
    row = out.loc[(out.team == "A") & (out.week == 2)].iloc[0]
    assert row["prior_def_pressure_rate_generated"] == 0.20
    assert row["prior_opp_pressure_rate_allowed"] == 0.25
    assert integrity["prior_def_state_chronology_violations"] == 0
    assert integrity["prior_opponent_state_chronology_violations"] == 0
