import numpy as np
import pandas as pd

from scripts.football_context import qualify_ol_roster_continuity_v1 as cont
from scripts.football_context import qualify_ol_roster_pairwise_cohesion_v1 as coh
from scripts.football_context.test_ol_pairwise_cohesion_pressure_mechanism_v1 import (
    _matrix,
    fit_nested_models,
    primary_gate,
    replication_gate,
)


def _synthetic(n=1200):
    rng = np.random.default_rng(123)
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(n):
        c = float(rng.uniform(0, 1))
        p = float(rng.uniform(0, 1))
        rows.append({
            "season": 2020 + (i % 4),
            "week": 1 + (i % 18),
            "team": teams[i % len(teams)],
            coh.CANDIDATE: c,
            cont.CANDIDATE: float(rng.uniform(0, 1)),
            "prior_pressure_rate_allowed": p,
            "prior_success_rate_off": float(rng.uniform(0.3, 0.7)),
            "prior_dropback_rate": float(rng.uniform(0.4, 0.8)),
            "prior_plays_est": float(rng.uniform(50, 80)),
            "prior_proe": float(rng.uniform(-0.2, 0.2)),
            "target_pressure_rate_allowed": 0.30 + 0.15 * p - 0.08 * c,
        })
    return pd.DataFrame(rows)


def test_nested_fit_recovers_negative_cohesion_direction():
    fit = fit_nested_models(_synthetic())
    assert fit["train_rows"] == 1200
    assert fit["cohesion_coefficient"] < 0
    assert abs(fit["cohesion_coefficient"] + 0.08) < 1e-8


def test_train_encoder_does_not_add_unseen_team_column():
    train = _synthetic()
    fit = fit_nested_models(train)
    test = train.iloc[:5].copy()
    test["team"] = "UNSEEN"
    X = _matrix(test, fit["cand_enc"])
    assert X.shape[1] == len(fit["beta_candidate"])
    assert np.isfinite(X).all()


def test_primary_gate_is_all_gates_and_requires_negative_coefficient():
    result = {
        "scored_rows": 500,
        "scoring_coverage": 0.9,
        "mae_gain": 0.01,
        "bootstrap_ci_low": 0.001,
        "rmse_gain": 0.001,
        "p90_abs_error_gain": 0.0,
    }
    gate, passed = primary_gate(result, -0.01)
    assert passed
    gate2, passed2 = primary_gate(result, 0.01)
    assert not passed2
    assert not gate2.loc[gate2.gate.eq("cohesion_coefficient_direction"), "passed"].iloc[0]


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
