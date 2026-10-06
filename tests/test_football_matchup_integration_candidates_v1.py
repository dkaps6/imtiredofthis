import numpy as np
import pandas as pd

import scripts.research.score_football_matchup_integration_candidates_v1 as m


def test_candidate_registry_is_exact_and_narrow():
    got = [
        (c.candidate_id, c.cohort, c.market, c.position, c.feature, c.sign)
        for c in m.CANDIDATES
    ]
    assert got == [
        ("FMT-RB1", "RB_RUSH", "rush_yards", "RB", "def_pass_rate_faced", -1),
        ("FMT-WR1", "WR_REC", "rec_yards", "WR", "off_true_proe", 1),
        (
            "FMT-TE1",
            "TE_REC",
            "rec_yards",
            "TE",
            "def_pass_success_allowed",
            1,
        ),
    ]
    assert all("ypt" not in c.feature for c in m.CANDIDATES)
    assert all("def_rush_epa" != c.feature for c in m.CANDIDATES)


def test_zero_intercept_beta_recovers_known_slope():
    rows = []
    xs = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), 60)
    for i, x in enumerate(xs):
        rows.append(
            {
                "season": 2022,
                "game_id": f"g{i % 60}",
                "actual": 100.0 + 3.0 * x,
                "baseline_projection": 100.0,
                "weakness_z": x,
            }
        )
    out = m.fit_beta(pd.DataFrame(rows))
    assert out["support"] is True
    assert out["positive_beta"] is True
    assert abs(out["beta_train"] - 3.0) < 1e-12


def test_negative_training_slope_fails_sign_gate():
    rows = []
    xs = np.tile(np.array([-1.5, -0.5, 0.5, 1.5]), 60)
    for i, x in enumerate(xs):
        rows.append(
            {
                "season": 2022,
                "game_id": f"g{i % 60}",
                "actual": 100.0 - 2.0 * x,
                "baseline_projection": 100.0,
                "weakness_z": x,
            }
        )
    out = m.fit_beta(pd.DataFrame(rows))
    assert out["support"] is True
    assert out["beta_train"] < 0
    assert out["positive_beta"] is False


def test_candidate_score_confirms_clean_synthetic_signal(monkeypatch):
    monkeypatch.setattr(m, "BOOT_REPS", 200)
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for i in range(240):
            x = [-1.5, -0.5, 0.5, 1.5][i % 4]
            noise = 0.15 * np.sin(i * 0.7 + season)
            baseline = 50.0
            actual = baseline + 2.0 * x + noise
            rows.append(
                {
                    "season": season,
                    "week": 2 + (i % 17),
                    "game_id": f"{season}_g{i % 60}",
                    "player_identity_key": f"p{i % 80}",
                    "actual": actual,
                    "baseline_projection": baseline,
                    "weakness_z": x,
                }
            )
    q = pd.DataFrame(rows)
    result, score = m.score_candidate(q, m.CANDIDATES[1], 0)
    assert result["train"]["positive_beta"] is True
    assert abs(result["train"]["beta_train"] - 2.0) < 0.02
    assert result["primary_2023_gates"]["mae_improves"] is True
    assert result["primary_2023_gates"]["rmse_nonworse"] is True
    assert result["primary_2023_gates"]["bootstrap_p_ge_080"] is True
    assert result["secondary_2024_2025_gates"]["mae_nonworse_2024"] is True
    assert result["secondary_2024_2025_gates"]["mae_nonworse_2025"] is True
    assert result["disposition"] == "INTEGRATION_CANDIDATE_CONFIRMED"
    assert len(score) == 4


def test_build_team_features_uses_only_prior_rows():
    team = pd.DataFrame(
        [
            {
                "season": 2021,
                "week": 18,
                "team": "ATL",
                "true_proe": -0.10,
                "pass_rate_faced": 0.55,
                "def_pass_success_allowed": 0.45,
            },
            {
                "season": 2021,
                "week": 18,
                "team": "NO",
                "true_proe": 0.05,
                "pass_rate_faced": 0.40,
                "def_pass_success_allowed": 0.60,
            },
            {
                "season": 2022,
                "week": 2,
                "team": "ATL",
                "true_proe": 9.99,
                "pass_rate_faced": 9.99,
                "def_pass_success_allowed": 9.99,
            },
            {
                "season": 2022,
                "week": 2,
                "team": "NO",
                "true_proe": 9.99,
                "pass_rate_faced": 9.99,
                "def_pass_success_allowed": 9.99,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {"season": 2022, "week": 2, "team": "ATL", "opponent": "NO"},
            {"season": 2022, "week": 2, "team": "NO", "opponent": "ATL"},
        ]
    )
    out = m.build_team_features(team, schedule)
    atl = out.loc[out["team"].eq("ATL")].iloc[0]
    assert atl["off_true_proe"] == -0.10
    assert atl["def_pass_rate_faced"] == 0.40
    assert atl["def_pass_success_allowed"] == 0.60
    assert np.isfinite(atl["off_true_proe__z"])
