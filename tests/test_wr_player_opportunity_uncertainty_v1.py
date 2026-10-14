import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_player_opportunity_uncertainty_v1 as m


def _frame():
    rows = []
    for season in (2023, 2024):
        for week in (5, 6):
            for i, mae in enumerate((6.0, 10.0, 14.0, 18.0)):
                rows.append({
                    "season": season,
                    "week": week,
                    "team": f"T{i}",
                    "player_clean_key": f"p{i}",
                    "wr_rank": i + 1,
                    "prior_games": 4,
                    "prior8_abs_opportunity_error_mean": mae,
                    "current_opportunity_error": float(i - 1),
                })
    return pd.DataFrame(rows)


def test_player_scale_and_week_pool_are_separate():
    q = m.prepare(_frame())
    w = q.loc[(q.season == 2023) & (q.week == 5)].sort_values("player_mae")
    assert np.allclose(w["week_pool_mae"].to_numpy(), 12.0)
    assert np.allclose(
        w["candidate_sigma"].to_numpy(),
        w["player_mae"].to_numpy() * m.SQRT_PI_OVER_2,
    )
    assert np.allclose(
        w["baseline_sigma"].to_numpy(),
        np.full(len(w), 12.0 * m.SQRT_PI_OVER_2),
    )


def test_normal_crps_same_scale_same_score():
    y = np.array([-3.0, 0.0, 8.0])
    sigma = np.array([7.0, 7.0, 7.0])
    assert np.allclose(m.normal_crps_zero(y, sigma), m.normal_crps_zero(y, sigma.copy()))


def test_role_is_descriptive_only_and_mean_is_untouched():
    x = _frame()
    q = m.prepare(x)
    assert set(q["role"]) == {"WR1", "WR2+"}
    expected = x.sort_values(["season", "week", "team", "player_clean_key"])[
        "current_opportunity_error"
    ].to_numpy()
    assert np.allclose(q["current_opportunity_error"].to_numpy(), expected)
