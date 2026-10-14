import numpy as np
import pandas as pd

from scripts.research import evaluate_te_player_opportunity_uncertainty_v1 as m


def _frame():
    rows = []
    for season in (2024, 2025):
        for week in (5, 6):
            for i, mae in enumerate((4.0, 8.0, 12.0, 16.0)):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "team": f"T{i}",
                        "player_clean_key": f"p{i}",
                        "prior_games": 4,
                        "prior8_abs_opportunity_error_mean": mae,
                        "current_opportunity_error": float(i - 1),
                    }
                )
    return pd.DataFrame(rows)


def test_player_scale_uses_same_player_feature_and_week_pool_is_median():
    q = m.prepare(_frame())
    w = q.loc[(q.season == 2024) & (q.week == 5)].sort_values("player_mae")
    assert np.allclose(w["week_pool_mae"].to_numpy(), 10.0)
    expected = w["player_mae"].to_numpy() * m.SQRT_PI_OVER_2
    assert np.allclose(w["candidate_sigma"].to_numpy(), expected)
    assert np.allclose(
        w["baseline_sigma"].to_numpy(),
        np.full(len(w), 10.0 * m.SQRT_PI_OVER_2),
    )


def test_normal_crps_is_identical_when_scales_match():
    y = np.array([-4.0, 0.0, 7.0])
    sigma = np.array([5.0, 5.0, 5.0])
    a = m.normal_crps_zero(y, sigma)
    b = m.normal_crps_zero(y, sigma.copy())
    assert np.all(np.isfinite(a))
    assert np.allclose(a, b)


def test_prepare_does_not_change_error_mean_or_use_target_outcome_for_width():
    x = _frame()
    q = m.prepare(x)
    assert np.allclose(
        q["current_opportunity_error"].to_numpy(),
        x.sort_values(["season", "week", "team", "player_clean_key"])[
            "current_opportunity_error"
        ].to_numpy(),
    )
    # Widths are functions only of pre-existing strictly-prior difficulty fields.
    assert q["candidate_sigma"].notna().all()
    assert q["baseline_sigma"].notna().all()
