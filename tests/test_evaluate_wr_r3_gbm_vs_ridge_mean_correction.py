import numpy as np
import pandas as pd

from scripts.backtest.evaluate_wr_r3_gbm_vs_ridge_mean_correction import (
    FEATURE_COLS,
    apply_model_correction,
    fit_frozen_models,
)


def _synthetic_train(n=300, seed=11):
    rng = np.random.default_rng(seed)
    bias = rng.uniform(-15, 15, n)
    mae = rng.uniform(5, 30, n)
    miss30 = rng.uniform(0, 0.5, n)
    prior_games = rng.integers(4, 17, n)
    # true residual is a simple linear function of bias plus noise -- both
    # Ridge and the GBM should beat the naive zero-correction baseline here.
    residual = -0.3 * bias + rng.normal(0, 2.0, n)
    return pd.DataFrame({
        "prior8_m38_bias": bias, "prior8_m38_mae": mae,
        "prior8_m38_miss30_rate": miss30, "prior_games": prior_games,
        "residual": residual,
    })


def test_fit_frozen_models_beats_naive_zero_correction_on_linear_signal():
    train = _synthetic_train()
    ridge, gbm, diag = fit_frozen_models(train)
    assert diag["ridge_train_mae"] < diag["naive_zero_correction_train_mae"]
    assert diag["gbm_train_mae"] < diag["naive_zero_correction_train_mae"]
    assert diag["train_rows"] == len(train)

    # frozen models must be directly usable for row-level prediction, exactly
    # how apply_model_correction calls them.
    row = train[FEATURE_COLS].iloc[[0]].to_numpy(dtype=float)
    assert np.isfinite(ridge.predict(row)[0])
    assert np.isfinite(gbm.predict(row)[0])


def test_apply_model_correction_clips_to_frozen_bound_and_valid_ypt_range():
    metrics = pd.DataFrame([
        {
            "event_id": "e1", "team": "KC", "player_clean_key": "wr1",
            "position": "WR", "rules_ypt": 7.0,
        },
    ])
    targets = pd.DataFrame([
        {"event_id": "e1", "team": "KC", "player_key": "wr1", "pred_targets": 5.0, "wr_role": "WR1"},
    ])
    fmap = {
        (2025, 1, "wr1"): {
            "prior_games": 8, "prior8_m38_bias": -20.0, "prior8_m38_mae": 10.0,
            "prior8_m38_miss30_rate": 0.2,
        }
    }

    class _HugeCorrectionModel:
        def predict(self, X):
            return np.array([1000.0])

    out = apply_model_correction(metrics, targets, fmap, 2025, 1, _HugeCorrectionModel())
    # correction clipped to +/-8 yards before dividing by pred_targets, and the
    # resulting rules_ypt clipped to the same [2, 20] bound apply_candidate uses.
    expected_ypt = float(np.clip(7.0 + 8.0 / 5.0, 2.0, 20.0))
    assert out.loc[0, "rules_ypt"] == expected_ypt


def test_apply_model_correction_skips_rows_below_min_prior():
    metrics = pd.DataFrame([
        {"event_id": "e1", "team": "KC", "player_clean_key": "wr1", "position": "WR", "rules_ypt": 7.0},
    ])
    targets = pd.DataFrame([
        {"event_id": "e1", "team": "KC", "player_key": "wr1", "pred_targets": 5.0, "wr_role": "WR1"},
    ])
    fmap = {(2025, 1, "wr1"): {"prior_games": 1, "prior8_m38_bias": -20.0}}

    class _AnyModel:
        def predict(self, X):
            return np.array([5.0])

    out = apply_model_correction(metrics, targets, fmap, 2025, 1, _AnyModel())
    assert out.loc[0, "rules_ypt"] == 7.0
