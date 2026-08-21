import numpy as np
import pandas as pd

from scripts.backtest.trace_rushing_allocation_probability import _probability_transform, _metrics


def test_probability_transform_matches_simulator_cap_and_residual():
    clean, raw_sum, probs, residual = _probability_transform(np.array([0.60, 0.30, 0.20]))
    assert np.allclose(clean, [0.60, 0.30, 0.20])
    assert np.isclose(raw_sum, 1.10)
    assert np.isclose(probs.sum(), 0.95)
    assert np.isclose(residual, 0.05)
    assert np.isclose(probs[0], 0.60 * 0.95 / 1.10)


def test_probability_transform_preserves_subcap_shares():
    _, raw_sum, probs, residual = _probability_transform(np.array([0.40, 0.25, 0.10]))
    assert np.isclose(raw_sum, 0.75)
    assert np.allclose(probs, [0.40, 0.25, 0.10])
    assert np.isclose(residual, 0.25)


def test_stage_metrics_reports_expected_correlation():
    x = pd.DataFrame({
        "actual": [2.0, 5.0, 9.0],
        "mc_proj": [2.0, 5.0, 9.0],
        "expected_carries_from_final_probability": [2.0, 5.0, 9.0],
        "realized_multinomial_mean_carries": [2.0, 5.0, 9.0],
    })
    out = _metrics(x)
    assert len(out) == 3
    assert (out["mae"] == 0.0).all()
    assert np.allclose(out["correlation"], 1.0)
