import numpy as np
import pandas as pd
import pytest

from scripts.modeling.qb_pass_synthesis_v1 import (
    attempt_conversion,
    load_artifact,
    predict_correction,
)


def test_promoted_artifact_preserves_exact_m89_football_contract():
    art = load_artifact()
    assert art["ridge_alpha"] == 20.0
    assert art["residual_cap"] == 45.0
    assert len(art["feature_contract"]) == 21
    assert not any(
        token in name
        for name in art["feature_contract"]
        for token in ("spread", "total", "moneyline", "implied", "sportsbook", "vegas")
    )
    assert art["training_seasons"] == [2023, 2024, 2025]
    assert art["training_rows"] == 1331


def test_promoted_inference_is_finite_and_respects_residual_cap():
    art = load_artifact()
    features = dict(zip(art["feature_contract"], art["imputer_statistics"]))
    features["base_proj"] = 250.0
    pred, correction, version = predict_correction(features, artifact=art)
    assert np.isfinite(pred)
    assert np.isfinite(correction)
    assert abs(correction) <= 45.0
    assert np.isclose(pred, 250.0 + correction)
    assert version == art["version"]


def test_promoted_inference_uses_frozen_missing_value_contract():
    art = load_artifact()
    features = {name: np.nan for name in art["feature_contract"]}
    features["base_proj"] = 225.0
    pred, correction, _ = predict_correction(features, artifact=art)
    assert np.isfinite(pred)
    assert abs(correction) <= 45.0


def test_attempt_conversion_uses_promoted_team_context_and_rejects_bad_values():
    row = pd.Series({"team": "BUF"})
    good = pd.DataFrame({"team": ["BUF"], "pass_attempts_per_dropback": [0.91]})
    assert np.isclose(attempt_conversion(row, good), 0.91)

    bad = pd.DataFrame({"team": ["BUF"], "pass_attempts_per_dropback": [1.03]})
    with pytest.raises(RuntimeError):
        attempt_conversion(row, bad)

    missing = pd.DataFrame({"team": ["BUF"], "pass_attempts_per_dropback": [np.nan]})
    with pytest.raises(RuntimeError):
        attempt_conversion(row, missing)
