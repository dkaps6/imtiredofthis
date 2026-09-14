from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.research.wr1_yardage_decomposition_v1 import shap, wb


def test_week_bucket_boundaries():
    assert wb(1) == "W1-4"
    assert wb(4) == "W1-4"
    assert wb(5) == "W5-9"
    assert wb(9) == "W5-9"
    assert wb(10) == "W10-13"
    assert wb(13) == "W10-13"
    assert wb(14) == "W14-18"
    assert wb(18) == "W14-18"


def test_three_factor_shapley_sums_to_yard_error():
    row = pd.Series({
        "at": 8.0,
        "acr": 0.625,
        "aypr": 14.0,
        "pt": 7.0,
        "pcr": 0.70,
        "pypr": 12.0,
    })
    contrib = shap(row)
    actual = row["at"] * row["acr"] * row["aypr"]
    pred = row["pt"] * row["pcr"] * row["pypr"]
    assert np.isclose(sum(contrib), pred - actual)
