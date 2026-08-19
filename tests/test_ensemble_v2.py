import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble, fit_market_weights


def test_uncalibrated_ensemble_is_explicit_mc_only_not_equal_weight_guess():
    frame = pd.DataFrame([{"market": "rec_yards", "mc_proj": 80.0, "ml_proj": 50.0, "state_proj": 20.0}])
    out = apply_ensemble(frame, weights=pd.DataFrame())
    row = out.iloc[0]
    assert row.ensemble_proj == 80.0
    assert row.ensemble_status == "uncalibrated_mc_only"
    assert row.ensemble_weight_mc == 1.0
    assert row.ensemble_weight_ml == 0.0
    assert row.ensemble_weight_state == 0.0


def test_calibration_learns_nonnegative_market_specific_weights_that_sum_to_one():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(120):
        mc = 50 + rng.normal(0, 8)
        ml = 50 + rng.normal(0, 8)
        state = 50 + rng.normal(0, 8)
        actual = 0.75 * mc + 0.20 * ml + 0.05 * state + rng.normal(0, 0.5)
        rows.append({"market": "rec_yards", "actual": actual, "mc_proj": mc, "ml_proj": ml, "state_proj": state})
    weights = fit_market_weights(pd.DataFrame(rows), min_rows=40)
    assert len(weights) == 1
    row = weights.iloc[0]
    assert row.mc_weight >= 0 and row.ml_weight >= 0 and row.state_weight >= 0
    assert abs((row.mc_weight + row.ml_weight + row.state_weight) - 1.0) < 1e-9
    assert row.mc_weight > row.ml_weight > row.state_weight


def test_calibrated_projection_renormalizes_when_component_missing():
    weights = pd.DataFrame([{
        "market": "rush_yards", "mc_weight": .5, "ml_weight": .3, "state_weight": .2,
        "calibration_rows": 100, "method": "test_oos",
    }])
    frame = pd.DataFrame([{"market": "rush_yards", "mc_proj": 60.0, "ml_proj": np.nan, "state_proj": 30.0}])
    out = apply_ensemble(frame, weights=weights)
    row = out.iloc[0]
    # .5 and .2 renormalize to 5/7 and 2/7.
    assert abs(row.ensemble_proj - ((5/7)*60 + (2/7)*30)) < 1e-9
    assert abs(row.ensemble_weight_mc + row.ensemble_weight_state - 1.0) < 1e-9
    assert row.ensemble_weight_ml == 0.0
    assert row.ensemble_status == "calibrated"


def test_sportsbook_line_is_not_part_of_weight_fit_or_projection():
    weights = pd.DataFrame([{
        "market": "pass_yards", "mc_weight": .6, "ml_weight": .25, "state_weight": .15,
        "calibration_rows": 80, "method": "test_oos",
    }])
    a = pd.DataFrame([{"market": "pass_yards", "mc_proj": 260.0, "ml_proj": 250.0, "state_proj": 240.0, "line": 200.5}])
    b = a.copy(); b["line"] = 350.5
    pa = apply_ensemble(a, weights=weights).iloc[0].ensemble_proj
    pb = apply_ensemble(b, weights=weights).iloc[0].ensemble_proj
    assert pa == pb
