import numpy as np
import pandas as pd
import pytest

from scripts.research.run_fair_prob_weight_interaction_v1 import (
    TARGET_MARKETS,
    build_overlay_weights,
    build_published_a1_trace,
)


def test_overlay_only_replaces_frozen_three_markets():
    current = pd.DataFrame([
        {"market": "pass_yards", "mc_weight": .2, "ml_weight": .3, "state_weight": .5},
        {"market": "rush_yards", "mc_weight": .6, "ml_weight": .4, "state_weight": 0.0},
    ])
    heldout = pd.DataFrame([
        {"market": "rec_yards", "mc_weight": .65, "ml_weight": .30, "state_weight": .05},
        {"market": "receptions", "mc_weight": .55, "ml_weight": .45, "state_weight": 0.0},
        {"market": "rush_rec_yards", "mc_weight": .50, "ml_weight": .47, "state_weight": .03},
        {"market": "pass_yards", "mc_weight": .99, "ml_weight": .01, "state_weight": 0.0},
    ])
    out = build_overlay_weights(current, heldout).set_index("market")
    assert set(TARGET_MARKETS).issubset(out.index)
    assert out.loc["pass_yards", "mc_weight"] == pytest.approx(.2)
    assert out.loc["rush_yards", "mc_weight"] == pytest.approx(.6)
    assert out.loc["rec_yards", "mc_weight"] == pytest.approx(.65)


def test_overlay_fails_closed_when_target_weight_missing():
    current = pd.DataFrame([{"market": "pass_yards", "mc_weight": 1.0}])
    heldout = pd.DataFrame([
        {"market": "rec_yards", "mc_weight": 1.0},
        {"market": "receptions", "mc_weight": 1.0},
    ])
    with pytest.raises(RuntimeError, match="missing frozen markets"):
        build_overlay_weights(current, heldout)


def test_published_a1_semantics_do_not_renormalize_missing_state():
    rows = []
    for market in TARGET_MARKETS:
        rows.append({
            "market": market,
            "mc_proj": 10.0,
            "ml_proj": 20.0,
            "state_proj": np.nan,
            "ensemble_proj": 99.0,
            "ensemble_weight_mc": 0.0,
            "ensemble_weight_ml": 0.0,
            "ensemble_weight_state": 0.0,
            "ensemble_calibration_rows": 0,
            "ensemble_method": "old",
            "ensemble_status": "old",
        })
    current = pd.DataFrame(rows)
    heldout = pd.DataFrame([
        {"market": "rec_yards", "mc_weight": .65, "ml_weight": .30, "state_weight": .05, "calibration_rows": 1},
        {"market": "receptions", "mc_weight": .55, "ml_weight": .45, "state_weight": 0.0, "calibration_rows": 1},
        {"market": "rush_rec_yards", "mc_weight": .50, "ml_weight": .47, "state_weight": .03, "calibration_rows": 1},
    ])
    out, audit = build_published_a1_trace(current, heldout)
    got = out.set_index("market")["ensemble_proj"]
    assert got["rec_yards"] == pytest.approx(10*.65 + 20*.30)
    assert got["receptions"] == pytest.approx(10*.55 + 20*.45)
    assert got["rush_rec_yards"] == pytest.approx(10*.50 + 20*.47)
    affected = audit.set_index("market")["rows_affected_by_missing_state_semantics"]
    assert affected["rec_yards"] == 1
    assert affected["receptions"] == 0
    assert affected["rush_rec_yards"] == 1
