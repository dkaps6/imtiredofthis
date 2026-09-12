import pandas as pd
import pytest

from scripts.research.run_fair_prob_weight_interaction_v1 import (
    TARGET_MARKETS,
    build_overlay_weights,
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
