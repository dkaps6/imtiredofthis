import pandas as pd
import pytest

from scripts.backtest.rb_lane_a_comparator_reconstruction_v1 import (
    FROZEN_PARENT_BLOBS,
    build_promotion_comparator,
    load_rotation_rush_yards_weights,
    verify_frozen_parent_blobs,
)


def test_frozen_parent_blobs_match_pinned_shas():
    result = verify_frozen_parent_blobs()
    assert result["disposition"] == "PASS"
    assert result["mismatches"] == []
    assert set(result["files"]) == set(FROZEN_PARENT_BLOBS)


def test_rotation_1_uses_pre_2024_weights_not_production_row():
    w1 = load_rotation_rush_yards_weights(1)
    assert len(w1) == 1
    assert w1.iloc[0]["market"] == "rush_yards"
    # The already-merged 2023-only frozen fit (PR #545), not the production
    # 2024-fit row -- this is the Amendment-3 fatal-leak fix.
    assert w1.iloc[0]["mc_weight"] == pytest.approx(0.396067, abs=1e-6)
    assert w1.iloc[0]["ml_weight"] == pytest.approx(0.559625, abs=1e-6)


def test_rotation_2_uses_production_weights():
    w2 = load_rotation_rush_yards_weights(2)
    assert len(w2) == 1
    # data/model_ensemble_weights.csv's rush_yards row, legitimately prior to 2025.
    assert w2.iloc[0]["mc_weight"] == pytest.approx(0.5569542426070742, abs=1e-9)
    assert w2.iloc[0]["state_weight"] == pytest.approx(0.0, abs=1e-9)


def test_unknown_rotation_raises():
    with pytest.raises(RuntimeError, match="unknown rotation"):
        load_rotation_rush_yards_weights(3)


def test_build_promotion_comparator_filters_to_rush_yards_only():
    cp = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "market": ["rush_yards", "rec_yards"],
            "mc_proj": [80.0, 40.0],
            "ml_proj": [75.0, 42.0],
            "state_proj": [85.0, 38.0],
        }
    )
    out = build_promotion_comparator(cp, rotation=1)
    assert list(out["market"].unique()) == ["rush_yards"]
    assert "promotion_comparator_rush_yards" in out.columns
    assert out["promotion_comparator_rush_yards"].notna().all()


def test_build_promotion_comparator_rotation_gives_different_result_than_rotation_2():
    cp = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "team": ["KC"],
            "market": ["rush_yards"],
            "mc_proj": [80.0],
            "ml_proj": [60.0],
            "state_proj": [90.0],
        }
    )
    r1 = build_promotion_comparator(cp, rotation=1)["promotion_comparator_rush_yards"].iloc[0]
    r2 = build_promotion_comparator(cp, rotation=2)["promotion_comparator_rush_yards"].iloc[0]
    assert r1 != pytest.approx(r2)
