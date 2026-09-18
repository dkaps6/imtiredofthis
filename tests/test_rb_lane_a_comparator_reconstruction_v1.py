import pandas as pd
import pytest

from scripts.backtest.rb_lane_a_comparator_reconstruction_v1 import (
    FROZEN_PARENT_BLOBS,
    MECHANISM_COMPARATOR_COLUMN,
    build_dual_market_promotion_comparator,
    build_input_manifest,
    build_promotion_comparator,
    compare_component_predictions_parity,
    compare_mechanism_comparator_parity,
    load_rotation_market_weights,
    load_rotation_rush_yards_weights,
    same_job_double_build_disposition,
    sha256_of_file,
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


def _rows(rows):
    cols = ["season", "week", "team", "player_clean_key", "market", "mc_proj", "ml_proj", "state_proj"]
    return pd.DataFrame(rows, columns=cols)


def test_parity_passes_on_identical_frames():
    frame = _rows(
        [
            [2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0],
            [2024, 1, "SF", "p2", "rush_yards", 60.0, 55.0, 65.0],
        ]
    )
    result = compare_component_predictions_parity(frame, frame.copy())
    assert result["disposition"] == "PASS"
    assert result["rows_fresh_only"] == 0
    assert result["rows_canonical_only"] == 0
    assert result["max_abs_value_delta"]["mc_proj"] == 0.0


def test_parity_fails_closed_on_value_mismatch():
    fresh = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    canonical = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.5, 75.0, 85.0]])
    result = compare_component_predictions_parity(fresh, canonical)
    assert result["disposition"] == "PARITY_FAILURE"
    assert result["max_abs_value_delta"]["mc_proj"] == pytest.approx(0.5)


def test_parity_fails_closed_on_unmatched_rows():
    fresh = _rows(
        [
            [2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0],
            [2024, 1, "SF", "p2", "rush_yards", 60.0, 55.0, 65.0],
        ]
    )
    canonical = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    result = compare_component_predictions_parity(fresh, canonical)
    assert result["disposition"] == "PARITY_FAILURE"
    assert result["rows_fresh_only"] == 1


def test_parity_fails_closed_on_missing_columns():
    fresh = pd.DataFrame({"season": [2024]})
    canonical = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    result = compare_component_predictions_parity(fresh, canonical)
    assert result["disposition"] == "PARITY_FAILURE"
    assert "fresh frame missing" in result["reason"]


def test_same_job_double_build_passes_on_identical_builds():
    frame = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    result = same_job_double_build_disposition(frame, frame.copy())
    assert result["disposition"] == "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS"


def test_same_job_double_build_fails_closed_on_value_mismatch():
    build_a = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    build_b = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.000002, 75.0, 85.0]])
    result = same_job_double_build_disposition(build_a, build_b)
    assert result["disposition"] == "SAME_JOB_AUTHORITY_RECONSTRUCTION_FAILURE"


def test_same_job_double_build_fails_closed_on_row_mismatch():
    build_a = _rows(
        [
            [2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0],
            [2024, 1, "SF", "p2", "rush_yards", 60.0, 55.0, 65.0],
        ]
    )
    build_b = _rows([[2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0]])
    result = same_job_double_build_disposition(build_a, build_b)
    assert result["disposition"] == "SAME_JOB_AUTHORITY_RECONSTRUCTION_FAILURE"
    assert result["rows_fresh_only"] == 1


def test_sha256_of_file_matches_known_content(tmp_path):
    import hashlib

    content = b"season,week\n2024,1\n"
    path = tmp_path / "sample.csv"
    path.write_bytes(content)
    assert sha256_of_file(path) == hashlib.sha256(content).hexdigest()


def test_build_input_manifest_records_sha256_per_label(tmp_path):
    a = tmp_path / "a.csv"
    a.write_bytes(b"a")
    b = tmp_path / "b.csv"
    b.write_bytes(b"b")
    manifest = build_input_manifest({"schedule": a, "injuries": b})
    assert set(manifest) == {"schedule", "injuries"}
    assert manifest["schedule"]["sha256"] != manifest["injuries"]["sha256"]


def test_build_input_manifest_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="missing file"):
        build_input_manifest({"schedule": tmp_path / "does_not_exist.csv"})


def _mech_rows(rows):
    cols = ["season", "week", "team", "name_key", MECHANISM_COMPARATOR_COLUMN]
    return pd.DataFrame(rows, columns=cols)


def test_mechanism_parity_passes_on_identical_frames():
    frame = _mech_rows(
        [
            [2025, 1, "KC", "p1", 80.0],
            [2025, 1, "SF", "p2", 60.0],
        ]
    )
    result = compare_mechanism_comparator_parity(frame, frame.copy())
    assert result["disposition"] == "MECHANISM_AUTHORITY_RECONSTRUCTION_PASS"
    assert result["rows_fresh_only"] == 0
    assert result["rows_canonical_only"] == 0
    assert result["max_abs_value_delta"][MECHANISM_COMPARATOR_COLUMN] == 0.0


def test_mechanism_parity_fails_closed_on_value_mismatch():
    fresh = _mech_rows([[2025, 1, "KC", "p1", 80.0]])
    canonical = _mech_rows([[2025, 1, "KC", "p1", 80.5]])
    result = compare_mechanism_comparator_parity(fresh, canonical)
    assert result["disposition"] == "MECHANISM_PARITY_FAILURE"
    assert result["max_abs_value_delta"][MECHANISM_COMPARATOR_COLUMN] == pytest.approx(0.5)


def test_mechanism_parity_fails_closed_on_unmatched_rows():
    fresh = _mech_rows(
        [
            [2025, 1, "KC", "p1", 80.0],
            [2025, 1, "SF", "p2", 60.0],
        ]
    )
    canonical = _mech_rows([[2025, 1, "KC", "p1", 80.0]])
    result = compare_mechanism_comparator_parity(fresh, canonical)
    assert result["disposition"] == "MECHANISM_PARITY_FAILURE"
    assert result["rows_fresh_only"] == 1


def test_mechanism_parity_fails_closed_on_missing_columns():
    fresh = pd.DataFrame({"season": [2025]})
    canonical = _mech_rows([[2025, 1, "KC", "p1", 80.0]])
    result = compare_mechanism_comparator_parity(fresh, canonical)
    assert result["disposition"] == "MECHANISM_PARITY_FAILURE"
    assert "fresh frame missing" in result["reason"]


def test_load_rotation_market_weights_rush_att_rotation_1():
    w = load_rotation_market_weights(1, "rush_att")
    assert len(w) == 1
    assert w.iloc[0]["market"] == "rush_att"
    assert w.iloc[0]["mc_weight"] == pytest.approx(0.260264, abs=1e-6)


def test_load_rotation_market_weights_rush_att_rotation_2():
    w = load_rotation_market_weights(2, "rush_att")
    assert len(w) == 1
    assert w.iloc[0]["mc_weight"] == pytest.approx(0.3164919683016017, abs=1e-9)


def _dual_cp(rows):
    cols = ["season", "week", "team", "player_clean_key", "market", "mc_proj", "ml_proj", "state_proj"]
    return pd.DataFrame(rows, columns=cols)


def test_build_dual_market_promotion_comparator_joins_both_markets():
    cp = _dual_cp(
        [
            [2024, 1, "KC", "p1", "rush_att", 15.0, 14.0, 16.0],
            [2024, 1, "KC", "p1", "rush_yards", 80.0, 75.0, 85.0],
        ]
    )
    out = build_dual_market_promotion_comparator(cp, rotation=1)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["promotion_rush_att"] > 0
    assert row["promotion_rush_yards"] > 0


def test_build_dual_market_promotion_comparator_fails_closed_on_missing_market():
    cp = _dual_cp([[2024, 1, "KC", "p1", "rush_att", 15.0, 14.0, 16.0]])
    with pytest.raises(RuntimeError, match="rush_yards reconstruction is empty"):
        build_dual_market_promotion_comparator(cp, rotation=1)

