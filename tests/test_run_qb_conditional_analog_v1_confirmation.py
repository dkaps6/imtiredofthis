from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.research import run_qb_conditional_analog_v1_confirmation as m


def _fixture(tmp_path, n24=80, n25=50):
    rng = np.random.default_rng(123)
    vegas_rows = []
    feature_rows = []
    for season, n in [(2024, n24), (2025, n25)]:
        for i in range(n):
            key = {
                "season": season,
                "week": (i % 18) + 1,
                "team": f"T{i % 32:02d}",
                "player_clean_key": f"p_{season}_{i}",
            }
            vegas_rows.append({
                **key,
                "benchmark_arm": m.ARM,
                "position": m.POSITION,
                "market": m.MARKET,
                "unit_result": 0.8 if i % 3 else -1.0,
            })
            feature_rows.append({
                **key,
                **{c: float(rng.normal()) for c in m.FEATURE_COLUMNS},
            })
    vegas = tmp_path / "vegas.csv"
    features = tmp_path / "features.csv"
    pd.DataFrame(vegas_rows).to_csv(vegas, index=False)
    pd.DataFrame(feature_rows).to_csv(features, index=False)
    return vegas, features


def test_prepare_is_outcome_blind_and_freezes_expected_geometry(tmp_path):
    vegas, features = _fixture(tmp_path)
    out = tmp_path / "prepared"
    result = m.prepare(vegas, features, out)
    assert result["phase"] == "PREPARED_OUTCOME_BLIND"
    assert result["outcome_values_read"] is False
    assert result["feature_count"] == 17
    assert result["k"] == 15
    assert result["scope_rows"] == 130
    assert result["joined_rows"] == 130
    assert result["unmatched_rows"] == 0
    assert result["missing_feature_rows"] == 0
    assert result["reference_2024_rows"] == 80
    assert result["evaluation_2025_rows"] == 50
    assert (out / "reference_neighbor_indices.npy").exists()
    assert (out / "evaluation_neighbor_indices.npy").exists()


def test_prepare_excludes_missing_features_without_imputation(tmp_path):
    vegas, features = _fixture(tmp_path)
    f = pd.read_csv(features)
    f.loc[0, "qb_prior_attempts"] = np.nan
    f.to_csv(features, index=False)
    out = tmp_path / "prepared"
    result = m.prepare(vegas, features, out)
    assert result["missing_feature_rows"] == 1
    assert result["clean_rows"] == 129
    excluded = pd.read_csv(out / "excluded_missing_feature_rows.csv")
    assert len(excluded) == 1


def test_confirm_fails_closed_or_passes_once_without_rescue(tmp_path):
    vegas, features = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    confirmed = tmp_path / "confirmed"
    m.prepare(vegas, features, prepared)
    result = m.confirm(vegas, prepared, confirmed)
    assert result["disposition"] in {m.SUPPORTED, m.NOT_ACTIONABLE}
    assert result["rescue_authorized"] is False
    assert result["production_mutation_authorized"] is False
    assert set(result["gates"]) == {
        "n_2025_ge_40",
        "roi_2025_positive",
        "roi_2025_beats_unconditional_baseline",
        "roi_2024_supported_positive",
    }
    saved = json.loads((confirmed / "QB_CONDITIONAL_ANALOG_V1_RESULT.json").read_text())
    assert saved["disposition"] == result["disposition"]


def test_confirm_rejects_vegas_hash_drift(tmp_path):
    vegas, features = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    m.prepare(vegas, features, prepared)
    raw = pd.read_csv(vegas)
    raw.loc[0, "unit_result"] = -0.75
    raw.to_csv(vegas, index=False)
    with pytest.raises(RuntimeError, match="hash drifted"):
        m.confirm(vegas, prepared, tmp_path / "confirmed")


def test_failed_architecture_never_emits_supported_final_class(tmp_path):
    vegas, features = _fixture(tmp_path, n24=80, n25=20)
    prepared = tmp_path / "prepared"
    confirmed = tmp_path / "confirmed"
    m.prepare(vegas, features, prepared)
    result = m.confirm(vegas, prepared, confirmed)
    assert result["disposition"] == m.NOT_ACTIONABLE
    rows = pd.read_csv(confirmed / "evaluation_2025_confirmed.csv")
    assert "SUPPORTED" not in set(rows["final_evidence_class"])
