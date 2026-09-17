import pandas as pd
import pytest

from scripts.data_frontier.advanced_feature_wr_integration_preflight_v1 import (
    clean_name,
    height_inches,
    normalize_position,
)
from scripts.data_frontier.advanced_feature_materializer_common_v1 import assert_strict_prior


def test_identity_normalizers_are_deterministic():
    assert clean_name("Amon-Ra St. Brown Jr.") == "amonrastbrown"
    assert height_inches("6-2") == 74.0
    assert height_inches("74") == 74.0
    assert normalize_position("HB") == "RB"


def test_integration_preflight_retains_strict_prior_gate():
    good = pd.DataFrame({"target_week": [5, 9], "history_max_source_week": [4, 8]})
    assert assert_strict_prior(good)["violations"] == 0

    bad = pd.DataFrame({"target_week": [5], "history_max_source_week": [5]})
    with pytest.raises(ValueError, match="temporal leakage"):
        assert_strict_prior(bad)
