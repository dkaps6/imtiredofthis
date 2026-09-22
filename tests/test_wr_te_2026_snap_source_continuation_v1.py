import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling import te_r5p_entitlement_adapter_v1 as te


def test_snap_source_activation_boundary():
    legacy = [2020, 2021, 2022, 2023, 2024, 2025]
    assert te._snap_source_seasons(2025, 18) == legacy
    assert te._snap_source_seasons(2026, 1) == legacy
    assert te._snap_source_seasons(2026, 2) == legacy
    assert te._snap_source_seasons(2026, 3) == legacy + [2026]
    assert te._snap_source_seasons(2026, 18) == legacy + [2026]
    with pytest.raises(RuntimeError, match="not certified"):
        te._snap_source_seasons(2027, 1)


def test_target_frame_must_have_one_season_and_week():
    good = pd.DataFrame({"season": [2026, 2026], "week": [3, 3]})
    assert te._target_season_week(good, "test") == (2026, 3)

    with pytest.raises(RuntimeError, match="one target season/week"):
        te._target_season_week(
            pd.DataFrame({"season": [2026, 2026], "week": [2, 3]}), "test"
        )


def test_strict_prior_still_excludes_target_week_and_future_2026_rows():
    target = pd.DataFrame([{
        "player_clean_key": "examplewr",
        "team": "CHI",
        "season": 2026,
        "week": 3,
    }])
    snaps = pd.DataFrame([
        {
            "player_key": "examplewr", "team": "CHI", "season": 2026, "week": 1,
            "ordinal": 202601, "offense_pct": 0.40, "offense_snaps": 25,
        },
        {
            "player_key": "examplewr", "team": "CHI", "season": 2026, "week": 2,
            "ordinal": 202602, "offense_pct": 0.70, "offense_snaps": 45,
        },
        {
            "player_key": "examplewr", "team": "CHI", "season": 2026, "week": 3,
            "ordinal": 202603, "offense_pct": 0.99, "offense_snaps": 70,
        },
    ])
    got = te._strict_prior_features(target, snaps).iloc[0]
    assert got["prior_count_anyteam"] == 2
    assert got["prior_count_same_team"] == 2
    assert got["prior1_same_team_offense_pct"] == pytest.approx(0.70)
    assert got["prior1_anyteam_offense_pct"] == pytest.approx(0.70)


def test_frozen_model_jsons_are_unchanged():
    expected = {
        Path("data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json"):
            "a47730312a6d7ea8de2c4034ddd2be5ebbac72c67647dfbfa7b1985305e1249b",
        Path("data/models/wr_r15_production_model_v1/wr_r15_production_model_v1.json"):
            "ac1058f534c7923e8ad41e52a92e7001c1de309bb04874d76ae60a83e3a1b2ff",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
