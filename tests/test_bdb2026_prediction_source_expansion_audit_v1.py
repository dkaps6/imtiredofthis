from pathlib import Path

import pandas as pd

from scripts.football_context.audit_bdb2026_prediction_source_expansion_v1 import compare


def _frame(rows):
    return pd.DataFrame(rows)


def test_prediction_same_2023_inputs_is_not_material_expansion():
    analytics = _frame([
        {
            "historical_regular_season_file": True,
            "kind": "input",
            "basename": "input_2023_w01.csv",
            "sha256": "same",
            "season": 2023,
            "columns": "game_id|play_id|nfl_id|frame_id|player_role|x|y|s|a|o|dir",
        }
    ])
    prediction = analytics.copy()
    out = compare(analytics, prediction)
    assert out["materially_expands_certified_analytics_history"] is False
    assert out["disposition"] == "NO_MATERIAL_HISTORICAL_TRACKING_EXPANSION"
    assert out["byte_identical_common_historical_input_files"] == 1
    assert out["prediction_has_required_tracking_schema"] is True


def test_prediction_2024_input_is_material_expansion():
    analytics = _frame([
        {
            "historical_regular_season_file": True,
            "kind": "input",
            "basename": "input_2023_w01.csv",
            "sha256": "a",
            "season": 2023,
            "columns": "game_id|play_id|nfl_id|frame_id|player_role|x|y|s|a|o|dir",
        }
    ])
    prediction = _frame([
        {
            "historical_regular_season_file": True,
            "kind": "input",
            "basename": "input_2023_w01.csv",
            "sha256": "a",
            "season": 2023,
            "columns": "game_id|play_id|nfl_id|frame_id|player_role|x|y|s|a|o|dir",
        },
        {
            "historical_regular_season_file": True,
            "kind": "input",
            "basename": "input_2024_w01.csv",
            "sha256": "b",
            "season": 2024,
            "columns": "game_id|play_id|nfl_id|frame_id|player_role|x|y|s|a|o|dir",
        },
    ])
    out = compare(analytics, prediction)
    assert out["materially_expands_certified_analytics_history"] is True
    assert out["prediction_additional_historical_seasons_vs_certified_2023"] == [2024]
    assert out["disposition"] == "MATERIAL_HISTORICAL_TRACKING_EXPANSION_AVAILABLE"


def test_mock_test_files_cannot_create_historical_expansion():
    analytics = _frame([
        {
            "historical_regular_season_file": True,
            "kind": "input",
            "basename": "input_2023_w01.csv",
            "sha256": "a",
            "season": 2023,
            "columns": "game_id|play_id|nfl_id|frame_id|player_role|x|y|s|a|o|dir",
        }
    ])
    prediction = pd.concat(
        [
            analytics.copy(),
            _frame([
                {
                    "historical_regular_season_file": False,
                    "kind": "",
                    "basename": "test_input.csv",
                    "sha256": "x",
                    "season": None,
                    "columns": "game_id|play_id|nfl_id|frame_id",
                }
            ]),
        ],
        ignore_index=True,
    )
    out = compare(analytics, prediction)
    assert out["materially_expands_certified_analytics_history"] is False
