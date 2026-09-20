import pandas as pd

from scripts.football_context.qualify_bdb2023_exact_blocker_rusher_assignment_v1 import (
    strict_prior_support,
)


def test_strict_prior_support_excludes_same_week_edges():
    edges = pd.DataFrame([
        {
            "source_season": 2021,
            "week": 1,
            "game_id": 1,
            "play_id": 1,
            "blocker_nfl_id": 100,
            "blocked_nfl_id": 200,
            "block_interaction_role": "Pass Block",
            "blocked_defender_source_role": "Pass Rush",
        },
        {
            "source_season": 2021,
            "week": 1,
            "game_id": 1,
            "play_id": 2,
            "blocker_nfl_id": 100,
            "blocked_nfl_id": 200,
            "block_interaction_role": "Pass Block",
            "blocked_defender_source_role": "Pass Rush",
        },
        {
            "source_season": 2021,
            "week": 2,
            "game_id": 2,
            "play_id": 3,
            "blocker_nfl_id": 100,
            "blocked_nfl_id": 200,
            "block_interaction_role": "Pass Block",
            "blocked_defender_source_role": "Pass Rush",
        },
        {
            "source_season": 2021,
            "week": 2,
            "game_id": 2,
            "play_id": 4,
            "blocker_nfl_id": 100,
            "blocked_nfl_id": 201,
            "block_interaction_role": "Pass Block",
            "blocked_defender_source_role": "Pass Rush",
        },
    ])
    out = strict_prior_support(edges)
    w1 = out.loc[out["week"].eq(1)]
    w2 = out.loc[out["week"].eq(2)].sort_values("play_id")

    assert w1["prior_blocker_edges"].eq(0).all()
    assert w1["prior_defender_edges"].eq(0).all()
    assert w1["prior_pair_edges"].eq(0).all()
    assert list(w2["prior_blocker_edges"]) == [2, 2]
    assert list(w2["prior_defender_edges"]) == [2, 0]
    assert list(w2["prior_pair_edges"]) == [2, 0]


def test_id_normalization_handles_float_serialization():
    edges = pd.DataFrame([
        {
            "source_season": 2021,
            "week": 1,
            "game_id": 1,
            "play_id": 1,
            "blocker_nfl_id": 100.0,
            "blocked_nfl_id": 200.0,
            "block_interaction_role": "Pass Block",
            "blocked_defender_source_role": "Pass Rush",
        }
    ])
    out = strict_prior_support(edges)
    assert out.iloc[0]["blocker_nfl_id"] == "100"
    assert out.iloc[0]["blocked_nfl_id"] == "200"
