import pandas as pd

from scripts.football_context.qualify_bdb2025_exact_blocker_rusher_assignment_v1 import (
    add_strict_prior_support,
    explode_edges,
)


def test_explode_edges_preserves_exact_assignments():
    games = pd.DataFrame([
        {"gameId": 1, "season": 2022, "week": 1},
    ])
    pp = pd.DataFrame([
        {
            "gameId": 1,
            "playId": 10,
            "nflId": 100,
            "blockedPlayerNFLId1": 200,
            "blockedPlayerNFLId2": 201,
            "blockedPlayerNFLId3": None,
        }
    ])
    edges, audit = explode_edges(pp, games)
    assert audit["missing_game_week_rows"] == 0
    assert len(edges) == 2
    assert set(edges["rusher_nfl_id"]) == {"200", "201"}


def test_strict_prior_support_never_uses_same_week():
    edges = pd.DataFrame([
        {"season": 2022, "week": 1, "gameId": 1, "playId": 1, "blocker_nfl_id": "100", "rusher_nfl_id": "200"},
        {"season": 2022, "week": 1, "gameId": 1, "playId": 2, "blocker_nfl_id": "100", "rusher_nfl_id": "200"},
        {"season": 2022, "week": 2, "gameId": 2, "playId": 3, "blocker_nfl_id": "100", "rusher_nfl_id": "200"},
        {"season": 2022, "week": 2, "gameId": 2, "playId": 4, "blocker_nfl_id": "100", "rusher_nfl_id": "201"},
    ])
    out = add_strict_prior_support(edges)
    w1 = out.loc[out["week"].eq(1)]
    w2 = out.loc[out["week"].eq(2)].sort_values("playId")
    assert w1["prior_blocker_assignment_edges"].eq(0).all()
    assert w1["prior_rusher_assignment_edges"].eq(0).all()
    assert list(w2["prior_blocker_assignment_edges"]) == [2, 2]
    assert list(w2["prior_rusher_assignment_edges"]) == [2, 0]
    assert list(w2["prior_pair_assignment_edges"]) == [2, 0]


def test_duplicate_exact_edge_is_deduplicated_and_reported():
    games = pd.DataFrame([{"gameId": 1, "season": 2022, "week": 1}])
    pp = pd.DataFrame([
        {
            "gameId": 1,
            "playId": 10,
            "nflId": 100,
            "blockedPlayerNFLId1": 200,
            "blockedPlayerNFLId2": 200,
            "blockedPlayerNFLId3": None,
        }
    ])
    edges, audit = explode_edges(pp, games)
    assert len(edges) == 1
    assert audit["duplicate_exact_edges"] == 2
