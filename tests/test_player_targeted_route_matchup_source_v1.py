import pandas as pd

from scripts.research.audit_player_targeted_route_matchup_source_v1 import (
    ROUTE_FAMILIES,
    join_frames,
    route_norm,
)


def test_route_norm_preserves_documented_taxonomy():
    s = pd.Series(["HITCH/CURL", "hitch curl", "SHALLOW CROSS/DRAG", "texas-angle"])
    out = route_norm(s).tolist()
    assert out == ["HITCH/CURL", "HITCH/CURL", "SHALLOW CROSS/DRAG", "TEXAS/ANGLE"]
    assert all(x in ROUTE_FAMILIES for x in out)


def test_join_attributes_route_only_to_pbp_target_receiver():
    participation = pd.DataFrame(
        {
            "nflverse_game_id": ["2024_01_AAA_BBB"],
            "play_id": [10],
            "route": ["SLANT"],
        }
    )
    pbp = pd.DataFrame(
        {
            "game_id": ["2024_01_AAA_BBB"],
            "play_id": [10],
            "season": [2024],
            "week": [1],
            "posteam": ["AAA"],
            "defteam": ["BBB"],
            "receiver_player_id": ["00-001"],
            "receiver_player_name": ["Player A"],
            "season_type": ["REG"],
        }
    )
    rosters = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "gsis_id": ["00-001"],
            "position": ["WR"],
        }
    )

    rows, audit = join_frames(participation, pbp, rosters)
    assert audit["join_rate"] == 1.0
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["route_family"] == "SLANT"
    assert row["receiver_id"] == "00-001"
    assert row["position"] == "WR"
    assert bool(row["valid_receiver_identity"])
    assert bool(row["valid_route_taxonomy"])
    assert bool(row["valid_wr_te"])
