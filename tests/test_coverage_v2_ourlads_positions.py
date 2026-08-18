import pandas as pd

from scripts.run_coverage_v2 import normalize_ourlads_roles_for_coverage
from scripts.build.build_coverage_v2 import load_wr_universe


def test_ourlads_lwr_swr_rwr_are_semantic_wrs_for_coverage():
    roles = pd.DataFrame(
        {
            "player": ["Alpha Left", "Bravo Slot", "Charlie Right", "Delta Back"],
            "team": ["SF", "SF", "SF", "SF"],
            "position": ["LWR", "SWR", "RWR", "RB"],
            "role": ["WR1", "WR2", "WR3", "RB1"],
        }
    )
    team_map = pd.DataFrame(
        {
            "team": ["SF"],
            "opponent": ["CAR"],
            "season": [2026],
            "week": [1],
            "game_id": ["2026_01_CAR_SF"],
            "game_timestamp": [pd.Timestamp("2026-09-13T17:00:00Z")],
        }
    )

    adapted = normalize_ourlads_roles_for_coverage(roles)
    assert adapted["coverage_source_position"].tolist() == ["LWR", "SWR", "RWR", "RB"]
    assert adapted["position"].tolist() == ["WR", "WR", "WR", "RB"]

    universe = load_wr_universe(team_map, roles=adapted)
    assert set(universe["player"]) == {"Alpha Left", "Bravo Slot", "Charlie Right"}
    assert universe["opponent"].eq("CAR").all()
