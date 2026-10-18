import pandas as pd

from scripts.research.audit_prism_stage2_player_mechanism_benefit_history_v1 import (
    build_strict_prior_state,
    support_tier,
)


def test_support_tiers_are_fixed():
    assert support_tier(0) == "TIER0_0"
    assert support_tier(1) == "TIER1_1_2"
    assert support_tier(2) == "TIER1_1_2"
    assert support_tier(3) == "TIER2_3_5"
    assert support_tier(5) == "TIER2_3_5"
    assert support_tier(6) == "TIER3_6_9"
    assert support_tier(9) == "TIER3_6_9"
    assert support_tier(10) == "TIER4_10_PLUS"


def test_current_row_never_enters_its_own_prior_state():
    x = pd.DataFrame(
        [
            {"season": 2024, "week": 5, "game_key": "g1", "team": "A", "cell": "RB_RECEIVING",
             "family": "RB", "player_id": "p1", "player_name_key": "p1", "stage1_p_active": 0.9,
             "authority_q": 0.5, "delta_crps": 1.0},
            {"season": 2024, "week": 6, "game_key": "g2", "team": "A", "cell": "RB_RECEIVING",
             "family": "RB", "player_id": "p1", "player_name_key": "p1", "stage1_p_active": 0.9,
             "authority_q": 0.5, "delta_crps": -1.0},
            {"season": 2025, "week": 5, "game_key": "g3", "team": "B", "cell": "RB_RECEIVING",
             "family": "RB", "player_id": "p1", "player_name_key": "p1", "stage1_p_active": 0.9,
             "authority_q": 0.4, "delta_crps": 7.0},
        ]
    )
    z, violations = build_strict_prior_state(x)
    assert violations == 0
    z = z.sort_values(["season", "week"]).reset_index(drop=True)
    assert z.loc[0, "prior_mechanism_rows"] == 0
    assert pd.isna(z.loc[0, "prior_mean_delta_crps"])
    assert z.loc[1, "prior_mechanism_rows"] == 1
    assert z.loc[1, "prior_mean_delta_crps"] == 1.0
    assert z.loc[2, "prior_mechanism_rows"] == 2
    assert z.loc[2, "prior_mean_delta_crps"] == 0.0


def test_authority_rank_is_pregame_only_and_deterministic():
    x = pd.DataFrame(
        [
            {"season": 2025, "week": 5, "game_key": "g", "team": "A", "cell": "TE_RECEIVING",
             "family": "TE", "player_id": "p1", "player_name_key": "p1", "stage1_p_active": 0.9,
             "authority_q": 0.6, "delta_crps": 0.1},
            {"season": 2025, "week": 5, "game_key": "g", "team": "A", "cell": "TE_RECEIVING",
             "family": "TE", "player_id": "p2", "player_name_key": "p2", "stage1_p_active": 0.9,
             "authority_q": 0.3, "delta_crps": 0.2},
            {"season": 2025, "week": 5, "game_key": "g", "team": "A", "cell": "TE_RECEIVING",
             "family": "TE", "player_id": "p3", "player_name_key": "p3", "stage1_p_active": 0.9,
             "authority_q": 0.0, "delta_crps": 0.0},
        ]
    )
    z, violations = build_strict_prior_state(x)
    assert violations == 0
    ranks = dict(zip(z["player_id"], z["authority_rank_state"]))
    assert ranks == {"p1": "RANK1", "p2": "RANK2", "p3": "ZERO"}
