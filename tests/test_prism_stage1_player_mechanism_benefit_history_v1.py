import numpy as np
import pandas as pd

from scripts.research.audit_prism_stage1_player_mechanism_benefit_history_v1 import (
    add_benefit,
    build_prior_state,
    tier,
)


def _row(season, week, pid, y, pb, pc, game):
    return {
        "season": season,
        "week": week,
        "team": "A",
        "opponent": "B",
        "game_key": game,
        "family": "WR",
        "player_id": pid,
        "player_name_key": pid,
        "y": y,
        "n_pos_prior": 100,
        "k_pos_prior": 90,
        "p_baseline": pb,
        "n_player_prior": 3,
        "k_player_prior": 3,
        "p_candidate": pc,
    }


def test_delta_brier_identity():
    x = pd.DataFrame([_row(2024, 4, "p1", 1, 0.5, 0.8, "g1")])
    z = add_benefit(x)
    assert np.isclose(z.loc[0, "baseline_brier"], 0.25)
    assert np.isclose(z.loc[0, "candidate_brier"], 0.04)
    assert np.isclose(z.loc[0, "delta_brier"], 0.21)


def test_current_row_never_enters_own_prior_history():
    x = pd.DataFrame([
        _row(2024, 4, "p1", 1, 0.5, 0.8, "g1"),
        _row(2024, 5, "p1", 0, 0.5, 0.2, "g2"),
        _row(2025, 4, "p1", 1, 0.5, 0.8, "g3"),
    ])
    z, violations = build_prior_state(x)
    assert violations == 0
    z = z.sort_values(["season", "week"]).reset_index(drop=True)
    assert z.loc[0, "prior_mechanism_rows"] == 0
    assert pd.isna(z.loc[0, "prior_mean_delta_brier"])
    assert z.loc[1, "prior_mechanism_rows"] == 1
    assert z.loc[2, "prior_mechanism_rows"] == 2
    assert np.isclose(
        z.loc[2, "prior_mean_delta_brier"],
        z.loc[:1, "delta_brier"].mean(),
    )


def test_support_tiers_fixed():
    assert tier(0) == "TIER0_0"
    assert tier(1) == "TIER1_1_2"
    assert tier(2) == "TIER1_1_2"
    assert tier(3) == "TIER2_3_5"
    assert tier(5) == "TIER2_3_5"
    assert tier(6) == "TIER3_6_9"
    assert tier(9) == "TIER3_6_9"
    assert tier(10) == "TIER4_10_PLUS"
