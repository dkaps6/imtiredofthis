from __future__ import annotations

import pandas as pd

from scripts.research.rush_att_zero_mc_allocation_lineage_v1 import (
    _classify,
    _selected_authority,
)


def test_selected_authority_reproduces_literal_top_five_and_stable_rank():
    rows = []
    shares = [
        ("Alpha", "a", 0.30),
        ("Bravo", "b", 0.25),
        ("Charlie", "c", 0.20),
        ("Delta", "d", 0.10),
        ("Echo", "e", 0.08),
        ("Foxtrot", "f", 0.07),
    ]
    for player, key, share in shares:
        rows.append(
            {
                "event_id": "ATL|CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player": player,
                "player_clean_key": key,
                "market": "rush_att",
                "rules_rush_share": share,
            }
        )
    got = _selected_authority(pd.DataFrame(rows)).set_index("player_clean_key")

    assert list(got.sort_values("sim_selected_share_rank").index) == ["a", "b", "c", "d", "e", "f"]
    assert got.loc["a", "sim_selected_top5_member"] == 1
    assert got.loc["e", "sim_selected_top5_member"] == 1
    assert got.loc["f", "sim_selected_top5_member"] == 0
    assert got.loc["f", "sim_post_top5_share"] == 0.0
    assert got.loc["e", "sim_post_top5_share"] == 0.08


def test_selected_authority_matches_keep_last_player_row_semantics():
    frame = pd.DataFrame(
        [
            {
                "event_id": "ATL|CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player": "Alpha",
                "player_clean_key": "alpha",
                "market": "rush_att",
                "rules_rush_share": 0.25,
            },
            {
                "event_id": "ATL|CAR",
                "team": "ATL",
                "opponent": "CAR",
                "player": "Alpha",
                "player_clean_key": "alpha",
                "market": "rush_yards",
                "rules_rush_share": 0.20,
            },
        ]
    )
    got = _selected_authority(frame)
    assert len(got) == 1
    assert got.iloc[0]["sim_selected_market"] == "rush_yards"
    assert got.iloc[0]["sim_selected_rules_rush_share"] == 0.20


def test_first_zero_stage_classifies_top_five_exclusion():
    row = pd.Series(
        {
            "rush_att_row_rules_rush_share": 0.10,
            "sim_selected_rules_rush_share": 0.10,
            "sim_selected_top5_member": 0,
            "final_player_probability": 0.0,
            "realized_multinomial_mean_carries": 0.0,
            "keyed_lookup_mean": 0.0,
            "canonical_mc_proj": 0.0,
        }
    )
    assert _classify(row) == "TOP5_EXCLUDED"


def test_first_zero_stage_classifies_sampling_zero_after_positive_probability():
    row = pd.Series(
        {
            "rush_att_row_rules_rush_share": 0.10,
            "sim_selected_rules_rush_share": 0.10,
            "sim_selected_top5_member": 1,
            "final_player_probability": 0.01,
            "realized_multinomial_mean_carries": 0.0,
            "keyed_lookup_mean": 0.0,
            "canonical_mc_proj": 0.0,
        }
    )
    assert _classify(row) == "REALIZED_MC_ZERO_WITH_POSITIVE_PROBABILITY"
