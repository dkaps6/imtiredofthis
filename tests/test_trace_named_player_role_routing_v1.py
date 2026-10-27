"""Pin the named-player role-routing findings against production source."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.research.trace_named_player_role_routing_v1 import (
    ROUTING,
    assert_routing_matches_source,
    build,
    eligible_gates,
)

UNIV = Path("data/research/week7_opportunity_trace/"
            "week7_trace__data__football_simulation_universe.csv")


def test_routing_matches_production_source():
    c = assert_routing_matches_source()
    assert c["wr_keyed_on_role_label"] is True
    assert c["te_rb_keyed_on_position"] is True
    assert c["only_two_perimeter_wrs_labelled"] is True


@pytest.mark.parametrize("role,position,expected", [
    ("WR1", "WR", "wr1_target_mult"),
    ("WR1_5", "WR", "wr1_5_target_mult"),
    ("SLOT", "SWR", "slot_target_mult"),
    # The finding: an unlabelled WR falls through the whole chain.
    ("", "WR", ""),
    (None, "WR", ""),
    # TE and RB are routed by POSITION, so they never need a label.
    ("", "TE", "te_target_mult"),
    ("", "RB", "rb_rec_target_mult"),
    ("", "FB", "rb_rec_target_mult"),
])
def test_unlabelled_wr_is_the_only_skill_position_with_no_channel(
        role, position, expected):
    assert eligible_gates(role, position)[0] == expected


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_kupp_is_structurally_excluded_while_a_third_te_is_not():
    out, status = build(pd.read_csv(UNIV, low_memory=False))
    by = out.set_index("player")
    assert by.loc["Cooper Kupp", "canonical_role_label"] == "NONE_ASSIGNED"
    assert by.loc["Cooper Kupp",
                  "receives_any_opponent_target_channel"] is False or \
        not bool(by.loc["Cooper Kupp", "receives_any_opponent_target_channel"])
    # the asymmetry, stated as a test
    assert bool(by.loc["Eric Saubert", "receives_any_opponent_target_channel"])
    assert bool(by.loc["Brady Russell", "receives_any_opponent_target_channel"])
    assert by.loc["Jaxon Smith-Njigba", "canonical_role_label"] == "WR1"
    assert by.loc["AJ Barner", "routed_to_multiplier"] == "te_target_mult"


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_ranking_key_differs_from_the_allocated_quantity():
    """Foster sorts last on the raw key while holding a large posterior."""
    out, status = build(pd.read_csv(UNIV, low_memory=False))
    f = out.loc[out["player"].eq("Montorie Foster")].iloc[0]
    assert pd.isna(f["role_ranking_key_raw_tgt_share"])
    assert float(f["pre_gate_bayes_tgt_share"]) > 0.16
    assert f["canonical_role_label"] == "NONE_ASSIGNED"
    # and the labelled WR1_5 holds a SMALLER posterior than the unlabelled Foster
    s = out.loc[out["player"].eq("Rashid Shaheed")].iloc[0]
    assert float(s["pre_gate_bayes_tgt_share"]) < float(
        f["pre_gate_bayes_tgt_share"])
    assert "RANKING_KEY_MISMATCH" in status


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_man_gate_is_certified_not_fired_and_te_rb_stay_unresolved():
    _, status = build(pd.read_csv(UNIV, low_memory=False))
    g = status["gate_firing"]
    assert g["man_coverage_gate_fired_for_KC"] is False
    assert g["te_and_rb_gates_fired"] == "NOT_ESTABLISHED"
    assert g["shared_pass_eff_mult_movement"] == "NOT_ESTABLISHED"


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_trace_is_outcome_blind_and_keeps_defects_separate():
    _, status = build(pd.read_csv(UNIV, low_memory=False))
    assert status["outcomes_inspected"] is False
    assert status["sportsbook_inputs_used"] == 0
    assert status["week7_used_to_score_anything"] is False
    assert status["walker_redone"] is False
    assert "separate_defect_not_merged" in status
    # structural vs circumstantial must not be pooled
    assert "STRUCTURAL" in status[
        "final_opponent_attributable_movement"]["distinction"]
