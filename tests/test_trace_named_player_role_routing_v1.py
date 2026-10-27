"""Pin the named-player role-routing findings against production source."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.research.trace_named_player_role_routing_v1 import (
    GATE_SPECS,
    ROUTING,
    allocator_chain,
    decode_gates,
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
    # audit 6445134492: this must be a real guard, and the labels must be
    # proven by parsing the assignments rather than by a missing substring.
    assert c["unlabelled_wr_falls_through_to_1.0"] is True
    assert c["labels_assigned_exactly"] == ["SLOT", "WR1", "WR1_5"]


def test_guard_raises_when_the_fallthrough_would_end(tmp_path, monkeypatch):
    """A boolean that never raises is not a guard."""
    src = Path("scripts/modeling/simulation_rules.py").read_text(encoding="utf-8")
    broken = src.replace(
        'elif pos in {"RB", "FB"}:\n            tgt_mult *= mods.rb_rec_target_mult',
        'elif pos in {"RB", "FB"}:\n            tgt_mult *= mods.rb_rec_target_mult\n'
        '        else:\n            tgt_mult *= 1.23', 1)
    assert broken != src
    d = tmp_path / "scripts" / "modeling"
    d.mkdir(parents=True)
    (d / "simulation_rules.py").write_text(broken, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="falls through|else branch"):
        assert_routing_matches_source()


def test_every_role_multiplier_product_set_is_distinct():
    """The decode is only sound if products are unique per role."""
    d = decode_gates({})
    for mult in GATE_SPECS:
        assert d["per_role"][mult]["products_all_distinct"] is True


def test_gate_decode_is_unique_and_cross_group_consistent():
    """1.265 is 1.15x1.10; 1.200 is the RB zone multiplier alone."""
    d = decode_gates({"te_target_mult": 1.265, "rb_rec_target_mult": 1.200,
                      "wr1_target_mult": 1.0, "wr1_5_target_mult": 1.0})
    assert d["cross_group_consistent"] is True
    g = d["gate_state"]
    assert g["coverage_zone_rate >= 0.60"] is True
    assert g["middle_open_rate >= 0.50"] is True
    assert g["coverage_man_rate >= 0.50"] is False
    assert g["pressure_mismatch > 0.075"] is False
    # pressure not fired => explicit sigma channel still, pass_eff bounded
    assert d["derived_consequences"]["volatility_mult"] == 1.0
    assert d["derived_consequences"]["pass_eff_mult_bound"] == ["1.00", "1.01"]


def test_a_contradictory_decode_is_reported_not_hidden():
    """TE says zone fired, RB says it did not -> must not silently resolve."""
    d = decode_gates({"te_target_mult": 1.265, "rb_rec_target_mult": 1.0})
    assert d["gate_state"]["coverage_zone_rate >= 0.60"] == "CONTRADICTION"
    assert d["cross_group_consistent"] is False


def test_no_wr_role_reads_zone_coverage():
    """The finding V1 obscured, pinned against the gate spec."""
    for mult in ("wr1_target_mult", "wr1_5_target_mult", "slot_target_mult"):
        gates = [g for g, _ in GATE_SPECS[mult]]
        assert not any("zone" in g for g in gates), (
            f"{mult} now reads zone coverage; the published finding is stale")
    for mult in ("te_target_mult", "rb_rec_target_mult"):
        assert any("zone" in g for g, _ in GATE_SPECS[mult])


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_allocator_chain_is_read_not_inferred():
    """Fields 7/10 come from the real allocator, and the room closes."""
    c = allocator_chain("SEA")
    assert c["available"] is True
    assert c["stages"] == ["rules_tgt_share", "post_m38_tgt_share",
                           "entitlement_tgt_share"]
    assert c["m38_conserves_team_mass"] is True
    room = c["team_room_audit"]
    assert abs(room["entitlement_sum"] - 0.95) < 1e-9
    assert abs(room["entitlement_gap"]) < 1e-9
    by = {r["player"]: r for r in c["rows"]}
    # M38 touches only WRs
    assert abs(by["AJ Barner"]["m38_delta_pct"]) < 1e-12
    # and it reduces the unlabelled WR while raising the labelled WR1
    assert by["Cooper Kupp"]["m38_delta_pct"] < -0.25
    assert by["Jaxon Smith-Njigba"]["m38_delta_pct"] > 0.25


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
def test_man_gate_not_fired_and_te_rb_now_decoded():
    _, status = build(pd.read_csv(UNIV, low_memory=False))
    g = status["gate_firing"]
    assert g["man_coverage_gate_fired_for_KC"] is False
    assert g["te_and_rb_gates_DECODED"]["coverage_zone_rate >= 0.60"] is True
    assert g["decode_cross_group_consistent"] is True
    assert g["shared_pass_eff_mult_movement"] == "BOUNDED_NOT_RESOLVED"
    # the rule-layer ratio must no longer be described as normalization
    rl = status["rule_layer_multiplier"]
    assert "NO team-room normalization" in rl["what_it_is"]
    assert "not a final allocated share" in rl["what_it_is_not"]


@pytest.mark.skipif(not UNIV.exists(), reason="week7 universe not present")
def test_zero_claim_is_scoped_to_the_direct_channel():
    _, status = build(pd.read_csv(UNIV, low_memory=False))
    z = status["direct_wr_coverage_target_rule_layer_delta"]
    assert "NOT total opponent attribution" in z["SCOPE"]
    assert "globally" in z["SCOPE"]
    assert "zone gate DID fire" in z["WHY_THIS_MATTERS_MORE_THAN_V1_SAID"]


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
        "direct_wr_coverage_target_rule_layer_delta"]["distinction"]
