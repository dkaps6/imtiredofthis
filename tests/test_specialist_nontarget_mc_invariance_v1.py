import pandas as pd

from scripts.research.audit_specialist_nontarget_mc_invariance_v1 import (
    TOL,
    attach_stage,
    build_entitlement_states,
)


def _target():
    return pd.DataFrame([
        {
            "event_id": "g1", "team": "A", "player_clean_key": "te1", "position": "TE",
            "m38_explicit_entitlement_tgt_share": 0.10,
            "entitlement_tgt_share": 0.12,
            "wr_r15_anchor": False,
        },
        {
            "event_id": "g1", "team": "A", "player_clean_key": "rb1", "position": "RB",
            "m38_explicit_entitlement_tgt_share": 0.08,
            "entitlement_tgt_share": 0.08,
            "wr_r15_anchor": False,
        },
        {
            "event_id": "g1", "team": "A", "player_clean_key": "wr1", "position": "WR",
            "m38_explicit_entitlement_tgt_share": 0.20,
            "entitlement_tgt_share": 0.20,
            "wr_r15_anchor": True,
        },
        {
            "event_id": "g1", "team": "A", "player_clean_key": "wr2", "position": "WR",
            "m38_explicit_entitlement_tgt_share": 0.14,
            "entitlement_tgt_share": 0.12,
            "wr_r15_anchor": False,
        },
    ])


def _te():
    return pd.DataFrame([
        {
            "event_id": "g1", "team": "A", "player_clean_key": "te1",
            "te_r5p_entitlement_tgt_share": 0.12,
        }
    ])


def test_entitlement_states_define_protection_by_exact_stage_delta():
    s = build_entitlement_states(_target(), _te()).set_index("player_clean_key")
    assert s.loc["te1", "te_entitlement_delta"] == 0.02
    assert not bool(s.loc["te1", "te_protected"])
    assert bool(s.loc["rb1", "te_protected"])
    assert bool(s.loc["wr1", "te_protected"])
    assert bool(s.loc["wr1", "wr_protected"])
    assert not bool(s.loc["wr2", "wr_protected"])


def test_stage_a_unrelated_output_drift_is_detected_on_protected_player():
    s = build_entitlement_states(_target(), _te())
    d = pd.DataFrame([
        {
            "event_id": "g1", "player_clean_key": "rb1", "position": "RB",
            "market": "rush_yards", "mean_delta": 0.01,
            "abs_mean_delta": 0.01, "max_element_gap": 3.0,
        },
        {
            "event_id": "g1", "player_clean_key": "te1", "position": "TE",
            "market": "rec_yards", "mean_delta": 4.0,
            "abs_mean_delta": 4.0, "max_element_gap": 20.0,
        },
    ])
    out = attach_stage(d, s, stage="TE_R5P").set_index("player_clean_key")
    assert bool(out.loc["rb1", "protected"])
    assert out.loc["rb1", "semantic_class"] == "SEMANTICALLY_UNRELATED"
    assert bool(out.loc["rb1", "mean_drift"])
    assert not bool(out.loc["te1", "protected"])


def test_tolerance_does_not_call_floating_noise_an_entitlement_change():
    t = _target()
    t.loc[t.player_clean_key.eq("rb1"), "entitlement_tgt_share"] = 0.08 + TOL / 2
    s = build_entitlement_states(t, _te()).set_index("player_clean_key")
    assert bool(s.loc["rb1", "wr_protected"])
