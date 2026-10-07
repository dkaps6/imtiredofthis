from __future__ import annotations

import pandas as pd
import pytest

import scripts.research.build_player_centric_week5_shadow_manifest_v1 as m


def _universe():
    return pd.DataFrame([
        {"player":"QB One","team":"IND","opponent":"HOU","position":"QB"},
        {"player":"RB One","team":"IND","opponent":"HOU","position":"RB"},
        {"player":"RB Two","team":"IND","opponent":"HOU","position":"RB"},
        {"player":"WR One","team":"IND","opponent":"HOU","position":"WR"},
        {"player":"TE One","team":"IND","opponent":"HOU","position":"TE"},
    ])


def _wrte():
    return pd.DataFrame([
        {
            "season":2026,"week":5,"event_id":"HOU|IND","team":"IND",
            "player_clean_key":"wrone","position_family":"WR",
            "baseline_entitlement_tgt_share":0.20,
            "shadow_entitlement_tgt_share":0.22,
            "entitlement_delta":0.02,"trajectory_available":True,
            "trajectory_route":"TRAJECTORY_AVAILABLE","trajectory_delta":0.10,
        },
        {
            "season":2026,"week":5,"event_id":"HOU|IND","team":"IND",
            "player_clean_key":"teone","position_family":"TE",
            "baseline_entitlement_tgt_share":0.12,
            "shadow_entitlement_tgt_share":0.11,
            "entitlement_delta":-0.01,"trajectory_available":True,
            "trajectory_route":"TRAJECTORY_AVAILABLE","trajectory_delta":-0.08,
        },
    ])


def _rb_target():
    return pd.DataFrame([
        {
            "season":2026,"week":5,"event_id":"HOU|IND","team":"IND",
            "player_clean_key":"rbone","position_family":"RB",
            "baseline_entitlement_tgt_share":0.08,
            "shadow_entitlement_tgt_share":0.09,
            "entitlement_delta":0.01,"trajectory_available":True,
            "trajectory_route":"TRAJECTORY_AVAILABLE","trajectory_delta":0.12,
        },
        {
            "season":2026,"week":5,"event_id":"HOU|IND","team":"IND",
            "player_clean_key":"rbtwo","position_family":"RB",
            "baseline_entitlement_tgt_share":0.06,
            "shadow_entitlement_tgt_share":0.05,
            "entitlement_delta":-0.01,"trajectory_available":False,
            "trajectory_route":"TRAJECTORY_UNAVAILABLE_NO_FIT_WEIGHT","trajectory_delta":0.0,
        },
    ])


def _rb_carry():
    return pd.DataFrame([
        {
            "season":2026,"week":5,"team":"IND","player":"RB One",
            "control_recent_carry_share":0.60,"shadow_player_state_share":0.70,
        },
        {
            "season":2026,"week":5,"team":"IND","player":"RB Two",
            "control_recent_carry_share":0.40,"shadow_player_state_share":0.30,
        },
    ])


def test_manifest_composes_disjoint_player_shadows_without_qb_change():
    out, audit, summary = m.build_manifest(_universe(), _wrte(), _rb_target(), _rb_carry())

    assert len(out) == 5
    assert audit["passed"].all()
    assert summary["target_shadow_rows"] == 4
    assert summary["rb_carry_shadow_rows"] == 2
    assert summary["rb_rows_with_both_shadows"] == 2
    assert summary["qb_shadow_rows"] == 0
    assert summary["target_depth_distribution_included"] is False

    qb = out.loc[out["position_family"].eq("QB")].iloc[0]
    assert qb["player_state_route"] == "PROTECTED_PRODUCTION_QB_NO_NEW_PLAYER_SHADOW"
    assert not qb["target_shadow_available"]
    assert not qb["rb_carry_shadow_available"]

    rb = out.loc[out["player_clean_key"].eq("rbone")].iloc[0]
    assert rb["player_state_route"] == "RB_RUSH_AND_TARGET_SHARE_SHADOWS"
    assert rb["target_shadow_family"] == "RB_TARGET_SHARE_TRAJECTORY_SHADOW_V1"
    assert rb["target_shadow_share"] == pytest.approx(0.09)
    assert rb["rb_carry_shadow_share"] == pytest.approx(0.70)

    wr = out.loc[out["player_clean_key"].eq("wrone")].iloc[0]
    assert wr["target_shadow_family"] == "WRTE_TARGET_SHARE_TRAJECTORY_V1"
    assert not wr["rb_carry_shadow_available"]


def test_rb_carry_locked_room_conservation_is_binding():
    carry = _rb_carry()
    carry.loc[0, "shadow_player_state_share"] = 0.8
    with pytest.raises(RuntimeError, match="room conservation"):
        m.build_manifest(_universe(), _wrte(), _rb_target(), carry)


def test_target_lock_row_missing_from_universe_fails_closed():
    wrte = pd.concat([
        _wrte(),
        pd.DataFrame([{
            "season":2026,"week":5,"event_id":"HOU|IND","team":"IND",
            "player_clean_key":"missingwr","position_family":"WR",
            "baseline_entitlement_tgt_share":0.01,
            "shadow_entitlement_tgt_share":0.01,
            "entitlement_delta":0.0,"trajectory_available":False,
            "trajectory_route":"TRAJECTORY_UNAVAILABLE_NO_CHANGE","trajectory_delta":0.0,
        }])
    ], ignore_index=True)
    with pytest.raises(RuntimeError, match="target lock rows missing"):
        m.build_manifest(_universe(), wrte, _rb_target(), _rb_carry())


def test_target_family_overlap_fails_closed():
    rb = _rb_target().copy()
    collision = _wrte().iloc[[0]].copy()
    collision["position_family"] = "RB"
    collision["player_clean_key"] = "rbone"
    collision["baseline_entitlement_tgt_share"] = 0.08
    collision["shadow_entitlement_tgt_share"] = 0.09
    with pytest.raises(RuntimeError):
        m.build_manifest(_universe(), pd.concat([_wrte(), collision]), rb, _rb_carry())


def test_unlocked_rb_remains_explicit_without_rush_shadow():
    carry = _rb_carry().iloc[[0]].copy()
    # Parent rushing shadow is defined only on its immutable locked cohort.
    # A partial synthetic room fails its own conservation before it can be
    # treated as a valid parent lock.
    with pytest.raises(RuntimeError, match="room conservation"):
        m.build_manifest(_universe(), _wrte(), _rb_target(), carry)
