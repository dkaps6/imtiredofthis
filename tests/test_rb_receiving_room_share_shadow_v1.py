from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.modeling.rb_receiving_room_share_shadow_v1 as shadow


def _base_frame():
    return pd.DataFrame([
        {
            "event_id":"G1","team":"IND","player":"Back A","player_clean_key":"backa",
            "position":"RB","entitlement_tgt_share":0.12,
        },
        {
            "event_id":"G1","team":"IND","player":"Back B","player_clean_key":"backb",
            "position":"RB","entitlement_tgt_share":0.08,
        },
        {
            "event_id":"G1","team":"IND","player":"WR One","player_clean_key":"wrone",
            "position":"WR","entitlement_tgt_share":0.25,
        },
        {
            "event_id":"G1","team":"IND","player":"TE One","player_clean_key":"teone",
            "position":"TE","entitlement_tgt_share":0.15,
        },
    ])


def test_shadow_preserves_team_rb_and_non_rb_mass(monkeypatch):
    frame=_base_frame()

    def fake_attach(frame, **kwargs):
        out=frame.copy()
        out["prior_rb_room_share"]=[0.8,0.2,np.nan,np.nan]
        out["prior_games"]=[10,10,np.nan,np.nan]
        return out

    monkeypatch.setattr(shadow,"attach_prior_room_share_raw",fake_attach)
    out,audit,summary=shadow.apply_rb_receiving_room_share_shadow(
        frame,season=2026,week=5,states=pd.DataFrame(),prev=pd.DataFrame()
    )
    rb=out.loc[out.position.isin(["RB","FB"])]
    assert rb.entitlement_tgt_share.sum()==pytest.approx(0.20)
    assert out.entitlement_tgt_share.sum()==pytest.approx(frame.entitlement_tgt_share.sum())
    assert out.loc[out.player.eq("WR One"),"entitlement_tgt_share"].iloc[0]==pytest.approx(0.25)
    assert out.loc[out.player.eq("TE One"),"entitlement_tgt_share"].iloc[0]==pytest.approx(0.15)
    a=out.loc[out.player.eq("Back A")].iloc[0]
    b=out.loc[out.player.eq("Back B")].iloc[0]
    assert a.rb_room_candidate_share==pytest.approx(0.8)
    assert b.rb_room_candidate_share==pytest.approx(0.2)
    assert a.entitlement_tgt_share==pytest.approx(0.16)
    assert b.entitlement_tgt_share==pytest.approx(0.04)
    assert summary["parameters_fit"]==0
    assert summary["max_team_entitlement_gap"] <= 1e-10
    assert summary["max_rb_entitlement_gap"] <= 1e-10
    assert summary["max_non_rb_entitlement_gap"] <= 1e-10
    assert audit.iloc[0].shadow_applied


def test_missing_history_player_keeps_exact_current_room_share(monkeypatch):
    frame=_base_frame()

    def fake_attach(frame, **kwargs):
        out=frame.copy()
        out["prior_rb_room_share"]=[0.9,np.nan,np.nan,np.nan]
        out["prior_games"]=[10,np.nan,np.nan,np.nan]
        return out

    monkeypatch.setattr(shadow,"attach_prior_room_share_raw",fake_attach)
    out,audit,_=shadow.apply_rb_receiving_room_share_shadow(
        frame,season=2026,week=5,states=pd.DataFrame(),prev=pd.DataFrame()
    )
    # Only one history-backed RB -> structural no-op.
    a=out.loc[out.player.eq("Back A")].iloc[0]
    b=out.loc[out.player.eq("Back B")].iloc[0]
    assert a.rb_room_candidate_share==pytest.approx(0.6)
    assert b.rb_room_candidate_share==pytest.approx(0.4)
    assert a.entitlement_tgt_share==pytest.approx(0.12)
    assert b.entitlement_tgt_share==pytest.approx(0.08)
    assert not audit.iloc[0].shadow_applied
    assert audit.iloc[0].fallback_reason=="LT2_HISTORY_PLAYERS"


def test_partial_missing_history_preserves_missing_mass(monkeypatch):
    frame=pd.DataFrame([
        {"event_id":"G1","team":"IND","player":"A","player_clean_key":"a","position":"RB","entitlement_tgt_share":0.10},
        {"event_id":"G1","team":"IND","player":"B","player_clean_key":"b","position":"RB","entitlement_tgt_share":0.06},
        {"event_id":"G1","team":"IND","player":"C","player_clean_key":"c","position":"RB","entitlement_tgt_share":0.04},
    ])

    def fake_attach(frame, **kwargs):
        out=frame.copy()
        out["prior_rb_room_share"]=[0.8,0.2,np.nan]
        out["prior_games"]=[10,10,np.nan]
        return out

    monkeypatch.setattr(shadow,"attach_prior_room_share_raw",fake_attach)
    out,_,_=shadow.apply_rb_receiving_room_share_shadow(
        frame,season=2026,week=5,states=pd.DataFrame(),prev=pd.DataFrame()
    )
    c=out.loc[out.player.eq("C")].iloc[0]
    # C was 20% of the room and stays exactly 20%.
    assert c.rb_room_current_share==pytest.approx(0.2)
    assert c.rb_room_candidate_share==pytest.approx(0.2)
    assert c.entitlement_tgt_share==pytest.approx(0.04)
    a=out.loc[out.player.eq("A")].iloc[0]
    b=out.loc[out.player.eq("B")].iloc[0]
    assert a.rb_room_candidate_share==pytest.approx(0.64)
    assert b.rb_room_candidate_share==pytest.approx(0.16)
