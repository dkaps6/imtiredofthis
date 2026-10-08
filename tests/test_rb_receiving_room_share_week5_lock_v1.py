from __future__ import annotations

import pandas as pd
import pytest

import scripts.research.lock_rb_receiving_room_share_week5_v1 as lock


def _locked_parent():
    rows=[]
    for team, vals in {
        "IND":[("a",0.75),("b",0.25)],
        "HOU":[("c",0.20),("d",0.30),("e",0.50)],
    }.items():
        for i,(key,state) in enumerate(vals):
            rows.append({
                "season":2026,"week":5,"team":team,"opponent":"X",
                "state_key":f"gsis:{key}","player":key.upper(),
                "gsis_id":key,"pfr_id":"","room_size":len(vals),
                "control_recent_carry_share":1/len(vals),
                "shadow_player_state_share":1/len(vals),
                "_state":state,
            })
    return pd.DataFrame(rows)


def test_build_candidate_normalizes_only_frozen_prior_room_state(monkeypatch):
    parent=_locked_parent()

    def fake_attach(p, live):
        out=p.copy()
        out["name_key"]=out["player"].str.lower()
        out["prior_games"]=10.0
        out["prior_rb_room_share"]=out["_state"]
        out["same_team_prior_rb_room_share"]=out["_state"]
        out["current_games"]=4
        out["same_team_current_games"]=4
        out["last3_targets"]=3.0
        out["last3_source_max_week"]=4
        out["snap_source_max_week"]=4
        out["chronology_valid"]=True
        out["roster_source_week"]=5
        out["identity_history_route"]="EXACT_GSIS_STRICT_PRIOR"
        return out

    monkeypatch.setattr(lock,"_attach_prior_room_share",fake_attach)
    out,teams,result=lock.build_candidate(parent,pd.DataFrame())

    ind=out.loc[out.team.eq("IND")].sort_values("state_key")
    assert list(ind.candidate_rb_receiving_room_share)==pytest.approx([0.75,0.25])
    hou=out.loc[out.team.eq("HOU")].sort_values("state_key")
    assert list(hou.candidate_rb_receiving_room_share)==pytest.approx([0.20,0.30,0.50])
    assert (teams.candidate_sum-1).abs().max() <= 1e-10
    assert result["parameters_fit"]==0
    assert result["week5_outcomes_read"]==0
    assert result["retrospective_result_used_to_change_rule"] is False
    assert result["receiving_state"]=="prior_rb_room_share"


def test_build_candidate_fails_closed_on_nonpositive_room_state(monkeypatch):
    parent=_locked_parent().loc[lambda d:d.team.eq("IND")].copy()

    def fake_attach(p, live):
        out=p.copy()
        out["name_key"]=out["player"].str.lower()
        out["prior_games"]=10.0
        out["prior_rb_room_share"]=0.0
        out["same_team_prior_rb_room_share"]=0.0
        out["current_games"]=4
        out["same_team_current_games"]=4
        out["last3_targets"]=0.0
        out["last3_source_max_week"]=4
        out["snap_source_max_week"]=4
        out["chronology_valid"]=True
        out["roster_source_week"]=5
        return out

    monkeypatch.setattr(lock,"_attach_prior_room_share",fake_attach)
    with pytest.raises(RuntimeError,match="nonpositive Week-5 prior RB receiving-room mass"):
        lock.build_candidate(parent,pd.DataFrame())


def test_gsis_snapshot_ignores_name_suffix_aliases():
    queries=pd.DataFrame([
        {
            "team":"JAX","player":"Chris Rodriguez Jr.","state_key":"gsis:00-1",
            "gsis_id":"00-1",
        },
        {
            "team":"MIA","player":"Ollie Gordon II","state_key":"gsis:00-2",
            "gsis_id":"00-2",
        },
    ])
    states=pd.DataFrame([
        {"player_id":"00-1","time_key":202604,"team":"JAX","after_rb_room_share":0.70,"after_games":4.0},
        {"player_id":"00-2","time_key":202603,"team":"MIA","after_rb_room_share":0.35,"after_games":3.0},
        # Exact target week must not be consumed.
        {"player_id":"00-1","time_key":202605,"team":"JAX","after_rb_room_share":0.99,"after_games":5.0},
    ])
    out=lock._snapshot_prior_room_share_by_gsis(queries,states)
    a=out.loc[out.gsis_id.eq("00-1")].iloc[0]
    b=out.loc[out.gsis_id.eq("00-2")].iloc[0]
    assert a.prior_rb_room_share==pytest.approx(0.70)
    assert a.prior_games==pytest.approx(4.0)
    assert b.prior_rb_room_share==pytest.approx(0.35)
    assert b.prior_games==pytest.approx(3.0)
