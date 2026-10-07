import pandas as pd
import scripts.research.lock_rb_player_state_allocation_shadow_v1 as m

def _rows():
    return pd.DataFrame([
        {"team":"IND","opponent":"DEN","state_key":"a","player":"A","gsis_id":"a","pfr_id":"",
         "position_group":"RB","room_size":3,"last3_room_opportunity_share":0.75,
         "last3_room_snap_fraction":0.60,"chronology_valid":True},
        {"team":"IND","opponent":"DEN","state_key":"b","player":"B","gsis_id":"b","pfr_id":"",
         "position_group":"RB","room_size":3,"last3_room_opportunity_share":0.25,
         "last3_room_snap_fraction":0.40,"chronology_valid":True},
    ])

def test_equal_weight_shadow_and_conservation():
    out,teams,result=m.build(_rows())
    a=out.set_index("state_key")
    assert abs(a.loc["a","control_recent_carry_share"]-.75)<1e-12
    assert abs(a.loc["a","shadow_player_state_share"]-.675)<1e-12
    assert abs(out["shadow_player_state_share"].sum()-1)<1e-12
    assert result["parameters_fit"]==0
    assert result["week5_outcomes_read"]==0

def test_team_requires_two_lockable_players():
    x=_rows().iloc[[0]].copy()
    try:
        m.build(x)
    except RuntimeError as e:
        assert "zero lockable" in str(e)
    else:
        raise AssertionError("expected fail closed")
