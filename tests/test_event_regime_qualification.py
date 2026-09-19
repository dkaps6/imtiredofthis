import pandas as pd
from scripts.research.build_event_regime_qualification import build_event_qualification

def test_event_support_and_onset_are_outcome_free():
    rows=[]
    for season in [2022,2023,2024]:
        for i in range(100):
            rows.append({"season":season,"week":1,"position":"RB","player_identity_key":f"p{season}_{i}","known_context_flag":True,"team_change_flag":True,"joint_player_room_transition_flag":False,"target_room_churn_flag":False,"rush_room_churn_flag":True})
    out=build_event_qualification(pd.DataFrame(rows))
    tc=out[(out.event_name=="team_change_flag") & (out.position=="RB")].iloc[0]
    assert tc.positive_events==300
    assert tc.seasons_with_positive==3
    assert tc.support_gate_250
    assert tc.event_onsets==300
    assert "yards" not in " ".join(out.columns).lower()

def test_duplicate_keys_fail_closed():
    import pytest
    r={"season":2024,"week":1,"position":"WR","player_identity_key":"x","known_context_flag":True,"team_change_flag":False,"joint_player_room_transition_flag":False,"target_room_churn_flag":False,"rush_room_churn_flag":False}
    with pytest.raises(RuntimeError,match="duplicate"):
        build_event_qualification(pd.DataFrame([r,r]))
