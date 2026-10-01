import pandas as pd

from scripts.research.rb_vacancy_opportunity_state_v1 import build_state


def test_multi_vacancy_preserves_full_unavailable_conditioning_set():
    availability=pd.DataFrame([
        {"team":"PIT","player_clean_key":"u1","position_group":"RB","definitive_unavailable":1,"final_availability_state":"UNAVAILABLE_REPORTED"},
        {"team":"PIT","player_clean_key":"u2","position_group":"RB","definitive_unavailable":1,"final_availability_state":"UNAVAILABLE_REPORTED"},
        {"team":"PIT","player_clean_key":"a","position_group":"RB","definitive_unavailable":0,"final_availability_state":"AVAILABLE_REPORTED"},
        {"team":"PIT","player_clean_key":"b","position_group":"RB","definitive_unavailable":0,"final_availability_state":"AVAILABLE_REPORTED"},
    ])
    roles=pd.DataFrame([
        {"team":"PIT","player_clean_key":"a","position_group":"RB"},
        {"team":"PIT","player_clean_key":"b","position_group":"RB"},
    ])
    logs=pd.DataFrame([
        {"season":2026,"week":3,"team":"PIT","player_clean_key":"u1","rush_share_game":0.25},
        # u2 intentionally has no strict-prior rush-share row.
    ])
    snaps=pd.DataFrame([
        {"season":2026,"week":3,"team":"PIT","player_clean_key":"a","offense_pct":0.75},
        {"season":2026,"week":3,"team":"PIT","player_clean_key":"b","offense_pct":0.25},
    ])

    out,exc=build_state(availability,roles,logs,snaps,2026,4)

    assert len(out)==2
    assert set(out["unavailable_players"])=={"u1|u2"}
    assert set(out["unavailable_prior_rush_shares"])=={"0.25|NA"}
    assert abs(float(out["vacated_rush_share"].iloc[0])-0.25)<1e-12
    assert abs(float(out["transfer_rush_share"].sum())-0.25)<1e-12

    missing=exc.loc[exc["reason"].eq("NO_PRIOR_RUSH_SHARE")]
    assert len(missing)==1
    assert missing.iloc[0]["player_clean_key"]=="u2"
