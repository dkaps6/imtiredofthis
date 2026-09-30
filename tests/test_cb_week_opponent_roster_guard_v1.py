import pandas as pd
from scripts.research.cb_week_opponent_roster_guard_v1 import assess_cb_week_opponent

def test_correct_wrong_ambiguous_and_unobserved_cb_week_membership():
    rows=pd.DataFrame([
        {"season":2025,"week":4,"opponent":"KC","cb_gsis_id":"X","cb_identity_method":"WEEK_EXACT","source_quality_row_ready":True},
        {"season":2025,"week":4,"opponent":"LAC","cb_gsis_id":"X","cb_identity_method":"FANTASYALARM_STABLE_ID_BRIDGE","source_quality_row_ready":True},
        {"season":2025,"week":4,"opponent":"LV","cb_gsis_id":"Y","cb_identity_method":"FANTASYALARM_STABLE_ID_BRIDGE","source_quality_row_ready":True},
        {"season":2025,"week":4,"opponent":"SEA","cb_gsis_id":"Z","cb_identity_method":"FANTASYALARM_STABLE_ID_BRIDGE","source_quality_row_ready":True},
        {"season":2025,"week":4,"opponent":"PHI","cb_gsis_id":"","cb_identity_method":"UNRESOLVED","source_quality_row_ready":False},
    ])
    r=pd.DataFrame([
        {"season":2025,"week":4,"team":"KC","player_id":"X","position":"CB"},
        {"season":2025,"week":4,"team":"LV","player_id":"Y","position":"CB"},
        {"season":2025,"week":4,"team":"ATL","player_id":"Y","position":"CB"},
    ])
    a=assess_cb_week_opponent(rows,r)
    assert a.cb_week_roster_status.tolist()==[
        "WEEK_EXACT_DEFENDER_OPPONENT_CONFIRMED","CONFIRMED_OTHER_WEEK_TEAM",
        "AMBIGUOUS_MULTITEAM_WEEK_ROSTER","WEEKLY_DEFENSIVE_ROSTER_UNOBSERVED",
        "CB_GSIS_UNRESOLVED"]
    assert a.source_quality_with_cb_week_guard.tolist()==[True,False,False,False,False]
    assert a.previously_eligible_now_quarantined.tolist()==[False,True,True,True,False]
def test_existing_source_ineligible_never_promoted_by_roster_membership():
    x=pd.DataFrame([{"season":2024,"week":1,"opponent":"PHI","cb_gsis_id":"X",
                     "cb_identity_method":"WEEK_EXACT","source_quality_row_ready":False}])
    r=pd.DataFrame([{"season":2024,"week":1,"team":"PHI","player_id":"X","position":"CB"}])
    out=assess_cb_week_opponent(x,r)
    assert not bool(out.iloc[0].source_quality_with_cb_week_guard)

def test_pipeline_can_score_week_roster_before_previous_ready_is_computed():
    from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import DEF_POSITIONS
    from scripts.research.cb_week_opponent_roster_guard_v1 import DEF_POSITIONS as GUARD
    assert DEF_POSITIONS == GUARD
    rows=pd.DataFrame([{"season":2024,"week":3,"opponent":"KC","cb_gsis_id":"X",
                        "cb_identity_method":"FANTASYALARM_STABLE_ID_BRIDGE"}])
    rost=pd.DataFrame([{"season":2024,"week":3,"team":"PHI","player_id":"X","position":"CB"}])
    out=assess_cb_week_opponent(rows,rost)
    assert out.iloc[0].cb_week_roster_status=="CONFIRMED_OTHER_WEEK_TEAM"
