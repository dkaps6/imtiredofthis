import pandas as pd
import pytest

from scripts.research.build_event_regime_redundancy_audit import audit_event_redundancy


def _fixtures():
    hist=[]
    detail=[]
    players=[f"p{i:03d}" for i in range(100)]
    for season in range(2019, 2026):
        for week in (1, 2):
            for i,pid in enumerate(players):
                hist.append({
                    "season":season,"week":week,"team":"T","player_identity_key":pid,
                    "targets":2 + (i % 3),"rushes":3 + (i % 4),
                    "team_targets":30,"team_rushes":25,
                })
                detail.append({
                    "season":season,"week":week,"team":"T","player_identity_key":pid,
                    "position":"RB","known_context_flag":True,"stable_identity_flag":1,
                    # Perfectly reconstructible from current-season game count.
                    "team_change_flag": week == 2,
                    # Deliberately unrelated to production opportunity state.
                    "joint_player_room_transition_flag": (i % 2) == 0,
                    "target_room_churn_flag": (i % 2) == 0,
                    "rush_room_churn_flag": week == 2,
                    "returning_target_opportunity_overlap": 0.4 if (i % 2)==0 else 0.9,
                    "returning_rush_opportunity_overlap": 0.5 if (i % 2)==0 else 0.95,
                })
    summary=[]
    for event in ["team_change_flag","joint_player_room_transition_flag","target_room_churn_flag","rush_room_churn_flag"]:
        positive = 700
        summary.append({
            "event_name":event,"position":"RB","known_rows":1400,
            "positive_events":positive,"prevalence":0.5,
            "seasons_with_positive":7,"max_season_share":1/7,
            "event_onsets":positive,"onset_fraction_of_positive":1.0,
            "support_gate_250":True,"known_coverage_gate_080":True,
        })
    return pd.DataFrame(hist), pd.DataFrame(detail), pd.DataFrame(summary)


def test_event_redundancy_separates_reconstructible_from_incremental():
    h,d,s=_fixtures()
    out=audit_event_redundancy(h,d,s)
    team=out[(out.event_name=="team_change_flag") & (out.position=="RB")].iloc[0]
    joint=out[(out.event_name=="joint_player_room_transition_flag") & (out.position=="RB")].iloc[0]
    assert team.highly_reconstructible
    assert team.qualification_disposition=="DESCRIPTIVE_ONLY"
    assert not joint.highly_reconstructible
    assert joint.qualification_disposition=="READY_FOR_FROZEN_EXPERIMENT"
    assert not bool(joint.outcomes_read)
    assert not bool(joint.sportsbook_read)


def test_event_redundancy_rejects_duplicate_keys():
    h,d,s=_fixtures()
    d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_event_redundancy(h,d,s)


def test_source_thin_event_cannot_be_ready():
    h,d,s=_fixtures()
    mask=(s.event_name=="joint_player_room_transition_flag") & (s.position=="RB")
    s.loc[mask,"support_gate_250"]=False
    out=audit_event_redundancy(h,d,s)
    joint=out[(out.event_name=="joint_player_room_transition_flag") & (out.position=="RB")].iloc[0]
    assert joint.qualification_disposition=="ENGINEERING_READY_SOURCE_THIN"
