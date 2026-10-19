import pandas as pd

from scripts.research.audit_prism_player_team_context_transition_v1 import (
    CHANGED,
    CONTINUED,
    NO_PRIOR,
    build_states,
    normalize_rosters,
    normalize_schedule,
)


def _schedule():
    rows=[]
    for season in (2023,2024,2025,2026):
        for week in range(1,19):
            rows.append({
                "season":season,"week":week,"game_type":"REG",
                "home_team":"IND","away_team":"HOU",
            })
    return pd.DataFrame(rows)


def _row(season,week,team,pid,name,pos):
    return {
        "season":season,"week":week,"team":team,"gsis_id":pid,
        "player_name":name,"position":pos,"status":"ACT",
    }


def _empty():
    return pd.DataFrame(
        columns=["season","week","team","gsis_id","player_name","position","status"]
    )


def test_transition_is_exact_player_team_change_and_maps_competitors():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([
            _row(2023,15,"HOU","p1","Receiver One","WR"),
            _row(2023,16,"HOU","p1","Receiver One","WR"),
            _row(2023,17,"HOU","p1","Receiver One","WR"),
            _row(2023,18,"HOU","p1","Receiver One","WR"),
        ])),
        (2024,pd.DataFrame([
            _row(2024,1,"IND","p1","Receiver One","WR"),
            _row(2024,1,"IND","p2","Receiver Two","WR"),
            _row(2024,2,"IND","p1","Receiver One","WR"),
            _row(2024,2,"IND","p2","Receiver Two","WR"),
        ])),
        (2025,_empty()),
        (2026,_empty()),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,transitions,audit=build_states(roster)

    p1=states.loc[states["stable_id"].eq("p1")].sort_values(["season","week"])
    assert p1.iloc[0]["team_context_state"] == NO_PRIOR
    assert p1.iloc[-2]["team_context_state"] == CHANGED
    assert p1.iloc[-2]["transition_kind"] == "OFFSEASON_TEAM_CHANGE"
    assert p1.iloc[-2]["previous_rostered_team"] == "HOU"
    assert int(p1.iloc[-2]["prior_team_roster_week_count"]) == 4
    assert bool(p1.iloc[-2]["prior_team_support_ge4"])
    assert int(p1.iloc[-2]["same_family_competitor_count"]) == 1
    assert "p2" in p1.iloc[-2]["same_family_competitor_stable_ids"]
    assert p1.iloc[-1]["team_context_state"] == CONTINUED
    assert len(transitions.loc[transitions["stable_id"].eq("p1")]) == 1
    assert audit["chronology_violations"] == 0
    assert audit["skipped_previous_roster_violations"] == 0


def test_inseason_move_is_distinct_transition_kind():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,_empty()),
        (2024,pd.DataFrame([
            _row(2024,1,"HOU","r1","Runner One","RB"),
            _row(2024,2,"HOU","r1","Runner One","RB"),
            _row(2024,3,"IND","r1","Runner One","RB"),
        ])),
        (2025,_empty()),
        (2026,_empty()),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    _,transitions,_=build_states(roster)
    t=transitions.iloc[0]
    assert t["transition_kind"] == "INSEASON_TEAM_CHANGE"
    assert int(t["week"]) == 3
    assert t["previous_rostered_team"] == "HOU"
    assert t["team"] == "IND"


def test_same_week_multi_team_is_hard_ambiguity_not_transition():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([_row(2023,18,"HOU","t1","Tight End","TE")])),
        (2024,pd.DataFrame([
            _row(2024,1,"HOU","t1","Tight End","TE"),
            _row(2024,1,"IND","t1","Tight End","TE"),
        ])),
        (2025,_empty()),
        (2026,_empty()),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,transitions,audit=build_states(roster)
    assert audit["ambiguous_multi_team_player_weeks"] == 1
    assert not ((states["season"]==2024)&(states["week"]==1)&(states["stable_id"]=="t1")).any()
    assert transitions.empty


def test_blank_stable_id_is_never_admitted():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([_row(2023,18,"HOU","","Unknown","WR")])),
        (2024,pd.DataFrame([_row(2024,1,"IND","","Unknown","WR")])),
        (2025,_empty()),
        (2026,_empty()),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,transitions,_=build_states(roster)
    assert len(roster) == 2
    assert states.empty
    assert transitions.empty
