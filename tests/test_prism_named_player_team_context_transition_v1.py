import pandas as pd

from scripts.research.audit_prism_named_player_team_context_transition_v1 import (
    STATE_CONTINUED,
    STATE_NO_PRIOR,
    STATE_TRANSITION,
    build_team_context,
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


def _roster_row(season,week,team,pid,name="Player One",pos="WR"):
    return {
        "season":season,
        "week":week,
        "team":team,
        "gsis_id":pid,
        "player_name":name,
        "position":pos,
        "status":"ACT",
    }


def test_team_change_is_named_player_transition_not_rookie_state():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([
            _roster_row(2023,17,"HOU","p1"),
            _roster_row(2023,18,"HOU","p1"),
        ])),
        (2024,pd.DataFrame([
            _roster_row(2024,1,"IND","p1"),
            _roster_row(2024,2,"IND","p1"),
            _roster_row(2024,1,"IND","rookie","Rookie One"),
        ])),
        (2025,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
        (2026,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,audit=build_team_context(roster)

    p1=states.loc[states["stable_id"].eq("p1")].sort_values(["season","week"])
    assert p1.iloc[0]["team_context_state"] == STATE_NO_PRIOR
    assert p1.iloc[1]["team_context_state"] == STATE_CONTINUED
    assert p1.iloc[2]["team_context_state"] == STATE_TRANSITION
    assert p1.iloc[2]["prior_team"] == "HOU"
    assert p1.iloc[2]["team"] == "IND"
    assert bool(p1.iloc[2]["transition_week1"])
    assert p1.iloc[3]["team_context_state"] == STATE_CONTINUED

    rookie=states.loc[states["stable_id"].eq("rookie")].iloc[0]
    assert rookie["team_context_state"] == STATE_NO_PRIOR
    assert audit["chronology_violations"] == 0
    assert audit["skipped_prior_observation_violations"] == 0


def test_inseason_second_move_creates_second_transition():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([_roster_row(2023,18,"HOU","p2",pos="RB")])),
        (2024,pd.DataFrame([
            _roster_row(2024,1,"HOU","p2",pos="RB"),
            _roster_row(2024,2,"IND","p2",pos="RB"),
            _roster_row(2024,3,"IND","p2",pos="RB"),
            _roster_row(2024,4,"HOU","p2",pos="RB"),
        ])),
        (2025,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
        (2026,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,_=build_team_context(roster)
    p=states.loc[states["stable_id"].eq("p2")].sort_values(["season","week"])
    trans=p.loc[p["team_context_state"].eq(STATE_TRANSITION)]
    assert len(trans) == 2
    assert list(trans["week"]) == [2,4]
    assert trans["transition_inseason"].all()


def test_same_week_multi_team_is_not_guessed():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([_roster_row(2023,18,"HOU","p3")])),
        (2024,pd.DataFrame([
            _roster_row(2024,1,"HOU","p3"),
            _roster_row(2024,1,"IND","p3"),
        ])),
        (2025,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
        (2026,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,audit=build_team_context(roster)
    assert audit["same_week_multi_team_ambiguous_player_weeks"] == 1
    # 2024 Week 1 rows are excluded from transition certification.
    assert not ((states["season"]==2024)&(states["week"]==1)&(states["stable_id"]=="p3")).any()


def test_blank_stable_id_can_never_certify_transition():
    scheduled,_=normalize_schedule(_schedule())
    parts=[
        (2023,pd.DataFrame([_roster_row(2023,18,"HOU","")])),
        (2024,pd.DataFrame([_roster_row(2024,1,"IND","")])),
        (2025,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
        (2026,pd.DataFrame(columns=["season","week","team","gsis_id","player_name","position","status"])),
    ]
    roster,_=normalize_rosters(parts,scheduled)
    states,_=build_team_context(roster)
    assert len(roster) == 2
    assert states.empty
