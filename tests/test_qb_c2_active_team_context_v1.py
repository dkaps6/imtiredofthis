import pandas as pd
import pytest
from scripts.utils.qb_c2_active_team_context_v1 import (
    active_qb_c2_schedule, validate_qb_c2_state_context
)

TEAMS=["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
"HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG",
"NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS"]

def sample(n):
    active=TEAMS[:n]
    pairs=[]
    for i in range(0,n,2):
        a,b=active[i:i+2]
        pairs.extend([{"season":2026,"week":5,"team":a,"opponent":b,"bye":False},
                      {"season":2026,"week":5,"team":b,"opponent":a,"bye":False}])
    pairs.extend({"season":2026,"week":5,"team":t,"opponent":"","bye":True} for t in TEAMS[n:])
    return pd.DataFrame(pairs), set(active)

@pytest.mark.parametrize("n",[30,32])
def test_exact_active_schedule_and_context(n, tmp_path):
    frame, active=sample(n)
    slate=active_qb_c2_schedule(frame,season=2026,week=5,expected_teams=active)
    assert len(slate)==n
    context=slate[["season","week","team","opponent"]].copy()
    context["sportsbook_inputs_used"]=0
    for name in ["pass_opportunity_spot","pass_efficiency_spot","rush_opportunity_spot","rush_efficiency_spot"]:
        context[name]=0.123
    schedule=tmp_path/"map.csv";frame.to_csv(schedule,index=False)
    result=validate_qb_c2_state_context(
        context,season=2026,week=5,schedule_path=schedule,expected_teams=active)
    assert result["teams"]==n
    assert result["bye_teams"]==32-n

def test_missing_team_fails_even_if_pairs_reciprocate():
    frame,active=sample(30)
    # Remove an entire game. A weakened count-only validator would miss this.
    frame=frame.loc[~frame["team"].isin(["ARI","ATL"])].copy()
    with pytest.raises(RuntimeError,match="certified active roles"):
        active_qb_c2_schedule(frame,season=2026,week=5,expected_teams=active)

def test_one_way_opponent_fails():
    frame,active=sample(30)
    frame.loc[frame["team"].eq("ARI"),"opponent"]="BUF"
    with pytest.raises(RuntimeError,match="reciprocity"):
        active_qb_c2_schedule(frame,season=2026,week=5,expected_teams=active)

def test_duplicate_team_fails():
    frame,active=sample(32)
    with pytest.raises(RuntimeError,match="duplicate"):
        active_qb_c2_schedule(pd.concat([frame,frame.iloc[[0]]]),season=2026,week=5,expected_teams=active)
