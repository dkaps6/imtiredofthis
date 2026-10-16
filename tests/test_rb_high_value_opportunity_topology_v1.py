import pandas as pd

from scripts.research.audit_rb_high_value_opportunity_topology_v1 import (
    situation_flags,
)


def test_situation_boundaries_are_exact():
    x=pd.DataFrame({
        "yardline_100":[21,20,10,5,1,50],
        "goal_to_go":[0,0,0,1,1,0],
        "down":[1,1,2,3,4,3],
        "ydstogo":[10,10,5,2,1,3],
    })
    f=situation_flags(x)
    assert f["ANY_RB_RUSH"].all()
    assert f["RED_ZONE_RUSH"].tolist()==[False,True,True,True,True,False]
    assert f["INSIDE_10_RUSH"].tolist()==[False,False,True,True,True,False]
    assert f["INSIDE_5_RUSH"].tolist()==[False,False,False,True,True,False]
    assert f["GOAL_TO_GO_RUSH"].tolist()==[False,False,False,True,True,False]
    assert f["SHORT_YARDAGE_RUSH"].tolist()==[False,False,False,True,True,False]


def test_short_yardage_requires_third_or_fourth_down():
    x=pd.DataFrame({
        "yardline_100":[50,50,50,50],
        "goal_to_go":[0,0,0,0],
        "down":[1,2,3,4],
        "ydstogo":[1,1,2,2],
    })
    f=situation_flags(x)
    assert f["SHORT_YARDAGE_RUSH"].tolist()==[False,False,True,True]
