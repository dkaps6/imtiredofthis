import pandas as pd
from scripts.research.build_historical_analog_state_v1 import materialize,audit,FEATURES

def _rows():
    rows=[]
    for i in range(40):
        r={"season":2020+i//20,"week":1+i%20,"team":"A" if i%2 else "B","player_identity_key":f"p{i%5}","position":"RB"}
        for j,f in enumerate(FEATURES): r[f]=(i+j)/100
        rows.append(r)
    return pd.DataFrame(rows)

def test_analog_neighbors_are_strict_prior_and_outcome_free():
    s,n=materialize(_rows())
    assert len(s)==40
    assert n.strict_prior.all()
    assert set(s.analog_state)=={"NO_ANALOG_SUPPORT","VALID_ANALOG"}
    assert "outcome" not in " ".join(s.columns).lower()
    q=audit(s,n)
    assert q.chronology_gate.all()
    assert not q.outcomes_read.any()
    assert not q.sportsbook_read.any()

def test_geometry_is_deterministic():
    a,_=materialize(_rows()); b,_=materialize(_rows())
    pd.testing.assert_frame_equal(a,b)
