import pandas as pd
import scripts.research.audit_player_situational_target_earning_v1 as m

def test_target_contexts_basic():
    x=pd.DataFrame({
      "season":[2026],"week":[1],"posteam":["IND"],"receiver_player_id":["00-1"],
      "pass_attempt":[1],"sack":[0],"two_point_attempt":[0],"down":[3],
      "yardline_100":[15],"half_seconds_remaining":[90],
    })
    # local semantic equivalent
    x=m.regular_only(m.lower(x))
    for c in ["season","week","posteam","receiver_player_id","pass_attempt","sack","two_point_attempt","down","yardline_100","half_seconds_remaining"]:
        assert c in x.columns

def test_rho_positive():
    assert m.rho(pd.Series([1,2,3]),pd.Series([2,4,6]))>0.99
