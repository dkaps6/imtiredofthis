import numpy as np
import pandas as pd

import scripts.research.score_rb_opponent_defender_injury_candidate_v1 as m


def test_burden_zero_and_incomplete_semantics():
    x=pd.DataFrame([
        {"season":2024,"week":2,"team":"PIT","front7":True,"out_doubtful":True,
         "snap_joined":True,"chronology_valid":True,"prior_defense_pct":0.8},
        {"season":2024,"week":2,"team":"PIT","front7":True,"out_doubtful":True,
         "snap_joined":True,"chronology_valid":True,"prior_defense_pct":0.5},
        {"season":2024,"week":2,"team":"CLE","front7":True,"out_doubtful":True,
         "snap_joined":False,"chronology_valid":True,"prior_defense_pct":np.nan},
    ])
    b=m.build_burden(x).set_index("team")
    assert abs(b.loc["PIT","front7_out_doubtful_snap_mass"]-1.3)<1e-12
    assert bool(b.loc["PIT","candidate_team_week_complete"])
    assert not bool(b.loc["CLE","candidate_team_week_complete"])
    assert np.isnan(b.loc["CLE","front7_out_doubtful_snap_mass"])


def test_zero_intercept_fit_and_zero_burden_noop():
    q=pd.DataFrame([
        {"season":2024,"game_id":"g1","candidate_scoreable":True,"actual":12.0,
         "baseline_projection":10.0,"front7_out_doubtful_snap_mass":1.0},
        {"season":2024,"game_id":"g2","candidate_scoreable":True,"actual":14.0,
         "baseline_projection":10.0,"front7_out_doubtful_snap_mass":2.0},
    ])
    f=m.fit_beta(q)
    assert abs(f["beta_train"]-2.0)<1e-12


def test_metrics_tail_counts():
    z=m.metrics(pd.Series([0.0,0.0]),pd.Series([80.0,101.0]))
    assert z["tail75"]==2
    assert z["tail100"]==1
