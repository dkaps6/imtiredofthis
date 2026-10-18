import math
import pandas as pd

from scripts.research.score_prism_player_high_value_conversion_identity_v1 import (
    add_channel_baseline,
    candidate_probability,
)


def test_candidate_probability_fixed_shrinkage():
    p=float(candidate_probability([1],[3],[0.25])[0])
    assert math.isclose(p,(1+4*0.25)/(3+4),rel_tol=0,abs_tol=1e-12)


def test_channel_baseline_excludes_entire_target_week():
    ev=pd.DataFrame([
        {"channel":"RB_RUSH_INSIDE10","season":2022,"week":1,"play_id":"a","td_raw":1},
        {"channel":"RB_RUSH_INSIDE10","season":2022,"week":1,"play_id":"b","td_raw":0},
        {"channel":"RB_RUSH_INSIDE10","season":2022,"week":2,"play_id":"c","td_raw":0},
        {"channel":"RB_RUSH_INSIDE10","season":2022,"week":2,"play_id":"d","td_raw":0},
    ])
    out=add_channel_baseline(ev).sort_values(["season","week"])
    w1=out.iloc[0]; w2=out.iloc[1]
    assert w1.prior_channel_opps==0
    assert w1.prior_channel_tds==0
    assert w2.prior_channel_opps==2
    assert w2.prior_channel_tds==1
    assert math.isclose(w2.p_base,(1+0.5)/(2+1),abs_tol=1e-12)
