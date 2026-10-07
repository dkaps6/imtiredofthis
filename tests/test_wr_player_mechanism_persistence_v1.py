import pandas as pd
import scripts.research.evaluate_wr_player_mechanism_persistence_v1 as m

def test_decomposition_identity():
    x=pd.DataFrame([{
      "variant":m.VARIANT,"season":2023,"week":5,"team":"IND","player_clean_key":"wr","wr_rank":2,
      "pred_targets":8.0,"mc_rec_yards":64.0,"actual_targets":6.0,"actual_rec_yards":42.0
    }])
    q=m.add_components(x)
    assert q.iloc[0].decomp_gap<1e-12
    assert abs(q.iloc[0].opportunity_error+q.iloc[0].efficiency_error-22.0)<1e-12

def test_prior_mask():
    x=pd.DataFrame({"season":[2023,2023,2024],"week":[4,5,1]})
    assert m.prior_mask(x,2023,5).tolist()==[True,False,False]
