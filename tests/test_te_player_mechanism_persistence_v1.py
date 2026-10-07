import pandas as pd
import scripts.research.evaluate_te_player_mechanism_persistence_v1 as m

def test_decomposition_identity():
    x=pd.DataFrame([{
      "season":2024,"week":5,"team":"IND","player_clean_key":"te",
      "candidate_targets_r5p":5.0,"targets":4.0,
      "candidate_receptions_r5p":3.0,"receptions":2.0,
      "candidate_rec_yards_r5p":40.0,"rec_yards":28.0
    }])
    q=m.add_components(x)
    assert q.iloc[0].decomp_gap<1e-12
    assert abs(q.iloc[0].opportunity_error+q.iloc[0].efficiency_error-12.0)<1e-12

def test_prior_mask():
    x=pd.DataFrame({"season":[2023,2024,2024],"week":[18,4,5]})
    assert m.prior_mask(x,2024,5).tolist()==[True,True,False]
