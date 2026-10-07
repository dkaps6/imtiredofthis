import pandas as pd
import scripts.research.evaluate_te_player_error_persistence_v1 as m

def test_prior_history_excludes_target():
    x=pd.DataFrame([
      {"season":2023,"week":17},{"season":2024,"week":1},{"season":2024,"week":2}
    ])
    assert m.prior_mask(x,2024,2).tolist()==[True,True,False]

def test_sign_agreement():
    x=pd.DataFrame({
      "prior8_signed_error_mean":[1,-1,2,-2],
      "current_signed_error":[2,-2,-1,1],
    })
    assert abs(m.sign_agreement(x)-.5)<1e-12
