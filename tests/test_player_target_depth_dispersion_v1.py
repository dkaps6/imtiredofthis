import pandas as pd, numpy as np
import scripts.research.evaluate_player_target_depth_dispersion_v1 as m

def test_expected_positive_rho():
    x=pd.DataFrame({"prior8_target_depth_sd":[1,2,3],"abs_efficiency_error":[2,4,8]})
    assert m.rho(x)>.99

def test_before_excludes_target():
    x=pd.DataFrame({"season":[2023,2024,2024],"week":[18,4,5]})
    assert m.before(x,2024,5).tolist()==[True,True,False]
