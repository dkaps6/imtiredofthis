import pandas as pd
import scripts.research.evaluate_player_situational_target_residual_v1 as m

def test_context_sign_hypothesis():
    x=pd.DataFrame({"third_down_delta":[.1,.2,.3],"opportunity_error":[-1,-2,-3]})
    assert m.rho(x,"THIRD_DOWN") < -0.99

def test_chronology_before():
    x=pd.DataFrame({"season":[2022,2023,2023],"week":[18,4,5]})
    assert m.chronology_before(x,2023,5).tolist()==[True,True,False]
