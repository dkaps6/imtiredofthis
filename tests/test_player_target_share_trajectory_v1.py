import pandas as pd
import scripts.research.evaluate_player_target_share_trajectory_v1 as m

def test_trajectory_state_uses_recent2_vs_earlier():
    team_idx={(2024,"IND"):pd.DataFrame([
      {"season":2024,"week":1,"game_id":"g1","team":"IND","team_targets":10},
      {"season":2024,"week":2,"game_id":"g2","team":"IND","team_targets":10},
      {"season":2024,"week":3,"game_id":"g3","team":"IND","team_targets":10},
      {"season":2024,"week":4,"game_id":"g4","team":"IND","team_targets":10},
    ])}
    p={}
    for w,n in [(1,1),(2,1),(3,3),(4,3)]:
        p[(2024,w,f"g{w}","IND","p")]=n
    z=m.trajectory_state(team_idx,p,2024,5,"IND","p")
    assert abs(z["earlier_share"]-.10)<1e-12
    assert abs(z["recent2_share"]-.30)<1e-12
    assert abs(z["trajectory_delta"]-.20)<1e-12
    assert z["feature_max_week"]==4

def test_expected_sign():
    z=pd.DataFrame({"trajectory_delta":[-.1,0,.1],"opportunity_error":[2,0,-2]})
    assert m.rho(z)<-.99
