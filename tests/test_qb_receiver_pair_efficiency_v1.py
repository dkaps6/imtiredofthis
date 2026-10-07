import pandas as pd
import scripts.research.evaluate_qb_receiver_pair_efficiency_v1 as m

def test_metric_block_perfect_pair():
    z=pd.DataFrame({
      "actual_ypt":[10.0,5.0],
      "control_receiver_ypt":[8.0,7.0],
      "pair_ypt":[10.0,5.0],
      "actual_yards":[20.0,5.0],
      "actual_targets":[2,1],
      "control_yards_at_actual_targets":[16.0,7.0],
      "pair_yards_at_actual_targets":[20.0,5.0],
    })
    c=m.metric_block(z,"control_receiver_ypt")
    p=m.metric_block(z,"pair_ypt")
    assert p["mae"]==0.0
    assert p["yard_mae_actual_targets"]==0.0
    assert c["mae"]>0

def test_chrono_before():
    x=pd.DataFrame({"season":[2022,2023,2023],"week":[18,4,5]})
    q=m.chrono_before(x,2023,5)
    assert q.tolist()==[True,True,False]
