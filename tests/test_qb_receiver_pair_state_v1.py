import pandas as pd
import scripts.research.audit_qb_receiver_pair_state_v1 as m

def test_clean_and_team():
    assert m.clean(None)==""
    assert m.clean("00-123")=="00-123"

def test_source_summary_counts_pairs():
    x=pd.DataFrame([
      {"season":2026,"week":1,"official_pass_attempt":True,"target_like":True,"passer_id":"q","receiver_id":"r1"},
      {"season":2026,"week":1,"official_pass_attempt":True,"target_like":True,"passer_id":"q","receiver_id":"r2"},
    ])
    s=m.source_summary(x).iloc[0]
    assert s["target_events"]==2
    assert s["distinct_pairs"]==2
    assert s["joint_pair_id_coverage"]==1.0
