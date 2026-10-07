import pandas as pd
import scripts.research.audit_player_state_live_coverage_v1 as m

def test_position_groups():
    assert m._pos("HB")=="RB"
    assert m._pos("FB")=="RB"
    assert m._pos("SWR")=="WR"
    assert m._pos("TE")=="TE"

def test_stable_key_prefers_gsis_then_pfr():
    assert m.stable_key(pd.Series({"gsis_id":"00-1","pfr_id":"P1","team":"IND","name_key":"x"}))=="gsis:00-1"
    assert m.stable_key(pd.Series({"gsis_id":"","pfr_id":"P1","team":"IND","name_key":"x"}))=="pfr:P1"

def test_consumption_map_preserves_specialists():
    assert "M89" in m.consumption_label("QB")
    assert "WR_R15" in m.consumption_label("WR")
    assert "TE_R5P" in m.consumption_label("TE")
    assert "NO_WEEK5_ROOM_SPECIALIST" in m.consumption_label("RB")
