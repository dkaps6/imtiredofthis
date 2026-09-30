import pandas as pd
from scripts.research.audit_verified_wr_cb_snapshot_rows_2024w1_v1 import exact_week_lookup

def test_exact_week_lookup_never_uses_season_fallback_or_collision():
    rosters=pd.DataFrame([
      {"season":2024,"week":1,"team":"IND","position":"WR","name_key":"alpha","player_id":"A"},
      {"season":2024,"week":2,"team":"IND","position":"WR","name_key":"beta","player_id":"B"},
      {"season":2024,"week":1,"team":"TEN","position":"WR","name_key":"gamma","player_id":"C"},
      {"season":2024,"week":1,"team":"IND","position":"WR","name_key":"dup","player_id":"D1"},
      {"season":2024,"week":1,"team":"IND","position":"WR","name_key":"dup","player_id":"D2"},
    ])
    out=exact_week_lookup(rosters,{"WR"})
    assert out[(2024,1,"IND","alpha")]==("A","WEEK_EXACT")
    assert (2024,1,"IND","beta") not in out
    assert (2024,1,"IND","gamma") not in out
    assert out[(2024,1,"IND","dup")]==("","COLLISION")
