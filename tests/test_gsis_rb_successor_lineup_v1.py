import pandas as pd
from scripts.research.gsis_rb_successor_lineup_v1 import successor_weights, build_private_candidate

def row(team,players,plays):
    return {"team":team,"players":frozenset(players),"plays":float(plays)}

def test_successor_weights_condition_on_unavailable_absent():
    rows=[
      row("DEN",{"u","a","x","1","2","3","4","5","6","7","8"},100),
      row("DEN",{"a","x","1","2","3","4","5","6","7","8","9"},30),
      row("DEN",{"b","x","1","2","3","4","5","6","7","8","9"},10),
      row("DEN",{"a","b","1","2","3","4","5","6","7","8","9"},20),
    ]
    w=successor_weights(rows,team="DEN",unavailable={"u"},successors=["a","b"])
    assert abs(w["a"]-(50/80))<1e-12
    assert abs(w["b"]-(30/80))<1e-12
    assert abs(sum(w.values())-1.0)<1e-12

def test_multiple_unavailable_requires_all_absent():
    rows=[
      row("PIT",{"u1","a","1","2","3","4","5","6","7","8","9"},50),
      row("PIT",{"u2","b","1","2","3","4","5","6","7","8","9"},50),
      row("PIT",{"a","1","2","3","4","5","6","7","8","9","0"},20),
      row("PIT",{"b","1","2","3","4","5","6","7","8","9","0"},20),
    ]
    w=successor_weights(rows,team="PIT",unavailable={"u1","u2"},successors=["a","b"])
    assert w=={"a":0.5,"b":0.5}

def test_candidate_conserves_frozen_vacated_share():
    rows=[
      row("DEN",{"a","1","2","3","4","5","6","7","8","9","0"},30),
      row("DEN",{"b","1","2","3","4","5","6","7","8","9","0"},10),
    ]
    vacancy=pd.DataFrame([
      {"target_season":2026,"target_week":9,"team":"DEN","successor_player_clean_key":"a","vacated_rush_share":0.20,"unavailable_players":"u"},
      {"target_season":2026,"target_week":9,"team":"DEN","successor_player_clean_key":"b","vacated_rush_share":0.20,"unavailable_players":"u"},
    ])
    out,audit=build_private_candidate(rows,vacancy)
    assert audit["events_candidate_ready"]==1
    assert abs(out.gsis_transfer_rush_share.sum()-0.20)<1e-12
    assert audit["max_conservation_gap"]<=1e-12

def test_no_exposure_abstains():
    rows=[row("DEN",{"x","1","2","3","4","5","6","7","8","9","0"},20)]
    vacancy=pd.DataFrame([
      {"target_season":2026,"target_week":9,"team":"DEN","successor_player_clean_key":"a","vacated_rush_share":0.20,"unavailable_players":"u"}
    ])
    out,audit=build_private_candidate(rows,vacancy)
    assert out.empty
    assert audit["events_no_gsis_successor_exposure"]==1
