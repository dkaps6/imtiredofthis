import pandas as pd
from scripts.research.market_offer_residual_probability_v1 import implied_prob, prepare_offers

def test_implied_prob():
    assert abs(implied_prob(-110)-110/210)<1e-12
    assert abs(implied_prob(120)-100/220)<1e-12

def test_offer_features_and_weights():
    src=pd.DataFrame([{"game_id":"g","player_clean_key":"p","market":"pass_yards","season":2024,"week":1,"team":"A","opponent":"B","model_projection":260.0,"actual":270.0}])
    props=pd.DataFrame([
      {"game_id":"g","player_clean_key":"p","market":"pass_yards","book":"DK","line":250.5,"over_odds":-110,"under_odds":-110},
      {"game_id":"g","player_clean_key":"p","market":"pass_yards","book":"FD","line":252.5,"over_odds":-105,"under_odds":-115},
    ])
    z=prepare_offers(src,props)
    assert len(z)==2
    assert abs(z.sample_weight.sum()-1.0)<1e-12
    assert set(z.line_range)=={2.0}
    assert set(z.crossbook_median_line)=={251.5}


def test_offer_join_preserves_football_authority_columns_without_suffixes():
    src=pd.DataFrame([{
      "game_id":"g","player_clean_key":"p","market":"pass_yards",
      "season":2024,"week":7,"team":"A","opponent":"B","player":"QB",
      "model_projection":260.0,"actual":270.0,
      "model_authority":"QB_PASS_SYNTHESIS_V1","authority_scope":"exact"
    }])
    props=pd.DataFrame([{
      "game_id":"g","player_clean_key":"p","market":"pass_yards","book":"DK",
      "line":250.5,"over_odds":-110,"under_odds":-110,
      "season":1999,"week":99,"player":"stale sportsbook label"
    }])
    z=prepare_offers(src,props)
    assert len(z)==1
    assert int(z.iloc[0]["season"])==2024
    assert int(z.iloc[0]["week"])==7
    assert z.iloc[0]["team"]=="A"
    assert z.iloc[0]["opponent"]=="B"
    assert z.iloc[0]["model_authority"]=="QB_PASS_SYNTHESIS_V1"
    assert "season_x" not in z.columns and "season_y" not in z.columns
