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
