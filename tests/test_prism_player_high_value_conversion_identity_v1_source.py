import pandas as pd

from scripts.research.audit_prism_player_high_value_conversion_identity_v1_source import add_prior_support


def test_prior_support_excludes_entire_target_game():
    rows = [
        {"channel":"RB_RUSH_INSIDE10","season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p1","play_id":"1","td_raw":0},
        {"channel":"RB_RUSH_INSIDE10","season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p1","play_id":"2","td_raw":1},
        {"channel":"RB_RUSH_INSIDE10","season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p1","play_id":"3","td_raw":0},
        {"channel":"RB_RUSH_INSIDE10","season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p1","play_id":"4","td_raw":0},
        {"channel":"RB_RUSH_INSIDE10","season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p1","play_id":"5","td_raw":1},
    ]
    out = add_prior_support(pd.DataFrame(rows)).sort_values(["week"])
    assert out.iloc[0].high_value_opps == 2
    assert out.iloc[0].prior_high_value_opps == 0
    assert out.iloc[1].high_value_opps == 3
    assert out.iloc[1].prior_high_value_opps == 2
    assert out.iloc[1].prior_touchdowns == 1


def test_prior_support_is_player_specific():
    rows = [
        {"channel":"TE_RECEIVING_REDZONE","season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p1","play_id":"1","td_raw":1},
        {"channel":"TE_RECEIVING_REDZONE","season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p2","play_id":"2","td_raw":0},
    ]
    out = add_prior_support(pd.DataFrame(rows))
    assert out.prior_high_value_opps.eq(0).all()
