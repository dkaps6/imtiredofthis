import pandas as pd

from scripts.research.audit_prism_rb_short_yardage_carry_type_identity_v1_source import add_prior_support


def test_prior_support_same_team_and_excludes_target_game():
    rows=[
        {"season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p","play_id":"1","short_yardage":1},
        {"season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p","play_id":"2","short_yardage":0},
        {"season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p","play_id":"3","short_yardage":0},
        {"season":2023,"week":3,"game_id":"g3","team":"B","player_id":"p","play_id":"4","short_yardage":1},
    ]
    out=add_prior_support(pd.DataFrame(rows)).sort_values(["season","week"])
    assert out.iloc[0].prior_same_team_carries==0
    assert out.iloc[1].prior_same_team_carries==2
    assert out.iloc[1].prior_same_team_short_yardage_carries==1
    assert out.iloc[2].prior_same_team_carries==0
