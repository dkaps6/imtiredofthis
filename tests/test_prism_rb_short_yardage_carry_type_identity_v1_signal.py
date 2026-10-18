import math
import pandas as pd

from scripts.research.score_prism_rb_short_yardage_carry_type_identity_v1 import (
    add_team_prior,
    candidate_probability,
)


def test_team_prior_excludes_entire_target_game():
    pg=pd.DataFrame([
        {"season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p1","carries":2,"short_yardage_carries":1},
        {"season":2023,"week":1,"game_id":"g1","team":"A","player_id":"p2","carries":3,"short_yardage_carries":0},
        {"season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p1","carries":4,"short_yardage_carries":1},
        {"season":2023,"week":2,"game_id":"g2","team":"A","player_id":"p2","carries":1,"short_yardage_carries":1},
    ])
    out=add_team_prior(pg).sort_values(["week","player_id"]).reset_index(drop=True)
    w1=out.loc[out.week.eq(1)]
    w2=out.loc[out.week.eq(2)]
    assert set(w1.prior_team_rb_carries)=={0}
    assert set(w1.prior_team_rb_short_yardage_carries)=={0}
    assert set(w2.prior_team_rb_carries)=={5}
    assert set(w2.prior_team_rb_short_yardage_carries)=={1}
    assert all(math.isclose(v,(1+0.5)/(5+1),abs_tol=1e-12) for v in w2.p_team)


def test_candidate_probability_uses_frozen_20_pseudo_carries():
    p=float(candidate_probability([2],[10],[0.10])[0])
    assert math.isclose(p,(2+20*0.10)/(10+20),rel_tol=0,abs_tol=1e-12)


def test_player_signal_can_differ_from_team_baseline_without_changing_carries():
    pg=pd.DataFrame([
        {"season":2024,"week":1,"game_id":"g1","team":"A","player_id":"p","carries":10,"short_yardage_carries":4},
        {"season":2024,"week":2,"game_id":"g2","team":"A","player_id":"p","carries":5,"short_yardage_carries":0},
    ])
    # A player with 10 prior same-team carries and 4 prior short-yardage carries
    # receives a candidate probability shrunk toward, but distinct from, the team baseline.
    p=float(candidate_probability([4],[10],[0.10])[0])
    assert p > 0.10
    assert math.isclose(pg.loc[1,"carries"],5)
