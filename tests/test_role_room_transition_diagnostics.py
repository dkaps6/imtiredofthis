import pandas as pd
import pytest
from scripts.research.build_role_room_transition_diagnostics import build_transition_diagnostics


def base():
    return pd.DataFrame([
      {"season":2024,"week":2,"position":"WR","player_identity_key":"a","team_change_prior":0,
       "target_share_delta_vs_roll3":0.08,"rush_share_delta_vs_roll3":0.0,
       "returning_target_opportunity_overlap":0.60,"returning_rush_opportunity_overlap":0.90,
       "any_context_unknown_flag":0},
      {"season":2024,"week":2,"position":"WR","player_identity_key":"b","team_change_prior":1,
       "target_share_delta_vs_roll3":0.01,"rush_share_delta_vs_roll3":0.0,
       "returning_target_opportunity_overlap":0.90,"returning_rush_opportunity_overlap":0.90,
       "any_context_unknown_flag":0},
      {"season":2024,"week":2,"position":"WR","player_identity_key":"c","team_change_prior":0,
       "target_share_delta_vs_roll3":None,"rush_share_delta_vs_roll3":None,
       "returning_target_opportunity_overlap":None,"returning_rush_opportunity_overlap":None,
       "any_context_unknown_flag":1},
    ])


def test_transition_flags_and_known_denominator():
    detail,summary=build_transition_diagnostics(base())
    a=detail.loc[detail.player_identity_key.eq("a")].iloc[0]
    assert bool(a.target_usage_transition_flag)
    assert bool(a.target_room_churn_flag)
    assert bool(a.joint_player_room_transition_flag)
    s=summary.iloc[0]
    assert s.known_rows==2
    assert s.known_context_rate==pytest.approx(2/3)
    assert s.any_transition_rate==1.0


def test_unknown_rows_do_not_become_transitions():
    detail,_=build_transition_diagnostics(base())
    c=detail.loc[detail.player_identity_key.eq("c")].iloc[0]
    assert not bool(c.any_transition_flag)
    assert not bool(c.known_context_flag)


def test_duplicate_player_period_fails_closed():
    d=base(); d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
    with pytest.raises(RuntimeError,match="duplicate player-period"):
        build_transition_diagnostics(d)
