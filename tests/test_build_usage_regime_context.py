import numpy as np
import pandas as pd
import pytest

from scripts.research.build_usage_regime_context import build_usage_regime_context


def _history():
    return pd.DataFrame([
        dict(season=2025, week=1, team="A", player_identity_key="p1", position="RB", tgt_share_game=.10, rush_share_game=.50, route_rate_game=np.nan),
        dict(season=2025, week=2, team="A", player_identity_key="p1", position="RB", tgt_share_game=.20, rush_share_game=.60, route_rate_game=np.nan),
        dict(season=2025, week=3, team="B", player_identity_key="p1", position="RB", tgt_share_game=.90, rush_share_game=.90, route_rate_game=np.nan),
    ])


def test_target_game_usage_is_not_in_own_context():
    out = build_usage_regime_context(_history())
    w3 = out.loc[out.week.eq(3)].iloc[0]
    assert w3.prior_tgt_share_game == pytest.approx(.20)
    assert w3.prior3_tgt_share_game_mean == pytest.approx(.15)
    assert w3.prior_rush_share_game == pytest.approx(.60)
    assert w3.prior3_rush_share_game_mean == pytest.approx(.55)


def test_team_change_and_stint_are_strict_prior():
    out = build_usage_regime_context(_history())
    w1, w2, w3 = [out.loc[out.week.eq(w)].iloc[0] for w in (1, 2, 3)]
    assert pd.isna(w1.team_change_prior)
    assert w2.team_change_prior == 0
    assert w2.games_with_current_team_prior == 1
    assert w3.team_change_prior == 1
    assert w3.games_with_current_team_prior == 0


def test_duplicate_canonical_key_fails_closed():
    x = pd.concat([_history(), _history().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate canonical"):
        build_usage_regime_context(x)
