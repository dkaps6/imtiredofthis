import pandas as pd
import pytest

from scripts.research.build_role_room_context import build_role_room_context


def _usage():
    return pd.DataFrame([
        {"season":2024,"week":2,"team":"A","player_identity_key":"p1","position":"WR","career_games_prior":1,"usage_regime_coverage_flag":"known","prior_tgt_share_game":0.20},
        {"season":2024,"week":2,"team":"A","player_identity_key":"p2","position":"WR","career_games_prior":0,"usage_regime_coverage_flag":"no_prior_game","prior_tgt_share_game":None},
    ])


def _room():
    return pd.DataFrame([
        {"season":2024,"week":2,"team":"A","position":"WR","room_prior_games":1,"room_continuity_coverage_flag":"known","prior_tgt_share_game_top1":0.30},
    ])


def test_many_to_one_join_preserves_player_rows_and_support():
    out = build_role_room_context(_usage(), _room())
    assert len(out) == 2
    assert set(out.room_join_state) == {"matched"}
    p1 = out[out.player_identity_key == "p1"].iloc[0]
    p2 = out[out.player_identity_key == "p2"].iloc[0]
    assert p1.strict_prior_support_games == 1
    assert p1.any_context_unknown_flag == 0
    assert p2.strict_prior_support_games == 0
    assert p2.any_context_unknown_flag == 1


def test_missing_room_is_explicit_unknown_not_zero_context():
    out = build_role_room_context(_usage(), _room().iloc[0:0])
    assert set(out.room_join_state) == {"missing_room"}
    assert out.room_unknown_flag.eq(1).all()
    assert out.room_prior_games.isna().all()


def test_duplicate_player_game_fails_closed():
    u = pd.concat([_usage(), _usage().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate canonical player-game"):
        build_role_room_context(u, _room())


def test_duplicate_room_key_fails_closed():
    r = pd.concat([_room(), _room()], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate canonical team-position-week"):
        build_role_room_context(_usage(), r)
