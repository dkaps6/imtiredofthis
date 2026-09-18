import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "build_room_continuity_context.py"
spec = importlib.util.spec_from_file_location("room", P)
room = importlib.util.module_from_spec(spec)
spec.loader.exec_module(room)


def hist():
    return pd.DataFrame([
        # Week 1 WR room: A dominates targets.
        [2025,1,"AAA","A","WR",.70,0.0], [2025,1,"AAA","B","WR",.30,0.0],
        # Week 2: B remains, C enters. Target-game W3 must only see W1/W2.
        [2025,2,"AAA","B","WR",.60,0.0], [2025,2,"AAA","C","WR",.40,0.0],
        # Week 3 absurd spike must not alter W3 context.
        [2025,3,"AAA","C","WR",.99,0.0], [2025,3,"AAA","D","WR",.01,0.0],
    ], columns=["season","week","team","player_identity_key","position","tgt_share_game","rush_share_game"])


def test_target_week_is_strict_prior_and_overlap_is_weighted():
    out = room.build_room_continuity_context(hist())
    w3 = out[(out.season == 2025) & (out.week == 3) & (out.team == "AAA") & (out.position == "WR")].iloc[0]
    assert w3["prior_tgt_share_game_top1"] == pytest.approx(.60)
    assert w3["prior_tgt_share_game_top2"] == pytest.approx(1.0)
    # Of W2 target opportunity, only B (.60) was also present in W1.
    assert w3["prior_tgt_share_game_returning_overlap"] == pytest.approx(.60)


def test_first_room_game_has_explicit_unknown_state():
    out = room.build_room_continuity_context(hist())
    w1 = out[(out.week == 1) & (out.position == "WR")].iloc[0]
    assert w1["room_continuity_coverage_flag"] == "no_prior_room_game"
    assert np.isnan(w1["prior_tgt_share_game_top1"])


def test_duplicate_canonical_player_game_fails_closed():
    x = hist()
    x = pd.concat([x, x.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate canonical"):
        room.build_room_continuity_context(x)


def test_season_boundary_does_not_supply_prior_room_game():
    x = hist()
    nxt = pd.DataFrame([[2026,1,"AAA","B","WR",.5,0.0]], columns=x.columns)
    out = room.build_room_continuity_context(pd.concat([x,nxt], ignore_index=True))
    r = out[(out.season == 2026) & (out.week == 1)].iloc[0]
    assert r["room_continuity_coverage_flag"] == "no_prior_room_game"
