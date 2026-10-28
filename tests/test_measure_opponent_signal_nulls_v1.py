"""Guards for the three opponent-side nulls.

A null is only worth banking if the machinery could have detected a real effect and
if the null cannot quietly become a positive claim later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_opponent_signal_nulls_v1 as N


def _panel(n_players=60, games=16, effect=0.0, seed=0):
    """Panel where each player's sensitivity to opponent quality is `effect`-scaled."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        sens = effect * (i - n_players / 2) / n_players      # heterogeneous when effect>0
        for g in range(games):
            q = rng.uniform(-0.2, 0.2)
            rows.append({"pid": f"p{i}", "q": q, "y": 60.0 + sens * q + rng.normal(0, 8.0)})
    return pd.DataFrame(rows)


def test_frozen_thresholds():
    assert N.MIN_GAMES_FOR_SLOPE == 12
    assert N.MIN_DEF_PRIOR_PLAYS == 100
    assert N.DEEP_AIR_YARDS == 10.0
    assert N.N_PERMUTATIONS >= 200


def test_slope_estimator_recovers_a_planted_sensitivity():
    """If per-player slopes cannot recover a planted effect, null 1 means nothing."""
    flat = N.per_player_slopes(_panel(effect=0.0, seed=1))
    strong = N.per_player_slopes(_panel(effect=4000.0, seed=1))
    assert len(strong) == len(flat) > 0
    assert strong.std() > flat.std() * 3, "planted heterogeneity not detected"


def test_slope_estimator_requires_the_minimum_game_count():
    short = _panel(n_players=5, games=N.MIN_GAMES_FOR_SLOPE - 1)
    assert N.per_player_slopes(short).empty


def test_defense_depth_profile_separates_shape_from_level():
    """A defense soft deep and a defense soft short can share a level but not a shape."""
    rows = []
    for team, deep_y, short_y in (("AAA", 18.0, 4.0), ("BBB", 6.0, 8.0)):
        rows += [{"defteam": team, "band": "deep", "yg": deep_y}] * 70
        rows += [{"defteam": team, "band": "short", "yg": short_y}] * 120
    out = N._defense_depth_profile(pd.DataFrame(rows))
    assert out.loc["AAA", "shape"] > out.loc["BBB", "shape"]
    assert out.loc["AAA", "shape"] == pytest.approx(14.0)
    assert out.loc["BBB", "shape"] == pytest.approx(-2.0)


def test_defense_depth_profile_drops_thin_samples():
    rows = [{"defteam": "AAA", "band": "deep", "yg": 10.0}] * 10 + \
           [{"defteam": "AAA", "band": "short", "yg": 5.0}] * 10
    assert N._defense_depth_profile(pd.DataFrame(rows)).empty


def test_main_fails_closed_if_the_null_ever_becomes_real(monkeypatch):
    """This document records a null; it must refuse to keep publishing one if the
    finding flips, rather than silently contradicting its own text."""
    monkeypatch.setattr(N, "_frames", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(N, "null_1_player_matchup_sensitivity",
                        lambda w, p: {"verdict": "REAL", "permutation_p": 0.001})
    # nulls 2 and 3 run before the guard, so they must not crash on the stub frames
    monkeypatch.setattr(N, "null_2_defense_exploitable_shape", lambda p: {"verdict": "NOT_PERSISTENT"})
    monkeypatch.setattr(N, "null_3_game_script", lambda p: {"verdict": "NOT_FORECASTABLE"})
    with pytest.raises(RuntimeError, match="now reads REAL"):
        N.main()


def test_permutation_preserves_each_players_own_outcomes():
    """The null must destroy the player-opponent link WITHOUT changing his outcomes."""
    rng = np.random.default_rng(3)
    df = _panel(n_players=8, games=14, seed=2)
    shuffled = df.copy()
    shuffled["q"] = shuffled.groupby("pid")["q"].transform(lambda v: rng.permutation(v.to_numpy()))
    for pid, g in df.groupby("pid"):
        s = shuffled[shuffled["pid"] == pid]
        assert sorted(s["y"].round(9)) == sorted(g["y"].round(9)), "outcomes changed"
        assert sorted(s["q"].round(9)) == sorted(g["q"].round(9)), "opponent pool changed"
