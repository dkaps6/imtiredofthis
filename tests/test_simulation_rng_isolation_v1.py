import numpy as np
import pandas as pd
import pytest

from scripts.simulation_rng_isolation_v1 import simulate


def _frame(te_a=0.10, te_b=0.10, wr2=0.12, wr3=0.08):
    rows = [
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "QB",
            "player_clean_key": "qb", "position": "QB", "role": "QB1",
            "baseline_entitlement_tgt_share": 0.0, "entitlement_tgt_share": 0.0,
            "te_r5p_applied": False, "wr_r15_applied": False,
            "rules_plays_est": 64.0, "rules_pass_rate": .58,
            "rules_rush_share": .05, "rules_ypa": 7.0, "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "WR1",
            "player_clean_key": "wr1", "position": "WR", "role": "WR1",
            "baseline_entitlement_tgt_share": .20, "entitlement_tgt_share": .20,
            "te_r5p_applied": False, "wr_r15_applied": False,
            "rules_rush_share": 0.0, "rules_catch_rate": .65, "rules_ypt": 8.0,
            "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "WR2",
            "player_clean_key": "wr2", "position": "WR", "role": "WR2",
            "baseline_entitlement_tgt_share": .12, "entitlement_tgt_share": wr2,
            "te_r5p_applied": False, "wr_r15_applied": True,
            "rules_rush_share": 0.0, "rules_catch_rate": .62, "rules_ypt": 7.5,
            "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "WR3",
            "player_clean_key": "wr3", "position": "WR", "role": "WR3",
            "baseline_entitlement_tgt_share": .08, "entitlement_tgt_share": wr3,
            "te_r5p_applied": False, "wr_r15_applied": True,
            "rules_rush_share": 0.0, "rules_catch_rate": .60, "rules_ypt": 7.0,
            "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "TEA",
            "player_clean_key": "tea", "position": "TE", "role": "TE1",
            "baseline_entitlement_tgt_share": .10, "entitlement_tgt_share": te_a,
            "te_r5p_applied": True, "wr_r15_applied": False,
            "rules_rush_share": 0.0, "rules_catch_rate": .70, "rules_ypt": 7.0,
            "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "TEB",
            "player_clean_key": "teb", "position": "TE", "role": "TE2",
            "baseline_entitlement_tgt_share": .10, "entitlement_tgt_share": te_b,
            "te_r5p_applied": True, "wr_r15_applied": False,
            "rules_rush_share": 0.0, "rules_catch_rate": .60, "rules_ypt": 6.5,
            "rules_volatility_mult": 1.0,
        },
        {
            "event_id": "g", "team": "T", "opponent": "O", "player": "RB",
            "player_clean_key": "rb", "position": "RB", "role": "RB1",
            "baseline_entitlement_tgt_share": .10, "entitlement_tgt_share": .10,
            "te_r5p_applied": False, "wr_r15_applied": False,
            "rules_rush_share": .60, "rules_catch_rate": .72, "rules_ypt": 6.0,
            "rules_ypc": 4.2, "rules_volatility_mult": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def test_te_redistribution_does_not_move_protected_arrays():
    left = simulate(_frame(), iterations=1200, seed=42)
    right = simulate(_frame(te_a=.16, te_b=.04), iterations=1200, seed=42)
    for key in left.values:
        if key[1] in {"tea", "teb"}:
            continue
        assert np.array_equal(left.values[key], right.values[key]), key
    assert not np.array_equal(
        left.values[("g", "tea", "rec_yards")],
        right.values[("g", "tea", "rec_yards")],
    )


def test_wr_redistribution_does_not_move_protected_arrays():
    left = simulate(_frame(), iterations=1200, seed=42)
    right = simulate(_frame(wr2=.17, wr3=.03), iterations=1200, seed=42)
    for key in left.values:
        if key[1] in {"wr2", "wr3"}:
            continue
        assert np.array_equal(left.values[key], right.values[key]), key
    assert not np.array_equal(
        left.values[("g", "wr2", "receptions")],
        right.values[("g", "wr2", "receptions")],
    )


def test_room_mass_change_fails_closed():
    bad = _frame(te_a=.17, te_b=.04)
    with pytest.raises(RuntimeError, match="mass not conserved"):
        simulate(bad, iterations=100, seed=42)


def test_repeatability():
    frame = _frame(te_a=.15, te_b=.05, wr2=.16, wr3=.04)
    a = simulate(frame, iterations=500, seed=123)
    b = simulate(frame, iterations=500, seed=123)
    assert set(a.values) == set(b.values)
    for key in a.values:
        assert np.array_equal(a.values[key], b.values[key]), key
