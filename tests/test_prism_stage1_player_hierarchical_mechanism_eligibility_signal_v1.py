import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_stage1_player_hierarchical_mechanism_eligibility_signal_v1 import (
    build_strict_prior_meta_state,
)


def _row(season, week, family, pid, delta, game):
    # Construct p/y values that reproduce the requested delta only for state
    # tests; build_strict_prior_meta_state consumes delta_brier directly.
    return {
        "season": season,
        "week": week,
        "team": "A",
        "opponent": "B",
        "game_key": game,
        "family": family,
        "player_id": pid,
        "player_name_key": pid,
        "y": 1,
        "p_baseline": 0.5,
        "p_candidate": 0.5,
        "delta_brier": delta,
    }


def test_same_week_rows_do_not_enter_family_or_player_history():
    x = pd.DataFrame([
        _row(2024, 5, "WR", "p1", 0.20, "g1"),
        _row(2024, 5, "WR", "p2", -0.10, "g1"),
        _row(2024, 6, "WR", "p1", 0.30, "g2"),
    ])
    z, violations = build_strict_prior_meta_state(x)
    assert violations == 0
    w5 = z[z["week"].eq(5)]
    assert (w5["player_prior_mechanism_rows"] == 0).all()
    assert np.allclose(w5["family_prior_mean"], 0.0)


def test_family_prior_excludes_target_players_own_history():
    x = pd.DataFrame([
        _row(2024, 5, "RB", "p1", 0.20, "g1"),
        _row(2024, 5, "RB", "p2", -0.10, "g1"),
        _row(2024, 6, "RB", "p1", 0.30, "g2"),
    ])
    z, violations = build_strict_prior_meta_state(x)
    assert violations == 0
    r = z[(z["week"].eq(6)) & z["player_id"].eq("p1")].iloc[0]
    assert r["player_prior_mechanism_rows"] == 1
    assert np.isclose(r["player_prior_mean"], 0.20)
    # p1's own +0.20 is excluded from the family prior.
    assert np.isclose(r["family_prior_mean"], -0.10)


def test_fixed_three_pseudo_observation_shrinkage():
    rows = []
    for week in (5, 6, 7):
        rows.append(_row(2024, week, "TE", "p1", 0.12, f"g{week}"))
        rows.append(_row(2024, week, "TE", "p2", -0.12, f"g{week}"))
    rows.append(_row(2024, 8, "TE", "p1", 0.0, "g8"))
    z, violations = build_strict_prior_meta_state(pd.DataFrame(rows))
    assert violations == 0
    r = z[(z["week"].eq(8)) & z["player_id"].eq("p1")].iloc[0]
    assert r["player_prior_mechanism_rows"] == 3
    assert np.isclose(r["player_prior_mean"], 0.12)
    assert np.isclose(r["family_prior_mean"], -0.12)
    # (3*0.12 + 3*-0.12) / 6 = 0.
    assert np.isclose(r["hierarchical_expected_benefit"], 0.0)


def test_less_than_three_player_rows_falls_back_to_family_prior():
    x = pd.DataFrame([
        _row(2024, 5, "QB", "p1", 0.10, "g1"),
        _row(2024, 5, "QB", "p2", -0.20, "g1"),
        _row(2024, 6, "QB", "p1", 0.10, "g2"),
        _row(2024, 6, "QB", "p2", -0.20, "g2"),
        _row(2024, 7, "QB", "p1", 0.0, "g3"),
    ])
    z, violations = build_strict_prior_meta_state(x)
    assert violations == 0
    r = z[(z["week"].eq(7)) & z["player_id"].eq("p1")].iloc[0]
    assert r["player_prior_mechanism_rows"] == 2
    assert np.isclose(r["hierarchical_expected_benefit"], r["family_prior_mean"])
