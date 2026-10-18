import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_stage2_player_hierarchical_mechanism_eligibility_signal_v1 import (
    build_strict_prior_meta_state,
)


def _row(season, week, player, q, delta, game):
    return {
        "season": season,
        "week": week,
        "game_key": game,
        "team": "A",
        "cell": "RB_RECEIVING",
        "family": "RB",
        "player_id": player,
        "player_name_key": player,
        "authority_q": q,
        "baseline_crps": 1.0,
        "candidate_crps": 1.0 - delta,
        "delta_crps": delta,
    }


def test_same_week_rows_do_not_enter_role_or_player_history():
    x = pd.DataFrame([
        _row(2024, 5, "p1", 0.6, 1.0, "g1"),
        _row(2024, 5, "p2", 0.4, -1.0, "g1"),
        _row(2024, 6, "p1", 0.6, 2.0, "g2"),
    ])
    z, violations = build_strict_prior_meta_state(x)
    assert violations == 0
    w5 = z[z["week"].eq(5)]
    assert (w5["player_prior_positive_q_rows"] == 0).all()
    assert np.allclose(w5["role_prior_mean"], 0.0)


def test_role_prior_excludes_target_players_own_history():
    x = pd.DataFrame([
        _row(2024, 5, "p1", 0.6, 2.0, "g1"),
        _row(2024, 5, "p2", 0.4, -1.0, "g1"),
        _row(2024, 6, "p1", 0.6, 3.0, "g2"),
    ])
    z, _ = build_strict_prior_meta_state(x)
    r = z[(z["week"].eq(6)) & z["player_id"].eq("p1")].iloc[0]
    assert r["player_prior_positive_q_rows"] == 1
    assert np.isclose(r["player_prior_mean"], 2.0)
    # p1's +2 is excluded; p2's -1 is the other-player role prior.
    assert np.isclose(r["role_prior_mean"], -1.0)


def test_structural_zero_rows_never_count_as_player_mechanism_evidence():
    x = pd.DataFrame([
        _row(2024, 5, "p1", 0.0, 0.0, "g1"),
        _row(2024, 6, "p1", 0.5, 1.0, "g2"),
        _row(2024, 7, "p1", 0.5, 1.0, "g3"),
    ])
    z, _ = build_strict_prior_meta_state(x)
    r6 = z[z["week"].eq(6)].iloc[0]
    r7 = z[z["week"].eq(7)].iloc[0]
    assert r6["player_prior_positive_q_rows"] == 0
    assert r7["player_prior_positive_q_rows"] == 1


def test_fixed_three_pseudo_observation_shrinkage():
    rows = []
    # p1 gets three prior positive-q rows of +1.
    for week in (5, 6, 7):
        rows.append(_row(2024, week, "p1", 0.6, 1.0, f"g{week}"))
        rows.append(_row(2024, week, "p2", 0.4, -1.0, f"g{week}"))
    rows.append(_row(2024, 8, "p1", 0.6, 0.0, "g8"))
    x = pd.DataFrame(rows)
    z, _ = build_strict_prior_meta_state(x)
    r = z[(z["week"].eq(8)) & z["player_id"].eq("p1")].iloc[0]
    assert r["player_prior_positive_q_rows"] == 3
    assert np.isclose(r["player_prior_mean"], 1.0)
    assert np.isclose(r["role_prior_mean"], -1.0)
    # (3 * 1 + 3 * -1) / 6 = 0
    assert np.isclose(r["hierarchical_expected_benefit"], 0.0)
