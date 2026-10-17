from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.evaluate_prism_team_high_value_creation_conditional_parent_v1 import (
    build_team_rate_rows,
    poisson_crps,
)
from scripts.research.evaluate_player_active_role_conditional_allocation_bridge_v1 import (
    enumerate_active_states,
)


def _team_row(season, week, team, opponent, rushes, i10, targets=30, rz=4):
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{team}_{opponent}",
        "game_key": f"{season}_{week:02d}_{team}_{opponent}",
        "team": team,
        "opponent": opponent,
        "rush_source_present": True,
        "target_source_present": True,
        "team_rush_attempts": float(rushes),
        "inside10_rush": float(i10),
        "team_targets": float(targets),
        "red_zone_targets": float(rz),
    }


def test_team_rate_is_last4_min3_and_strictly_prior_by_week():
    rows = []
    # Target team: week 1 is deliberately extreme and must fall out of W6 last-4.
    rows += [
        _team_row(2024, 1, "AAA", "ZZZ", 20, 10),
        _team_row(2024, 2, "AAA", "ZZZ", 20, 2),
        _team_row(2024, 3, "AAA", "ZZZ", 20, 2),
        _team_row(2024, 4, "AAA", "ZZZ", 20, 2),
        _team_row(2024, 5, "AAA", "ZZZ", 20, 2),
        _team_row(2024, 6, "AAA", "ZZZ", 20, 1),
    ]
    # Other team contributes to league state. Its W6 value must not leak into W6.
    rows += [
        _team_row(2024, 1, "BBB", "YYY", 20, 0),
        _team_row(2024, 2, "BBB", "YYY", 20, 0),
        _team_row(2024, 3, "BBB", "YYY", 20, 0),
        _team_row(2024, 4, "BBB", "YYY", 20, 0),
        _team_row(2024, 5, "BBB", "YYY", 20, 0),
        _team_row(2024, 6, "BBB", "YYY", 20, 20),
    ]
    x = pd.DataFrame(rows)
    out = build_team_rate_rows(x, "RB_RUSH_INSIDE10")
    r = out.loc[(out["team"] == "AAA") & (out["week"] == 6)].iloc[0]

    # Team last-four for W6 is W2-W5: 8 high-value rushes / 80 rushes.
    assert np.isclose(r["team_high_value_rate"], 8 / 80)
    assert r["team_history_min_week"] == 2
    assert r["team_history_max_week"] == 5

    # League prior is every W1-W5 team-game only:
    # AAA = 18/100, BBB = 0/100, so 18/200.
    assert np.isclose(r["league_high_value_rate"], 18 / 200)
    assert r["league_history_max_week"] == 5
    assert r["chronology_violation"] == 0


def test_poisson_crps_rewards_mean_near_observed_count():
    near = poisson_crps(3.0, 3)
    far = poisson_crps(8.0, 3)
    assert np.isfinite(near)
    assert np.isfinite(far)
    assert near < far


def test_qualified_stage2_active_set_expectation_conserves_mass():
    q = np.asarray([0.40, 0.30], dtype=float)
    p = np.asarray([0.80, 0.50], dtype=float)
    weights, named, complement = enumerate_active_states(q, p)

    expected_named = weights @ named
    expected_complement = float(weights @ complement)

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(expected_named.sum() + expected_complement, 1.0)
    assert np.all(expected_named >= 0)
    assert expected_complement >= 0


def test_same_stage2_expected_share_can_be_reused_without_candidate_drift():
    q = np.asarray([0.35, 0.25, 0.10], dtype=float)
    p = np.asarray([0.90, 0.75, 0.60], dtype=float)
    weights, named, _ = enumerate_active_states(q, p)
    expected = weights @ named

    baseline_share = expected.copy()
    candidate_share = expected.copy()
    assert np.allclose(baseline_share, candidate_share, atol=0, rtol=0)
