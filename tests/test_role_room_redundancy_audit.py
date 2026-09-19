import numpy as np
import pandas as pd

from scripts.research.build_role_room_redundancy_audit import (
    audit_redundancy,
    build_production_opportunity_state,
)


def _history():
    rows = []
    for season in range(2019, 2026):
        for week in range(1, 7):
            for i in range(50):
                pid = f"p{i}"
                tgt = (i % 8) + week
                rush = (i % 6) + week
                rows.append({
                    "season": season, "week": week, "team": f"T{i%10}",
                    "player_identity_key": pid, "targets": tgt, "rushes": rush,
                    "team_targets": 35 + (i % 4), "team_rushes": 28 + (i % 3),
                })
    return pd.DataFrame(rows)


def test_production_state_is_strict_prior_and_playerform_weighted():
    h = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A", "player_identity_key": "p", "targets": 2, "rushes": 1, "team_targets": 10, "team_rushes": 10},
        {"season": 2024, "week": 2, "team": "A", "player_identity_key": "p", "targets": 8, "rushes": 3, "team_targets": 20, "team_rushes": 10},
        {"season": 2025, "week": 1, "team": "A", "player_identity_key": "p", "targets": 9, "rushes": 4, "team_targets": 30, "team_rushes": 10},
        {"season": 2025, "week": 2, "team": "A", "player_identity_key": "p", "targets": 1, "rushes": 2, "team_targets": 10, "team_rushes": 10},
    ])
    s = build_production_opportunity_state(h)
    w1 = s[(s.season == 2025) & (s.week == 1)].iloc[0]
    w2 = s[(s.season == 2025) & (s.week == 2)].iloc[0]
    assert np.isnan(w1.prod_tgt_current_share)
    assert w1.prod_tgt_prior_games == 2
    assert np.isclose(w1.prod_tgt_prior_share, 10 / 30)
    assert np.isclose(w1.prod_tgt_playerform_blend, 10 / 30)
    # Week 2 may use Week 1, but must not use Week 2 itself.
    assert np.isclose(w2.prod_tgt_current_share, 9 / 30)
    expected = (1 - 1 / 5) * (10 / 30) + (1 / 5) * (9 / 30)
    assert np.isclose(w2.prod_tgt_playerform_blend, expected)


def test_redundancy_audit_is_outcome_free_and_uses_temporal_holdout():
    h = _history()
    state = build_production_opportunity_state(h)
    c = h[["season", "week", "team", "player_identity_key"]].copy()
    # Deliberately reconstructible candidate.
    c = c.merge(state[[*c.columns, "prod_tgt_playerform_blend", "prod_rush_playerform_blend"]], on=list(c.columns), how="left")
    c["prior3_tgt_share_game_mean"] = c["prod_tgt_playerform_blend"]
    c["prior5_tgt_share_game_mean"] = c["prod_tgt_playerform_blend"]
    c["prior3_rush_share_game_mean"] = c["prod_rush_playerform_blend"]
    c["prior5_rush_share_game_mean"] = c["prod_rush_playerform_blend"]
    out = audit_redundancy(h, c)
    assert len(out) == 4
    assert set(out["train_seasons"]) == {"2019-2023"}
    assert set(out["holdout_seasons"]) == {"2024-2025"}
    assert not out["outcomes_read"].any()
    assert not out["sportsbook_read"].any()
    assert (out["holdout_reconstructibility_r2"] > 0.99).all()
    assert set(out["redundancy_disposition"]) == {"HIGHLY_RECONSTRUCTIBLE_REDUNDANT"}
