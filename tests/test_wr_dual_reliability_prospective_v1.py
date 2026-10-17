import numpy as np
import pandas as pd
import pytest

from scripts.research.lock_wr_dual_reliability_prospective_v1 import build_lock


def cohort():
    return pd.DataFrame([
        {"season": 2026, "week": 7, "event_id": "g7", "team": "A", "opponent": "B",
         "player": "P1", "player_clean_key": "p1", "wr_r15_projection": 60.0},
        {"season": 2026, "week": 7, "event_id": "g7", "team": "A", "opponent": "B",
         "player": "P2", "player_clean_key": "p2", "wr_r15_projection": 55.0},
        {"season": 2026, "week": 7, "event_id": "g7", "team": "B", "opponent": "A",
         "player": "P3", "player_clean_key": "p3", "wr_r15_projection": 50.0},
        {"season": 2026, "week": 7, "event_id": "g7", "team": "B", "opponent": "A",
         "player": "P4", "player_clean_key": "p4", "wr_r15_projection": 45.0},
    ])


def histories(max_week=6):
    opp, eff = [], []
    values = {
        "p1": (1.0, 4.0),
        "p2": (2.0, 3.0),
        "p3": (3.0, 2.0),
        "p4": (4.0, 1.0),
    }
    for pk, (o, e) in values.items():
        for week in range(1, max_week + 1):
            opp.append({
                "season": 2026, "week": week, "player_clean_key": pk,
                "opportunity_error": o,
            })
            eff.append({
                "season": 2026, "week": week, "player_clean_key": pk,
                "efficiency_error": e,
            })
    return pd.DataFrame(opp), pd.DataFrame(eff)


def test_dual_lock_uses_fixed_half_half_rank_and_preserves_projection():
    opp, eff = histories()
    rows, result = build_lock(cohort(), opp, eff, target_week=7)
    assert result["supported_wr"] == 4
    assert result["week6_composite_backfilled"] is False
    assert result["target_week_outcomes_read"] == 0
    p1 = rows.loc[rows.player_clean_key.eq("p1")].iloc[0]
    p4 = rows.loc[rows.player_clean_key.eq("p4")].iloc[0]
    # Opp ranks ascend p1->p4 while efficiency ranks descend p1->p4.
    # Their fixed 50/50 average is therefore identical.
    assert np.isclose(p1.dual_fragility_score, p4.dual_fragility_score)
    assert np.isclose(p1.wr_r15_projection, 60.0)


def test_dual_lock_fails_closed_when_week6_history_is_not_complete():
    opp, eff = histories(max_week=5)
    with pytest.raises(RuntimeError, match="require 6"):
        build_lock(cohort(), opp, eff, target_week=7)


def test_dual_lock_rejects_target_week_outcomes_in_history():
    opp, eff = histories()
    opp = pd.concat([
        opp,
        pd.DataFrame([{
            "season": 2026, "week": 7, "player_clean_key": "p1",
            "opportunity_error": 1.0,
        }]),
    ], ignore_index=True)
    with pytest.raises(RuntimeError, match="target/future outcomes"):
        build_lock(cohort(), opp, eff, target_week=7)


def test_dual_lock_requires_four_prior_games_per_component():
    opp, eff = histories()
    opp = opp.loc[~((opp.player_clean_key == "p4") & (opp.week > 3))].copy()
    # Keep global max week 6 using other players; p4 itself has only 3 opp games.
    rows, result = build_lock(cohort(), opp, eff, target_week=7)
    p4 = rows.loc[rows.player_clean_key.eq("p4")].iloc[0]
    assert bool(p4.supported) is False
    assert p4.dual_trust_bucket == "UNSUPPORTED"
    assert result["supported_wr"] == 3
