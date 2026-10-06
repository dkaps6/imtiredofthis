import pandas as pd

from scripts.research.audit_opportunity_state_conflict_uncertainty_v1 import (
    classify,
    score_cell,
)


def _panel(season: int, positive: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(120):
        conflict = (i + 1) / 1200.0
        bayes = 0.40
        # Make error rise with conflict for the positive fixture.
        err = conflict * (2.0 if positive else -2.0)
        actual = bayes + (err if positive else abs(err))
        if not positive:
            actual = bayes + (0.25 - conflict)
        playerform = bayes + conflict
        rows.append({
            "season": season,
            "target_week": 2 + (i % 16),
            "player_identity_key": f"p{i%30}",
            "player": f"Player {i%30}",
            "team": "IND",
            "position": "RB",
            "metric": "rush_share",
            "prior_games": 8,
            "current_games": 2,
            "playerform_value": playerform,
            "bayes_value": bayes,
            "actual_value": actual,
        })
    return pd.DataFrame(rows)


def test_score_cell_detects_positive_conflict_uncertainty():
    q = _panel(2024, positive=True)
    r = score_cell(q, 2024, "RB", "rush_share")
    assert r["cell_pass"] is True
    assert r["spearman_conflict_vs_bayes_ae"] > 0
    assert r["q4_minus_q1_bayes_ae"] > 0
    assert r["bootstrap_ci_low"] > 0


def test_classify_requires_both_seasons():
    cells = pd.DataFrame([
        {"season": 2024, "position": "RB", "metric": "rush_share", "cell_pass": True},
        {"season": 2025, "position": "RB", "metric": "rush_share", "cell_pass": True},
        {"season": 2024, "position": "WR", "metric": "tgt_share", "cell_pass": False},
        {"season": 2025, "position": "WR", "metric": "tgt_share", "cell_pass": False},
        {"season": 2024, "position": "TE", "metric": "tgt_share", "cell_pass": False},
        {"season": 2025, "position": "TE", "metric": "tgt_share", "cell_pass": False},
    ])
    out = classify(cells)
    assert out["disposition"] == "OPPORTUNITY_STATE_CONFLICT_UNCERTAINTY_SIGNAL_CONFIRMED"
    rb = [x for x in out["per_metric"] if x["position"] == "RB"][0]
    assert rb["disposition"] == "STATE_CONFLICT_UNCERTAINTY_REPLICATED"
