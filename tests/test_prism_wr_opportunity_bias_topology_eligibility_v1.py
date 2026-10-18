import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "evaluate_prism_wr_opportunity_bias_topology_eligibility_v1.py"
spec = importlib.util.spec_from_file_location("topology_eligibility", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _rows(topology_class="ANCHOR", season=2024, n=10):
    rows = []
    for i in range(n):
        target = float(i % 5 - 2)
        rows.append({
            "topology_class": topology_class,
            "season": season,
            "week": i + 1,
            "team": "AAA" if i % 2 == 0 else "BBB",
            "player_clean_key": "alpha" if i < 5 else "bravo",
            "team_week_cluster": f"{season}-{i+1}-{'AAA' if i % 2 == 0 else 'BBB'}",
            "position_prior_mean": 0.0,
            "hierarchical_signed_bias": target * 0.8,
            "target_component_error": target,
        })
    return pd.DataFrame(rows)


def test_score_slice_rewards_player_specific_bias_when_closer(monkeypatch):
    monkeypatch.setattr(m, "MIN_ROWS", 1)
    monkeypatch.setattr(m, "MIN_PLAYERS", 1)
    rec, players = m.score_slice(_rows(), "ANCHOR", 2024, 0)
    assert rec["rows"] == 10
    assert rec["players"] == 2
    assert rec["mean_delta_meta_ae"] > 0
    assert rec["candidate_meta_rmse"] < rec["baseline_meta_rmse"]
    assert rec["evaluable_players_ge5"] == 2
    assert len(players) == 2


def test_anchor_and_secondary_are_scored_separately(monkeypatch):
    monkeypatch.setattr(m, "MIN_ROWS", 1)
    monkeypatch.setattr(m, "MIN_PLAYERS", 1)
    x = pd.concat([
        _rows("ANCHOR", 2024, 10),
        _rows("SECONDARY", 2024, 10),
    ], ignore_index=True)
    a, _ = m.score_slice(x, "ANCHOR", 2024, 0)
    s, _ = m.score_slice(x, "SECONDARY", 2024, 100)
    assert a["rows"] == 10
    assert s["rows"] == 10


def test_qualify_cannot_be_rescued_by_pooled_result():
    good = {
        "support_rows_ge200": True,
        "support_players_ge40": True,
        "mean_delta_meta_ae": 1.0,
        "baseline_meta_rmse": 2.0,
        "candidate_meta_rmse": 1.0,
        "baseline_spearman": 0.0,
        "candidate_spearman": 0.2,
        "player_cluster_bootstrap": {"ci95_low": 0.1},
        "team_week_cluster_bootstrap": {"ci95_low": 0.1},
        "equal_player_mean_improvement": 0.5,
        "equal_player_bootstrap": {"ci95_low": 0.1},
        "equal_player_median_improvement": 0.2,
        "equal_player_improved_share": 0.6,
        "equal_player_p10_improvement": -0.1,
        "baseline_equal_player_meta_ae": 1.0,
    }
    gates, ok = m.qualify(dict(good), dict(good))
    assert ok is True

    bad_confirmation = dict(good)
    bad_confirmation["equal_player_improved_share"] = 0.49
    gates, ok = m.qualify(dict(good), bad_confirmation)
    assert ok is False
    assert gates["confirmation_player_improved_share_majority"] is False
