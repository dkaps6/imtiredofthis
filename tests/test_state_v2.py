import numpy as np
import pandas as pd

from scripts.modeling.state_v2 import apply_state_to_metrics, build_state_predictions, train_state_model


def _logs():
    rows = []
    players = [("QB A", "BUF", "QB"), ("QB B", "MIA", "QB"), ("WR A", "BUF", "WR"), ("WR B", "MIA", "WR"), ("RB A", "BUF", "RB"), ("RB B", "MIA", "RB")]
    for week in range(1, 18):
        for i, (player, team, pos) in enumerate(players):
            wave = (week + i) % 3
            rows.append({
                "season": 2025, "week": week, "player": player, "team": team, "position": pos,
                "pass_yards": (190 + 55 * wave + i) if pos == "QB" else 0,
                "rush_yards": 10 + 22 * wave + (25 if pos == "RB" else 0),
                "rec_yards": 8 + 28 * wave + (30 if pos == "WR" else 0),
                "receptions": 1 + 2 * wave + (2 if pos == "WR" else 0),
                "rushes": 2 + 5 * wave + (8 if pos == "RB" else 0),
            })
    return pd.DataFrame(rows)


def _consensus():
    return pd.DataFrame([
        {"player": "QB A", "team": "BUF", "position": "QB"},
        {"player": "WR A", "team": "BUF", "position": "WR"},
        {"player": "RB A", "team": "BUF", "position": "RB"},
    ])


def test_state_model_is_real_transition_model_with_normalized_probabilities():
    bundle, pred = build_state_predictions(_logs(), _consensus(), 2026, 1)
    assert bundle.specs
    spec = bundle.specs[("pass_yards", "QB")]
    for src, probs in spec.transition_probs.items():
        assert src in {"LOW", "MID", "HIGH"}
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert all(v > 0 for v in probs.values())
    qb = pred.loc[pred.player.eq("QB A")].iloc[0]
    assert qb.state_available == 1
    assert np.isfinite(qb.state_pass_yards)
    assert qb.state_pass_yards_current_state in {"LOW", "MID", "HIGH"}


def test_target_cutoff_excludes_target_week_and_future_games():
    logs = _logs()
    future = pd.DataFrame([{
        "season": 2026, "week": 1, "player": "QB A", "team": "BUF", "position": "QB",
        "pass_yards": 9999, "rush_yards": 999, "rec_yards": 0, "receptions": 0, "rushes": 40,
    }])
    a = train_state_model(logs, 2026, 1)
    b = train_state_model(pd.concat([logs, future], ignore_index=True), 2026, 1)
    sa, sb = a.specs[("pass_yards", "QB")], b.specs[("pass_yards", "QB")]
    assert sa.low_cut == sb.low_cut
    assert sa.high_cut == sb.high_cut
    assert sa.transition_probs == sb.transition_probs


def test_state_projection_attaches_by_market_without_using_sportsbook_line():
    _, pred = build_state_predictions(_logs(), _consensus(), 2026, 1)
    metrics = pd.DataFrame([
        {"player": "QB A", "team": "BUF", "market": "player_passing_yards", "line": 100.5},
        {"player": "QB A", "team": "BUF", "market": "player_passing_yards", "line": 500.5},
    ])
    out = apply_state_to_metrics(metrics, pred)
    assert out.state_applied.sum() == 2
    assert out.state_proj.nunique() == 1
