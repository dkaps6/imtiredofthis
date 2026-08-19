import numpy as np
import pandas as pd

import scripts.modeling.ml_v2 as ml


def _logs():
    rows = []
    players = [
        ("QB One", "BUF", "QB"),
        ("RB One", "BUF", "RB"),
        ("WR One", "BUF", "WR"),
        ("WR Two", "MIA", "WR"),
        ("TE One", "MIA", "TE"),
    ]
    for week in range(1, 13):
        for i, (player, team, pos) in enumerate(players):
            targets = 0 if pos == "QB" else 3 + i + (week % 3)
            receptions = int(round(targets * 0.65))
            rushes = (5 + week) if pos == "RB" else (3 + week % 4) if pos == "QB" else (1 if pos == "WR" and week % 4 == 0 else 0)
            pass_att = 28 + week if pos == "QB" else 0
            rows.append({
                "season": 2025,
                "week": week,
                "game_id": f"2025_{week:02d}_{team}",
                "player": player,
                "player_clean_key": "".join(ch.lower() for ch in player if ch.isalnum()),
                "team": team,
                "opponent": "MIA" if team == "BUF" else "BUF",
                "position": pos,
                "targets": targets,
                "receptions": receptions,
                "rec_yards": receptions * (9 + (week % 4)),
                "rushes": rushes,
                "rush_yards": rushes * (4.0 + 0.1 * (week % 5)),
                "pass_att": pass_att,
                "pass_yards": pass_att * (7.0 + 0.05 * week),
                "tgt_share_game": targets / 30.0 if targets else 0.0,
                "rush_share_game": rushes / 25.0 if rushes else 0.0,
                "ypt_game": (receptions * (9 + (week % 4)) / targets) if targets else np.nan,
                "ypc_game": (rushes * (4.0 + 0.1 * (week % 5)) / rushes) if rushes else np.nan,
                "ypa_game": (pass_att * (7.0 + 0.05 * week) / pass_att) if pass_att else np.nan,
                "catch_rate_game": receptions / targets if targets else np.nan,
            })
    return pd.DataFrame(rows)


def _consensus():
    return pd.DataFrame([
        {"player": "QB One", "player_clean_key": "qbone", "team": "BUF", "season": 2026, "week": 1, "position": "QB"},
        {"player": "RB One", "player_clean_key": "rbone", "team": "BUF", "season": 2026, "week": 1, "position": "RB"},
        {"player": "WR One", "player_clean_key": "wrone", "team": "BUF", "season": 2026, "week": 1, "position": "WR"},
        {"player": "WR Two", "player_clean_key": "wrtwo", "team": "MIA", "season": 2026, "week": 1, "position": "WR"},
        {"player": "TE One", "player_clean_key": "teone", "team": "MIA", "season": 2026, "week": 1, "position": "TE"},
    ])


def test_training_features_are_shifted_and_do_not_use_same_game_result():
    logs = _logs()
    a = ml.build_training_frame(logs, 2026, 1)
    key = "wrone"
    before = a.loc[(a.player_clean_key.eq(key)) & a.week.eq(6), ["prev_rec_yards", "mean3_rec_yards"]].iloc[0]

    changed = logs.copy()
    changed.loc[(changed.player_clean_key.eq(key)) & changed.week.eq(6), "rec_yards"] = 9999.0
    b = ml.build_training_frame(changed, 2026, 1)
    after = b.loc[(b.player_clean_key.eq(key)) & b.week.eq(6), ["prev_rec_yards", "mean3_rec_yards"]].iloc[0]
    pd.testing.assert_series_equal(before, after)


def test_real_models_train_and_predict_without_sportsbook_lines(monkeypatch):
    for target in ml.MIN_TRAIN_ROWS:
        monkeypatch.setitem(ml.MIN_TRAIN_ROWS, target, 3)
    bundle, pred = ml.build_and_train(_logs(), _consensus(), 2026, 1)
    assert "pass_yards" in bundle.models
    assert "rec_yards" in bundle.models
    assert len(pred) == 5
    assert pred["ml_rec_yards"].notna().all()
    assert pred.loc[pred.position.eq("QB"), "ml_pass_yards"].notna().all()
    assert "line" not in ml.FEATURE_COLUMNS
    assert "vegas_line" not in ml.FEATURE_COLUMNS


def test_market_attachment_preserves_ml_as_parallel_projection(monkeypatch):
    for target in ml.MIN_TRAIN_ROWS:
        monkeypatch.setitem(ml.MIN_TRAIN_ROWS, target, 3)
    _, pred = ml.build_and_train(_logs(), _consensus(), 2026, 1)
    metrics = pd.DataFrame([
        {"player": "WR One", "player_clean_key": "wrone", "team": "BUF", "market": "player_receiving_yards", "line": 70.5},
        {"player": "QB One", "player_clean_key": "qbone", "team": "BUF", "market": "player_passing_yards", "line": 250.5},
        {"player": "WR One", "player_clean_key": "wrone", "team": "BUF", "market": "player_anytime_td", "line": 0.5},
    ])
    out = ml.apply_ml_to_metrics(metrics, pred)
    assert out.loc[0, "ml_applied"] == 1
    assert out.loc[1, "ml_applied"] == 1
    assert out.loc[2, "ml_applied"] == 0  # ATD remains outside ML v2 until scoring-label work is added.
    assert out.loc[0, "ml_proj"] > 0
    assert out.loc[1, "ml_proj"] > 0
