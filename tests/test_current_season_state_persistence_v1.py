import pandas as pd
import pytest

from scripts.research import current_season_state_persistence_v1 as mod


def _row(*, season, week, rushes, team_rushes, targets, team_targets, rush_yards, rec_yards=0.0):
    ypc = rush_yards / rushes if rushes else float("nan")
    tgt_share = targets / team_targets if team_targets else float("nan")
    rush_share = rushes / team_rushes if team_rushes else float("nan")
    ypt = rec_yards / targets if targets else float("nan")
    return {
        "season": season,
        "week": week,
        "player_identity_key": "gsis:rb1",
        "player": "Example Back",
        "team": "CHI",
        "position": "RB",
        "targets": float(targets),
        "receptions": float(targets),
        "rec_yards": float(rec_yards),
        "rushes": float(rushes),
        "rush_yards": float(rush_yards),
        "pass_att": 0.0,
        "pass_yards": 0.0,
        "team_targets": float(team_targets),
        "team_rushes": float(team_rushes),
        "tgt_share_game": float(tgt_share),
        "rush_share_game": float(rush_share),
        "route_rate_game": float("nan"),
        "ypt_game": float(ypt),
        "ypc_game": float(ypc),
        "ypa_game": float("nan"),
        "catch_rate_game": 1.0 if targets else float("nan"),
        "yprr_game": float("nan"),
    }


def test_week2_uses_only_week1_current_state_and_exact_blend4():
    logs = pd.DataFrame([
        _row(season=2024, week=1, rushes=10, team_rushes=20, targets=2, team_targets=10, rush_yards=50),
        _row(season=2025, week=1, rushes=14, team_rushes=20, targets=4, team_targets=10, rush_yards=84),
        _row(season=2025, week=2, rushes=16, team_rushes=20, targets=3, team_targets=10, rush_yards=112),
    ])

    panel = mod.build_persistence_panel(logs)
    q = panel.loc[
        panel["season"].eq(2025)
        & panel["target_week"].eq(2)
        & panel["position"].eq("RB")
        & panel["metric"].eq("rush_share")
    ]
    assert len(q) == 1
    r = q.iloc[0]
    assert r["prior_value"] == pytest.approx(0.5)
    assert r["current_value"] == pytest.approx(0.7)
    assert r["actual_value"] == pytest.approx(0.8)
    assert r["current_games"] == 1
    assert r["w_current_blend4"] == pytest.approx(1.0 / 5.0)
    assert r["blend4_value"] == pytest.approx(0.54)


def test_week3_current_state_includes_weeks1_and2_but_not_target():
    logs = pd.DataFrame([
        _row(season=2024, week=1, rushes=10, team_rushes=20, targets=2, team_targets=10, rush_yards=50),
        _row(season=2025, week=1, rushes=10, team_rushes=20, targets=2, team_targets=10, rush_yards=50),
        _row(season=2025, week=2, rushes=18, team_rushes=20, targets=4, team_targets=10, rush_yards=108),
        _row(season=2025, week=3, rushes=20, team_rushes=20, targets=5, team_targets=10, rush_yards=140),
    ])

    panel = mod.build_persistence_panel(logs)
    q = panel.loc[
        panel["season"].eq(2025)
        & panel["target_week"].eq(3)
        & panel["metric"].eq("rush_share")
    ]
    assert len(q) == 1
    r = q.iloc[0]
    assert r["current_games"] == 2
    assert r["current_value"] == pytest.approx(28.0 / 40.0)
    assert r["actual_value"] == pytest.approx(1.0)
    assert r["w_current_blend4"] == pytest.approx(2.0 / 6.0)


def test_duplicate_target_identity_fails_closed():
    base = _row(season=2025, week=2, rushes=10, team_rushes=20, targets=1, team_targets=10, rush_yards=50)
    logs = pd.DataFrame([
        _row(season=2024, week=1, rushes=10, team_rushes=20, targets=1, team_targets=10, rush_yards=50),
        _row(season=2025, week=1, rushes=10, team_rushes=20, targets=1, team_targets=10, rush_yards=50),
        base,
        dict(base),
    ])
    with pytest.raises(RuntimeError, match="duplicate target-week player identity"):
        mod.build_persistence_panel(logs)


def test_game_count_bucket_contract():
    assert mod._game_bucket(1) == "1"
    assert mod._game_bucket(4) == "4"
    assert mod._game_bucket(5) == "5-8"
    assert mod._game_bucket(8) == "5-8"
    assert mod._game_bucket(9) == "9+"
