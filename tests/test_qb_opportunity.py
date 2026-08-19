import pandas as pd
import pytest

from scripts.backtest.qb_opportunity import add_qb_opportunity


def test_qb_opportunity_selects_recent_team_primary_and_not_every_qb():
    player_form = pd.DataFrame([
        {"player": "Starter", "player_clean_key": "starter", "team": "BUF", "position": "QB"},
        {"player": "Backup", "player_clean_key": "backup", "team": "BUF", "position": "QB"},
        {"player": "WR", "player_clean_key": "wr", "team": "BUF", "position": "WR"},
    ])
    universe = pd.DataFrame([
        {"player": "Starter", "player_clean_key": "starter", "team": "BUF", "position": "QB", "role": ""},
        {"player": "Backup", "player_clean_key": "backup", "team": "BUF", "position": "QB", "role": ""},
        {"player": "WR", "player_clean_key": "wr", "team": "BUF", "position": "WR", "role": ""},
    ])
    history = pd.DataFrame([
        {"season": 2025, "week": 5, "player_clean_key": "starter", "team": "BUF", "pass_att": 34, "team_dropbacks": 36},
        {"season": 2025, "week": 6, "player_clean_key": "starter", "team": "BUF", "pass_att": 32, "team_dropbacks": 34},
        {"season": 2024, "week": 17, "player_clean_key": "backup", "team": "CHI", "pass_att": 30, "team_dropbacks": 32},
    ])
    out = add_qb_opportunity(player_form, universe, history, season=2025, prior_season=2024)
    starter = out.loc[out["player_clean_key"].eq("starter")].iloc[0]
    backup = out.loc[out["player_clean_key"].eq("backup")].iloc[0]
    assert starter["qb_projection_eligible"] == 1
    assert starter["qb_pass_att_share"] == pytest.approx((34 / 36 + 32 / 34) / 2)
    assert backup["qb_projection_eligible"] == 0
    assert backup["qb_pass_att_share"] == 0.0


def test_explicit_qb1_role_beats_history_without_using_target_week_results():
    player_form = pd.DataFrame([
        {"player": "Rookie", "player_clean_key": "rookie", "team": "NYG", "position": "QB"},
        {"player": "Veteran", "player_clean_key": "vet", "team": "NYG", "position": "QB"},
    ])
    universe = pd.DataFrame([
        {"player": "Rookie", "player_clean_key": "rookie", "team": "NYG", "position": "QB", "role": "QB1"},
        {"player": "Veteran", "player_clean_key": "vet", "team": "NYG", "position": "QB", "role": "QB2"},
    ])
    history = pd.DataFrame([
        {"season": 2024, "week": 18, "player_clean_key": "vet", "team": "SEA", "pass_att": 38, "team_dropbacks": 40},
    ])
    out = add_qb_opportunity(player_form, universe, history, season=2025, prior_season=2024)
    rookie = out.loc[out["player_clean_key"].eq("rookie")].iloc[0]
    assert rookie["qb_projection_eligible"] == 1
    assert rookie["qb_role_source"] == "depth_role"
    assert rookie["qb_pass_att_share"] == pytest.approx(0.95)


def test_no_history_still_selects_exactly_one_qb_per_team():
    player_form = pd.DataFrame([
        {"player": "A", "player_clean_key": "a", "team": "TEN", "position": "QB"},
        {"player": "B", "player_clean_key": "b", "team": "TEN", "position": "QB"},
    ])
    universe = player_form.assign(role="")
    out = add_qb_opportunity(player_form, universe, pd.DataFrame(columns=["season", "week", "player_clean_key", "team", "pass_att", "team_dropbacks"]), season=2025, prior_season=2024)
    assert int(out["qb_projection_eligible"].sum()) == 1
    assert sorted(out["qb_pass_att_share"].tolist()) == [0.0, 0.95]
