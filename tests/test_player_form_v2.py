import numpy as np
import pandas as pd

from scripts.player_form_v2 import _normalize_weekly, _season_totals, _blend
from scripts.utils.player_identity_v3 import attach_historical_identity


def test_route_metrics_are_not_fabricated_from_targets():
    raw = pd.DataFrame([
        {
            "season": 2025,
            "week": 1,
            "player_id": "00-0010001",
            "player_display_name": "Test Receiver",
            "recent_team": "BUF",
            "position": "WR",
            "targets": 8,
            "receptions": 5,
            "receiving_yards": 70,
            "rushing_attempts": 0,
            "rushing_yards": 0,
            "attempts": 0,
            "passing_yards": 0,
        },
        {
            "season": 2025,
            "week": 1,
            "player_id": "00-0010002",
            "player_display_name": "Other Receiver",
            "recent_team": "BUF",
            "position": "WR",
            "targets": 12,
            "receptions": 8,
            "receiving_yards": 100,
            "rushing_attempts": 0,
            "rushing_yards": 0,
            "attempts": 0,
            "passing_yards": 0,
        },
    ])
    logs = _normalize_weekly(raw, 2025)
    assert logs["routes"].isna().all()
    assert logs["route_rate_game"].isna().all()
    assert logs["yprr_game"].isna().all()
    assert np.isclose(logs.loc[logs["player"] == "Test Receiver", "ypt_game"].iloc[0], 70 / 8)
    assert logs.loc[logs["player"] == "Test Receiver", "player_identity_key"].iloc[0] == "gsis:00-0010001"


def test_prior_blend_is_explicit_and_current_takes_over():
    prior = pd.DataFrame([{
        "player_identity_key": "gsis:00-0012345", "player_clean_key": "abc", "games": 17,
        "tgt_share": .20, "rush_share": 0.0, "route_rate": np.nan, "yprr": np.nan,
        "ypt": 8.0, "ypc": np.nan, "ypa": np.nan, "receptions_per_target": .65,
    }])
    current = pd.DataFrame([{
        "player_identity_key": "gsis:00-0012345", "player_clean_key": "abcjr", "games": 4,
        "tgt_share": .30, "rush_share": 0.0, "route_rate": np.nan, "yprr": np.nan,
        "ypt": 10.0, "ypc": np.nan, "ypa": np.nan, "receptions_per_target": .70,
    }])
    universe = pd.DataFrame([{
        "player": "A B Jr.", "player_clean_key": "abjr", "player_identity_key": "gsis:00-0012345",
        "player_id": "00-0012345", "team": "BUF", "opponent": "NYJ", "season": 2026,
        "week": 5, "position": "WR", "role": "WR1",
    }])
    out = _blend(prior, current, universe)
    # current weight = 4 / (4 + 4) = .5.  The deliberately different name keys
    # prove that the blend is keyed by stable player identity, not spelling.
    assert np.isclose(out.loc[0, "tgt_share"], .25)
    assert np.isclose(out.loc[0, "ypt"], 9.0)


def test_season_totals_yprr_requires_real_routes():
    logs = pd.DataFrame([
        {
            "season": 2025, "week": 1, "player_id": "00-0012345", "player_clean_key": "abc",
            "player": "A B", "team": "BUF", "position": "WR", "targets": 10,
            "receptions": 7, "rec_yards": 100, "rushes": 0, "rush_yards": 0,
            "pass_att": 0, "pass_yards": 0, "routes": np.nan, "team_targets": 30,
            "team_rushes": 20, "team_dropbacks": 35, "team_routes": np.nan,
        },
    ])
    logs = attach_historical_identity(logs)
    totals = _season_totals(logs)
    assert totals.loc[0, "player_identity_key"] == "gsis:00-0012345"
    assert np.isnan(totals.loc[0, "route_rate"])
    assert np.isnan(totals.loc[0, "yprr"])
    assert np.isclose(totals.loc[0, "ypt"], 10.0)