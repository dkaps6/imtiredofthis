import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import (
    _attach_component_projection,
    build_actual_rows,
    build_market_frame,
)
from scripts.backtest.historical_context import HistoricalContextBundle


def _bundle():
    pf = pd.DataFrame([
        {"player": "QB One", "player_clean_key": "qbone", "team": "BUF", "opponent": "MIA", "season": 2025, "week": 8, "position": "QB", "role": "QB1"},
        {"player": "WR One", "player_clean_key": "wrone", "team": "BUF", "opponent": "MIA", "season": 2025, "week": 8, "position": "WR", "role": "LWR"},
    ])
    return HistoricalContextBundle(2025, 8, 2024, pd.DataFrame(), pd.DataFrame(), pf, pf.copy(), pd.DataFrame(), {}, [])


def test_market_frame_uses_pregame_universe_not_results():
    out = build_market_frame(_bundle())
    qb = set(out.loc[out["player"].eq("QB One"), "market"])
    wr = set(out.loc[out["player"].eq("WR One"), "market"])
    assert "pass_yards" in qb
    assert "rec_yards" not in qb
    assert {"rec_yards", "receptions", "rush_yards", "rush_att", "rush_rec_yards"}.issubset(wr)
    assert out["event_id"].nunique() == 1


def test_component_projection_attaches_by_player_team_and_market():
    frame = pd.DataFrame([
        {"player_clean_key": "a", "team": "BUF", "market": "rec_yards"},
        {"player_clean_key": "a", "team": "BUF", "market": "receptions"},
    ])
    pred = pd.DataFrame([{
        "player": "A", "player_clean_key": "a", "team": "BUF",
        "ml_rec_yards": 72.5, "ml_receptions": 5.4,
    }])
    out = _attach_component_projection(frame, pred, "ml")
    assert out.loc[out["market"].eq("rec_yards"), "ml_proj"].iloc[0] == 72.5
    assert out.loc[out["market"].eq("receptions"), "ml_proj"].iloc[0] == 5.4


def test_actual_rows_are_target_week_only_and_joinable_after_prediction():
    logs = pd.DataFrame([
        {"season": 2025, "week": 7, "player": "A", "team": "BUF", "pass_yards": 100, "rush_yards": 1, "rec_yards": 0, "receptions": 0, "rushes": 1},
        {"season": 2025, "week": 8, "player": "A", "team": "BUF", "pass_yards": 250, "rush_yards": 12, "rec_yards": 0, "receptions": 0, "rushes": 3},
        {"season": 2025, "week": 9, "player": "A", "team": "BUF", "pass_yards": 999, "rush_yards": 99, "rec_yards": 0, "receptions": 0, "rushes": 9},
    ])
    out = build_actual_rows(logs, 2025, 8)
    p = out.loc[out["market"].eq("pass_yards"), "actual"].iloc[0]
    r = out.loc[out["market"].eq("rush_rec_yards"), "actual"].iloc[0]
    assert p == 250
    assert r == 12
    assert 999 not in set(out["actual"])


def test_missing_component_projection_stays_missing_not_neutral_placeholder():
    frame = pd.DataFrame([{"player_clean_key": "a", "team": "BUF", "market": "rec_yards"}])
    pred = pd.DataFrame([{"player": "A", "player_clean_key": "a", "team": "BUF", "ml_receptions": 5.0}])
    out = _attach_component_projection(frame, pred, "ml")
    assert np.isnan(out.iloc[0]["ml_proj"])
