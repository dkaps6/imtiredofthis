import pandas as pd

from scripts.backtest.build_historical_weather import build_historical_weather


def _schedule():
    return pd.DataFrame([
        {"season": 2025, "week": 1, "team": "IND", "opponent": "MIA", "home_away": "home", "game_id": "g1"},
        {"season": 2025, "week": 1, "team": "MIA", "opponent": "IND", "home_away": "away", "game_id": "g1"},
        {"season": 2025, "week": 1, "team": "BUF", "opponent": "NYJ", "home_away": "home", "game_id": "g2"},
        {"season": 2025, "week": 1, "team": "NYJ", "opponent": "BUF", "home_away": "away", "game_id": "g2"},
    ])


def test_historical_weather_is_one_row_per_game():
    out = build_historical_weather(_schedule())
    assert len(out) == 2
    assert set(out["game_id"]) == {"g1", "g2"}


def test_controlled_venue_is_neutral_without_inventing_temperature():
    out = build_historical_weather(_schedule())
    ind = out.loc[out["home"].eq("IND")].iloc[0]
    assert ind["controlled_environment"] == 1
    assert ind["forecast_ok"] == 1
    assert ind["wind_mph"] == 0
    assert ind["precip_flag"] == 0
    assert pd.isna(ind["temp_f"])
    assert ind["weather_source"] == "venue_architecture_neutral"


def test_outdoor_game_does_not_use_observed_weather_as_fake_pregame_forecast():
    out = build_historical_weather(_schedule())
    buf = out.loc[out["home"].eq("BUF")].iloc[0]
    assert buf["controlled_environment"] == 0
    assert buf["forecast_ok"] == 0
    assert pd.isna(buf["wind_mph"])
    assert pd.isna(buf["precip_flag"])
    assert pd.isna(buf["temp_f"])
    assert buf["weather_source"] == "archived_pregame_forecast_unavailable"
