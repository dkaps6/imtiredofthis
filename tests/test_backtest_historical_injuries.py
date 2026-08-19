import pandas as pd

from scripts.backtest.build_historical_injuries import normalize_historical_injuries
from scripts.backtest.walk_forward import _exact_week


def test_historical_injuries_preserve_week_grain_and_normalize():
    raw = pd.DataFrame([
        {"season": 2025, "week": 4, "full_name": "Player One", "team": "BUF", "report_status": "Questionable", "practice_status": "Limited", "report_primary_injury": "Hamstring"},
        {"season": 2025, "week": 5, "full_name": "Player One", "team": "BUF", "report_status": "Out", "practice_status": "DNP", "report_primary_injury": "Hamstring"},
    ])
    out = normalize_historical_injuries(raw)
    assert len(out) == 2
    assert set(out["week"].astype(int)) == {4, 5}
    assert set(out["source"]) == {"nflverse_historical"}


def test_exact_week_never_backfills_prior_injury_report():
    history = pd.DataFrame([
        {"season": 2025, "week": 4, "player": "Player One", "team": "BUF", "status": "Questionable"},
        {"season": 2025, "week": 6, "player": "Player One", "team": "BUF", "status": "Out"},
    ])
    week5 = _exact_week(history, 2025, 5)
    assert week5.empty
    week6 = _exact_week(history, 2025, 6)
    assert len(week6) == 1
    assert week6.iloc[0]["status"] == "Out"
    assert "season" not in week6.columns and "week" not in week6.columns
