import pandas as pd
import pytest

from scripts.backtest.validate_historical_inputs import validate_schedule, validate_team_history, validate_universe


def _schedule():
    return pd.DataFrame([
        {"season": 2025, "week": 1, "team": "BUF", "opponent": "MIA"},
        {"season": 2025, "week": 1, "team": "MIA", "opponent": "BUF"},
    ])


def test_schedule_requires_reciprocal_matchups():
    good = _schedule()
    out = validate_schedule(good, 2025, [1])
    assert out["weeks"] == 1
    bad = good.iloc[:1].copy()
    with pytest.raises(RuntimeError, match="non-reciprocal"):
        validate_schedule(bad, 2025, [1])


def test_team_history_requires_both_seasons():
    frame = pd.DataFrame([
        {"season": 2024, "week": 1, "team": f"T{i:02d}"} for i in range(32)
    ] + [
        {"season": 2025, "week": 1, "team": f"T{i:02d}"} for i in range(32)
    ])
    # Synthetic team codes do not canonicalize to blank, so coverage is what is tested.
    out = validate_team_history(frame, 2025, 2024)
    assert out[2024]["teams"] == 32
    assert out[2025]["teams"] == 32


def test_universe_matches_schedule_and_rejects_defense():
    sched = _schedule()
    players = []
    for team, opp in (("BUF", "MIA"), ("MIA", "BUF")):
        for i in range(160):
            players.append({
                "player": f"{team} Player {i}", "team": team, "opponent": opp,
                "position": "WR" if i % 2 else "RB", "season": 2025, "week": 1,
            })
    frame = pd.DataFrame(players)
    out = validate_universe(frame, sched, 2025, 1)
    assert out["players"] == 320
    bad = frame.copy()
    bad.loc[0, "position"] = "CB"
    with pytest.raises(RuntimeError, match="non-offensive positions"):
        validate_universe(bad, sched, 2025, 1)
