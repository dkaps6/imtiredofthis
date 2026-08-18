import sys
import types

import pandas as pd

from scripts.player_stats_loader_v2 import _load_nflreadpy, _validate_weekly


def test_nflreadpy_loader_uses_summary_level_not_stat_type(monkeypatch):
    calls = {}

    def fake_load_player_stats(*, seasons, summary_level="week"):
        calls["seasons"] = seasons
        calls["summary_level"] = summary_level
        return pd.DataFrame([
            {"season": 2025, "week": 1, "player_name": "A Player", "recent_team": "BUF", "targets": 4},
            {"season": 2025, "week": 2, "player_name": "A Player", "recent_team": "BUF", "targets": 6},
        ])

    fake_module = types.SimpleNamespace(load_player_stats=fake_load_player_stats)
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_module)

    out = _load_nflreadpy(2025)
    assert calls == {"seasons": [2025], "summary_level": "week"}
    assert len(out) == 2
    assert out["week"].tolist() == [1, 2]


def test_weekly_validator_rejects_season_level_shape():
    bad = pd.DataFrame([{"season": 2025, "player_name": "A Player"}])
    try:
        _validate_weekly(bad, 2025, "test")
    except RuntimeError as exc:
        assert "week" in str(exc)
    else:
        raise AssertionError("weekly validator accepted data without week column")
