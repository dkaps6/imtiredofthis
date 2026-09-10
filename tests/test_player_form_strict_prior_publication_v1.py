import pandas as pd

from scripts.run_player_form_current_roles_v1 import strict_prior_logs


def test_strict_prior_publication_excludes_same_and_future_week_rows():
    logs = pd.DataFrame(
        [
            {"season": 2025, "week": 18, "player": "Prior", "team": "AAA"},
            {"season": 2026, "week": 0, "player": "Earlier", "team": "BBB"},
            {"season": 2026, "week": 1, "player": "SameWeek", "team": "CCC"},
            {"season": 2026, "week": 2, "player": "Future", "team": "DDD"},
        ]
    )
    out = strict_prior_logs(logs, season=2026, prior_season=2025, week=1)
    assert list(out["player"]) == ["Prior", "Earlier"]
    assert not ((out.season.eq(2026)) & (out.week.ge(1))).any()


def test_strict_prior_publication_drops_unrelated_seasons():
    logs = pd.DataFrame(
        [
            {"season": 2024, "week": 18, "player": "TooOld", "team": "AAA"},
            {"season": 2025, "week": 1, "player": "Prior", "team": "BBB"},
        ]
    )
    out = strict_prior_logs(logs, season=2026, prior_season=2025, week=1)
    assert list(out["player"]) == ["Prior"]
