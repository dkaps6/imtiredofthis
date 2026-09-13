from pathlib import Path

import pandas as pd
import pytest

import scripts.slate_universe_v2 as slate


class DummyPF:
    def __init__(self, roles: pd.DataFrame, data_dir: Path):
        self._roles = roles
        self.DATA = data_dir

    def _load_roles(self) -> pd.DataFrame:
        return self._roles.copy()


def _roles(teams):
    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "player": f"{team} Quarterback",
                "player_clean_key": f"{team.lower()}quarterback",
                "role": "QB1",
                "position": "QB",
            }
        )
    return pd.DataFrame(rows)


def _schedule(pairs):
    rows = []
    for away, home in pairs:
        rows.extend(
            [
                {"season": 2026, "week": 1, "team": away, "opponent": home},
                {"season": 2026, "week": 1, "team": home, "opponent": away},
            ]
        )
    return pd.DataFrame(rows)


def test_two_team_certified_current_slate_is_valid_even_below_legacy_floor(tmp_path, monkeypatch):
    """A late-Sunday one-game current slate is valid when certification says so.

    This is the exact class of case that the old hard-coded 24-team floor
    incorrectly rejected and that caused PlayerForm to be widened back to all
    32 teams.
    """
    roles = _roles(["KC", "LV"])
    schedule = _schedule([("KC", "LV")])
    monkeypatch.setattr(slate, "expected_current_teams", lambda: {"KC", "LV"})

    out = slate.build_slate_universe(
        DummyPF(roles, tmp_path),
        lambda: schedule,
        2026,
        1,
        live_odds_enabled=True,
    )

    assert set(out["team"]) == {"KC", "LV"}
    assert out["team"].nunique() == 2


def test_explicit_current_scope_rejects_extra_team(tmp_path, monkeypatch):
    roles = _roles(["KC", "LV", "DEN"])
    schedule = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "team": "KC", "opponent": "LV"},
            {"season": 2026, "week": 1, "team": "LV", "opponent": "KC"},
            {"season": 2026, "week": 1, "team": "DEN", "opponent": "LAC"},
        ]
    )
    monkeypatch.setattr(slate, "expected_current_teams", lambda: {"KC", "LV"})

    with pytest.raises(RuntimeError, match="extra=\['DEN'\]"):
        slate.build_slate_universe(
            DummyPF(roles, tmp_path),
            lambda: schedule,
            2026,
            1,
            live_odds_enabled=True,
        )


def test_explicit_current_scope_rejects_missing_team(tmp_path, monkeypatch):
    roles = _roles(["KC"])
    schedule = _schedule([("KC", "LV")])
    monkeypatch.setattr(slate, "expected_current_teams", lambda: {"KC", "LV"})

    with pytest.raises(RuntimeError, match="missing=\['LV'\]"):
        slate.build_slate_universe(
            DummyPF(roles, tmp_path),
            lambda: schedule,
            2026,
            1,
            live_odds_enabled=True,
        )


def test_legacy_mode_keeps_historical_corruption_floor(tmp_path, monkeypatch):
    roles = _roles(["KC", "LV"])
    schedule = _schedule([("KC", "LV")])
    monkeypatch.setattr(slate, "expected_current_teams", lambda: None)

    with pytest.raises(RuntimeError, match="implausible team coverage: 2"):
        slate.build_slate_universe(
            DummyPF(roles, tmp_path),
            lambda: schedule,
            2026,
            1,
            live_odds_enabled=False,
        )
